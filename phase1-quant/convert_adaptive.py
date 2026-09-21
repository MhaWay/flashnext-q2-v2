#!/usr/bin/env python3
"""Generate adaptive-Q2 base sidecars and selected Q4 precision islands."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402
from safetensors.torch import save_file  # noqa: E402

from calibration import load_layer_calibration  # noqa: E402
from source_reader import (HIDDEN, INTER, NUM_EXPERTS, W13_OUT, build_source_map,
                           load_matrix, load_weight_map)  # noqa: E402
from torch_codec import fwht128, quantize_adaptive_q2, quantize_symmetric_q4  # noqa: E402


FORMAT = "flashnext-q2-adaptive-v1"
ISLAND_FORMAT = "flashnext-q4-islands-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_save(state: dict[str, torch.Tensor], destination: Path, metadata: dict[str, str]) -> None:
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    save_file({key: value.contiguous() for key, value in state.items()}, str(temporary), metadata=metadata)
    os.replace(temporary, destination)


def load_precision_map(path: Path) -> tuple[dict[int, set[int]], str]:
    data = json.loads(path.read_text())
    if data.get("format") != "flashnext-q2-precision-map-v1":
        raise ValueError(f"unsupported precision map format {data.get('format')!r}")
    result = {int(layer): {int(expert) for expert in experts}
              for layer, experts in data.get("experts_by_layer", {}).items()}
    return result, sha256(path)


def base_valid(path: Path, layer: int, spec_sha: str, calibration_sha: str,
               precision_sha: str) -> bool:
    if not path.is_file():
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            md = handle.metadata() or {}
            expected_md = {
                "format": FORMAT,
                "layer": str(layer),
                "spec_sha256": spec_sha,
                "calibration_sha256": calibration_sha,
                "precision_map_sha256": precision_sha,
            }
            if any(md.get(key) != value for key, value in expected_md.items()):
                return False
            shapes = {
                "w13_weight": (NUM_EXPERTS, W13_OUT, HIDDEN // 4),
                "w2_weight": (NUM_EXPERTS, HIDDEN, INTER // 4),
                "w13_weight_scale_inv": (NUM_EXPERTS, W13_OUT, HIDDEN // 128),
                "w2_weight_scale_inv": (NUM_EXPERTS, HIDDEN, INTER // 128),
                "w13_alpha_index": (NUM_EXPERTS, W13_OUT, HIDDEN // 128),
                "w2_alpha_index": (NUM_EXPERTS, HIDDEN, INTER // 128),
                "alpha_table": (6,),
            }
            return all(key in handle.keys() and tuple(handle.get_slice(key).get_shape()) == shape
                       for key, shape in shapes.items())
    except Exception:
        return False


def island_valid(path: Path, layer: int, selected: list[int], spec_sha: str,
                 calibration_sha: str, precision_sha: str) -> bool:
    if not selected:
        return not path.exists()
    if not path.is_file():
        return False
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            md = handle.metadata() or {}
            expected_md = {
                "format": ISLAND_FORMAT,
                "layer": str(layer),
                "spec_sha256": spec_sha,
                "calibration_sha256": calibration_sha,
                "precision_map_sha256": precision_sha,
                "selected_experts": ",".join(map(str, selected)),
            }
            if any(md.get(key) != value for key, value in expected_md.items()):
                return False
            n = len(selected)
            shapes = {
                "expert_ids": (n,),
                "w13_weight_q4": (n, W13_OUT, HIDDEN // 2),
                "w2_weight_q4": (n, HIDDEN, INTER // 2),
                "w13_weight_scale_inv_q4": (n, W13_OUT, HIDDEN // 128),
                "w2_weight_scale_inv_q4": (n, HIDDEN, INTER // 128),
            }
            if not all(key in handle.keys() and tuple(handle.get_slice(key).get_shape()) == shape
                       for key, shape in shapes.items()):
                return False
            ids = handle.get_tensor("expert_ids").tolist()
            return ids == selected
    except Exception:
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--calibration-dir", required=True, type=Path)
    ap.add_argument("--precision-map", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--spec", type=Path, default=HERE / "specs/adaptive-q2-v1.json")
    ap.add_argument("--start-layer", type=int, default=0)
    ap.add_argument("--end-layer", type=int, default=48)
    ap.add_argument("--minimum-routes", type=int, default=32)
    ap.add_argument("--gpu-memory-fraction", type=float, default=0.15)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, 0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    island_root = args.out_dir / "islands"
    island_root.mkdir(exist_ok=True)
    spec = json.loads(args.spec.read_text())
    spec_sha = sha256(args.spec)
    alphas = torch.tensor(spec["q2"]["alpha_table"], dtype=torch.float32, device="cuda")
    precision_map, precision_sha = load_precision_map(args.precision_map)
    weight_map = load_weight_map(args.snapshot)
    source = build_source_map(weight_map)
    source_sha = sha256(args.snapshot / "model.safetensors.index.json")
    manifest_layers = []

    for layer in range(max(0, args.start_layer), min(48, args.end_layer)):
        started = time.time()
        calibration_path = args.calibration_dir / f"layer-{layer:02d}.safetensors"
        calibration_sha = sha256(calibration_path)
        base_path = args.out_dir / f"layer-{layer:02d}.safetensors"
        island_path = island_root / f"layer-{layer:02d}.safetensors"
        selected = sorted(precision_map.get(layer, set()))
        if (base_valid(base_path, layer, spec_sha, calibration_sha, precision_sha)
                and island_valid(island_path, layer, selected, spec_sha,
                                 calibration_sha, precision_sha)
                and not args.force):
            print(f"L{layer:02d}: resume valid adaptive sidecar", flush=True)
            manifest_layers.append({
                "layer": layer, "base": base_path.name, "base_sha256": sha256(base_path),
                "base_bytes": base_path.stat().st_size,
                "island": str(island_path.relative_to(args.out_dir)) if selected else None,
                "island_sha256": sha256(island_path) if selected else None,
                "island_bytes": island_path.stat().st_size if selected else 0,
                "selected_experts": selected, "resumed": True,
            })
            continue
        calibration = load_layer_calibration(args.calibration_dir, layer, args.minimum_routes)
        print(f"L{layer:02d}: allocate adaptive Q2 buffers; Q4 islands={len(selected)}", flush=True)
        w13p = torch.empty((NUM_EXPERTS, W13_OUT, HIDDEN // 4), dtype=torch.uint8)
        w2p = torch.empty((NUM_EXPERTS, HIDDEN, INTER // 4), dtype=torch.uint8)
        s13 = torch.empty((NUM_EXPERTS, W13_OUT, HIDDEN // 128), dtype=torch.float16)
        s2 = torch.empty((NUM_EXPERTS, HIDDEN, INTER // 128), dtype=torch.float16)
        a13 = torch.empty_like(s13, dtype=torch.uint8)
        a2 = torch.empty_like(s2, dtype=torch.uint8)
        selected_position = {expert: index for index, expert in enumerate(selected)}
        q4_w13 = torch.empty((len(selected), W13_OUT, HIDDEN // 2), dtype=torch.uint8)
        q4_w2 = torch.empty((len(selected), HIDDEN, INTER // 2), dtype=torch.uint8)
        q4_s13 = torch.empty((len(selected), W13_OUT, HIDDEN // 128), dtype=torch.float16)
        q4_s2 = torch.empty((len(selected), HIDDEN, INTER // 128), dtype=torch.float16)
        metrics = []

        for expert in range(NUM_EXPERTS):
            q2_errors, q4_errors = [], []
            for projection in ("gate_proj", "up_proj", "down_proj"):
                is_down = projection == "down_proj"
                importance = (calibration["w2_moment"][expert] if is_down
                              else calibration["w13_moment"][expert]).to("cuda")
                weight = load_matrix(args.snapshot, source, layer, expert, projection)
                q2 = quantize_adaptive_q2(weight, importance, alphas)
                if is_down:
                    w2p[expert].copy_(q2.packed.cpu())
                    s2[expert].copy_(q2.scales.cpu())
                    a2[expert].copy_(q2.alpha_index.cpu())
                else:
                    sl = slice(0, INTER) if projection == "gate_proj" else slice(INTER, W13_OUT)
                    w13p[expert, sl].copy_(q2.packed.cpu())
                    s13[expert, sl].copy_(q2.scales.cpu())
                    a13[expert, sl].copy_(q2.alpha_index.cpu())
                q2_errors.append(q2.weighted_rel_mse)
                if expert in selected_position:
                    q4p, q4s, q4rel, _, _ = quantize_symmetric_q4(fwht128(weight), importance)
                    pos = selected_position[expert]
                    if is_down:
                        q4_w2[pos].copy_(q4p.cpu()); q4_s2[pos].copy_(q4s.cpu())
                    else:
                        sl = slice(0, INTER) if projection == "gate_proj" else slice(INTER, W13_OUT)
                        q4_w13[pos, sl].copy_(q4p.cpu()); q4_s13[pos, sl].copy_(q4s.cpu())
                    q4_errors.append(q4rel)
                    del q4p, q4s
                del weight, importance, q2
            metrics.append({
                "expert": expert,
                "route_count": int(calibration["count"][expert].item()),
                "q2_weighted_rel_mse_mean": sum(q2_errors) / len(q2_errors),
                "q4_weighted_rel_mse_mean": (sum(q4_errors) / len(q4_errors) if q4_errors else None),
            })
            if expert % 32 == 0:
                print(f"L{layer:02d} E{expert:03d}/{NUM_EXPERTS}", flush=True)
            if expert % 16 == 15:
                gc.collect(); torch.cuda.empty_cache()

        common_md = {
            "format": FORMAT,
            "layer": str(layer),
            "source": "bf16",
            "source_index_sha256": source_sha,
            "source_layout": source["layout"],
            "spec_sha256": spec_sha,
            "calibration_sha256": calibration_sha,
            "precision_map_sha256": precision_sha,
            "group": "128",
            "rotation": "normalized-fwht-h128-input-blocks",
        }
        atomic_save({
            "w13_weight": w13p, "w2_weight": w2p,
            "w13_weight_scale_inv": s13, "w2_weight_scale_inv": s2,
            "w13_alpha_index": a13, "w2_alpha_index": a2,
            "alpha_table": torch.tensor(spec["q2"]["alpha_table"], dtype=torch.float16),
        }, base_path, common_md)
        island_sha = None
        if selected:
            atomic_save({
                "expert_ids": torch.tensor(selected, dtype=torch.int32),
                "w13_weight_q4": q4_w13, "w2_weight_q4": q4_w2,
                "w13_weight_scale_inv_q4": q4_s13, "w2_weight_scale_inv_q4": q4_s2,
            }, island_path, {**common_md, "format": ISLAND_FORMAT,
                             "selected_experts": ",".join(map(str, selected))})
            island_sha = sha256(island_path)
        elif island_path.exists():
            island_path.unlink()
        metrics_path = args.out_dir / f"layer-{layer:02d}.metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
        record = {
            "layer": layer,
            "base": base_path.name,
            "base_sha256": sha256(base_path),
            "base_bytes": base_path.stat().st_size,
            "island": str(island_path.relative_to(args.out_dir)) if selected else None,
            "island_sha256": island_sha,
            "island_bytes": island_path.stat().st_size if selected else 0,
            "selected_experts": selected,
            "elapsed_s": time.time() - started,
        }
        manifest_layers.append(record)
        print(f"L{layer:02d}: complete base={record['base_bytes']/2**20:.1f} MiB "
              f"islands={record['island_bytes']/2**20:.1f} MiB", flush=True)
        del w13p, w2p, s13, s2, a13, a2, q4_w13, q4_w2, q4_s13, q4_s2
        gc.collect(); torch.cuda.empty_cache()

    manifest = {
        "schema_version": 1,
        "format": FORMAT,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "spec_sha256": spec_sha,
        "source_index_sha256": source_sha,
        "precision_map_sha256": precision_sha,
        "layers": manifest_layers,
    }
    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
