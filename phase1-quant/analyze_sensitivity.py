#!/usr/bin/env python3
"""Measure adaptive-Q2 versus Q4 error for every routed expert."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))

import torch  # noqa: E402

from calibration import load_layer_calibration  # noqa: E402
from source_reader import (HIDDEN, INTER, NUM_EXPERTS, build_source_map, load_matrix,
                           load_weight_map)  # noqa: E402
from torch_codec import fwht128, quantize_adaptive_q2, quantize_symmetric_q4  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True, type=Path)
    ap.add_argument("--calibration-dir", required=True, type=Path)
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
    spec = json.loads(args.spec.read_text())
    alphas = torch.tensor(spec["q2"]["alpha_table"], dtype=torch.float32, device="cuda")
    weight_map = load_weight_map(args.snapshot)
    source = build_source_map(weight_map)
    index_path = args.snapshot / "model.safetensors.index.json"
    common = {
        "format": "flashnext-q2-sensitivity-v1",
        "spec_sha256": sha256(args.spec),
        "source_index_sha256": sha256(index_path),
        "source_layout": source["layout"],
        "minimum_routes": args.minimum_routes,
    }

    for layer in range(max(0, args.start_layer), min(48, args.end_layer)):
        output = args.out_dir / f"layer-{layer:02d}.jsonl"
        meta_path = args.out_dir / f"layer-{layer:02d}.meta.json"
        calibration_path = args.calibration_dir / f"layer-{layer:02d}.safetensors"
        expected_meta = {**common, "layer": layer, "calibration_sha256": sha256(calibration_path)}
        if output.is_file() and meta_path.is_file() and not args.force:
            if json.loads(meta_path.read_text()) == expected_meta:
                print(f"L{layer:02d}: resume valid analysis", flush=True)
                continue
        calibration = load_layer_calibration(args.calibration_dir, layer, args.minimum_routes)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        with temporary.open("w") as sink:
            for expert in range(NUM_EXPERTS):
                count = int(calibration["count"][expert].item())
                q2_error = q4_error = weighted_ref = 0.0
                detail = {}
                for projection in ("gate_proj", "up_proj", "down_proj"):
                    importance_cpu = (calibration["w13_moment"][expert]
                                      if projection != "down_proj"
                                      else calibration["w2_moment"][expert])
                    importance = importance_cpu.to("cuda")
                    weight = load_matrix(args.snapshot, source, layer, expert, projection)
                    q2 = quantize_adaptive_q2(weight, importance, alphas)
                    _, _, q4_rel, q4_abs, q4_ref = quantize_symmetric_q4(
                        fwht128(weight), importance
                    )
                    # Conditional second moments become expected layer cost
                    # only after weighting by how often the expert was routed.
                    route_weight = max(count, 0)
                    q2_error += q2.weighted_error * route_weight
                    q4_error += q4_abs * route_weight
                    weighted_ref += q4_ref * route_weight
                    detail[projection] = {
                        "q2_weighted_rel_mse": q2.weighted_rel_mse,
                        "q2_rel_mse": q2.rel_mse,
                        "q4_weighted_rel_mse": q4_rel,
                    }
                    del weight, importance, q2
                recovery = max(q2_error - q4_error, 0.0)
                # Q4 is a companion copy: 0.5 B/weight + FP16/128.  All three
                # matrices contain the same number of weights for this model.
                weights_per_matrix = INTER * HIDDEN
                q4_added_bytes = int(3 * weights_per_matrix * (0.5 + 2 / 128))
                record = {
                    "layer": layer,
                    "expert": expert,
                    "route_count": count,
                    "q2_weighted_error": q2_error,
                    "q4_weighted_error": q4_error,
                    "weighted_reference": weighted_ref,
                    "weighted_error_recovery": recovery,
                    "q4_added_bytes": q4_added_bytes,
                    "recovery_per_added_byte": recovery / q4_added_bytes,
                    "detail": detail,
                }
                sink.write(json.dumps(record) + "\n")
                sink.flush()
                if expert % 32 == 0:
                    print(f"L{layer:02d} E{expert:03d}: count={count} recovery={recovery:.6e}", flush=True)
                if expert % 16 == 15:
                    gc.collect()
                    torch.cuda.empty_cache()
        os.replace(temporary, output)
        atomic_json(meta_path, expected_meta)
        print(f"L{layer:02d}: sensitivity complete", flush=True)


if __name__ == "__main__":
    main()
