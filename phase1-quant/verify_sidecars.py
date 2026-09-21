#!/usr/bin/env python3
"""Fail-closed validation for Phase 1 adaptive-Q2/Q4 sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


NUM_EXPERTS = 512
HIDDEN = 2560
INTER = 640
W13_OUT = 1280
BASE_FORMAT = "flashnext-q2-adaptive-v1"
ISLAND_FORMAT = "flashnext-q4-islands-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_tensor(handle, key: str, shape: tuple[int, ...], dtype: str) -> None:
    if key not in handle.keys():
        raise ValueError(f"missing tensor {key}")
    tensor = handle.get_slice(key)
    if tuple(tensor.get_shape()) != shape:
        raise ValueError(f"{key}: shape {tuple(tensor.get_shape())} != {shape}")
    if tensor.get_dtype() != dtype:
        raise ValueError(f"{key}: dtype {tensor.get_dtype()} != {dtype}")


def verify_base(path: Path, layer: int, spec_sha: str, calibration_sha: str,
                precision_sha: str, alpha_count: int) -> dict:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        md = handle.metadata() or {}
        required = {
            "format": BASE_FORMAT, "layer": str(layer), "spec_sha256": spec_sha,
            "calibration_sha256": calibration_sha,
            "precision_map_sha256": precision_sha,
        }
        if any(md.get(key) != value for key, value in required.items()):
            raise ValueError(f"{path}: metadata mismatch")
        check_tensor(handle, "w13_weight", (NUM_EXPERTS, W13_OUT, HIDDEN // 4), "U8")
        check_tensor(handle, "w2_weight", (NUM_EXPERTS, HIDDEN, INTER // 4), "U8")
        check_tensor(handle, "w13_weight_scale_inv", (NUM_EXPERTS, W13_OUT, HIDDEN // 128), "F16")
        check_tensor(handle, "w2_weight_scale_inv", (NUM_EXPERTS, HIDDEN, INTER // 128), "F16")
        check_tensor(handle, "w13_alpha_index", (NUM_EXPERTS, W13_OUT, HIDDEN // 128), "U8")
        check_tensor(handle, "w2_alpha_index", (NUM_EXPERTS, HIDDEN, INTER // 128), "U8")
        check_tensor(handle, "alpha_table", (alpha_count,), "F16")
        for key in ("w13_alpha_index", "w2_alpha_index"):
            maximum = int(handle.get_tensor(key).max().item())
            if maximum >= alpha_count:
                raise ValueError(f"{path}: {key} contains alpha index {maximum}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_island(path: Path, layer: int, selected: list[int], spec_sha: str,
                  calibration_sha: str, precision_sha: str) -> dict | None:
    if not selected:
        if path.exists():
            raise ValueError(f"{path}: stale island file exists for empty selection")
        return None
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        md = handle.metadata() or {}
        required = {
            "format": ISLAND_FORMAT, "layer": str(layer), "spec_sha256": spec_sha,
            "calibration_sha256": calibration_sha,
            "precision_map_sha256": precision_sha,
            "selected_experts": ",".join(map(str, selected)),
        }
        if any(md.get(key) != value for key, value in required.items()):
            raise ValueError(f"{path}: metadata mismatch")
        n = len(selected)
        check_tensor(handle, "expert_ids", (n,), "I32")
        check_tensor(handle, "w13_weight_q4", (n, W13_OUT, HIDDEN // 2), "U8")
        check_tensor(handle, "w2_weight_q4", (n, HIDDEN, INTER // 2), "U8")
        check_tensor(handle, "w13_weight_scale_inv_q4", (n, W13_OUT, HIDDEN // 128), "F16")
        check_tensor(handle, "w2_weight_scale_inv_q4", (n, HIDDEN, INTER // 128), "F16")
        if handle.get_tensor("expert_ids").tolist() != selected:
            raise ValueError(f"{path}: expert_ids do not match precision map")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-dir", required=True, type=Path)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--precision-map", required=True, type=Path)
    parser.add_argument("--spec", type=Path,
                        default=Path(__file__).resolve().parent / "specs/adaptive-q2-v1.json")
    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=48)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    precision = json.loads(args.precision_map.read_text())
    if precision.get("format") != "flashnext-q2-precision-map-v1":
        raise SystemExit("unsupported precision map")
    spec_sha, precision_sha = sha256(args.spec), sha256(args.precision_map)
    alpha_count = len(spec["q2"]["alpha_table"])
    rows, total_bytes = [], 0
    for layer in range(max(0, args.start_layer), min(48, args.end_layer)):
        calibration = args.calibration_dir / f"layer-{layer:02d}.safetensors"
        selected = sorted(int(x) for x in precision.get("experts_by_layer", {}).get(str(layer), []))
        if len(selected) != len(set(selected)) or any(x < 0 or x >= NUM_EXPERTS for x in selected):
            raise ValueError(f"layer {layer}: invalid selected expert list")
        base = verify_base(args.sidecar_dir / f"layer-{layer:02d}.safetensors", layer,
                           spec_sha, sha256(calibration), precision_sha, alpha_count)
        island = verify_island(args.sidecar_dir / "islands" / f"layer-{layer:02d}.safetensors",
                               layer, selected, spec_sha, sha256(calibration), precision_sha)
        row = {"layer": layer, "base": base, "island": island, "selected_experts": selected}
        rows.append(row)
        total_bytes += base["bytes"] + (island["bytes"] if island else 0)
        print(f"L{layer:02d}: OK base={base['bytes']/2**20:.1f} MiB "
              f"islands={(island['bytes']/2**20 if island else 0):.1f} MiB")
    result = {
        "schema_version": 1, "status": "ok", "format": BASE_FORMAT,
        "spec_sha256": spec_sha, "precision_map_sha256": precision_sha,
        "layers_verified": len(rows), "total_bytes": total_bytes,
        "total_gib": total_bytes / 2**30, "layers": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(args.out)
    print(json.dumps({key: value for key, value in result.items() if key != "layers"}, indent=2))


if __name__ == "__main__":
    main()
