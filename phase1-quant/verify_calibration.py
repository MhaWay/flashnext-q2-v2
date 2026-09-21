#!/usr/bin/env python3
"""Validate one complete Phase 1 activation-calibration set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from safetensors import safe_open


BASELINE_SHA256 = "006a19accf3067f51cd2ed8409af64b569494dbf1dd049f4d0f148f3c33eda58"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--minimum-routes", type=int, default=32)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = []
    for layer in range(48):
        path = args.calibration_dir / f"layer-{layer:02d}.safetensors"
        if not path.is_file():
            raise FileNotFoundError(path)
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            md = handle.metadata() or {}
            expected = {
                "format": "flashnext-q2-activation-calibration-v1",
                "layer": str(layer), "num_experts": "512", "hidden": "2560",
                "intermediate": "640", "baseline_sha256": BASELINE_SHA256,
                "capture_path": "route-direct-prefill-v0.7",
            }
            if any(md.get(key) != value for key, value in expected.items()):
                raise ValueError(f"{path}: metadata mismatch")
            expected_tensors = {
                "w13_sq_sum": ((512, 2560), "F32"),
                "w2_sq_sum": ((512, 640), "F32"),
                "route_count": ((512,), "I64"),
            }
            for key, (shape, dtype) in expected_tensors.items():
                if key not in handle.keys():
                    raise ValueError(f"{path}: missing {key}")
                tensor = handle.get_slice(key)
                if tuple(tensor.get_shape()) != shape or tensor.get_dtype() != dtype:
                    raise ValueError(f"{path}: invalid {key}")
            counts = handle.get_tensor("route_count")
            if bool((counts < 0).any()):
                raise ValueError(f"{path}: negative route count")
            total = int(counts.sum().item())
            if total <= 0:
                raise ValueError(f"{path}: empty calibration")
            observed = int((counts >= args.minimum_routes).sum().item())
        rows.append({
            "layer": layer, "sha256": sha256(path), "bytes": path.stat().st_size,
            "total_routes": total, "experts_at_minimum": observed,
            "coverage": observed / 512,
        })
        print(f"L{layer:02d}: routes={total} experts>={args.minimum_routes}: {observed}/512")
    result = {
        "schema_version": 1, "status": "ok",
        "format": "flashnext-q2-activation-calibration-v1",
        "baseline_sha256": BASELINE_SHA256, "minimum_routes": args.minimum_routes,
        "layers_verified": len(rows),
        "minimum_expert_coverage": min(row["coverage"] for row in rows),
        "layers": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "layers"}, indent=2))


if __name__ == "__main__":
    main()
