#!/usr/bin/env python3
"""Select Q4 precision islands from Phase 1 sensitivity records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--budget-fraction", type=float, default=0.05)
    group.add_argument("--budget-gib", type=float)
    ap.add_argument("--minimum-routes", type=int, default=32)
    args = ap.parse_args()
    records, sources = [], []
    for path in sorted(args.analysis_dir.glob("layer-*.jsonl")):
        sources.append({"path": path.name, "sha256": file_sha(path)})
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if not records:
        raise SystemExit("no sensitivity records found")
    eligible = [record for record in records if record["route_count"] >= args.minimum_routes]
    eligible.sort(key=lambda record: record["recovery_per_added_byte"], reverse=True)
    if args.budget_gib is not None:
        budget_bytes = int(args.budget_gib * 2**30)
    else:
        if not 0 <= args.budget_fraction <= 1:
            raise SystemExit("--budget-fraction must be in [0,1]")
        count = round(len(records) * args.budget_fraction)
        budget_bytes = sum(record["q4_added_bytes"] for record in eligible[:count])
    chosen, used = [], 0
    for record in eligible:
        cost = int(record["q4_added_bytes"])
        if used + cost > budget_bytes:
            continue
        chosen.append(record)
        used += cost
    by_layer = {}
    for record in chosen:
        by_layer.setdefault(str(record["layer"]), []).append(int(record["expert"]))
    total_q2_error = sum(record["q2_weighted_error"] for record in records)
    recovered = sum(record["weighted_error_recovery"] for record in chosen)
    result = {
        "schema_version": 1,
        "format": "flashnext-q2-precision-map-v1",
        "selection_unit": "whole-expert-gate-up-down",
        "selection_metric": "weighted-error-recovery-per-added-byte",
        "minimum_routes": args.minimum_routes,
        "analysis_sources": sources,
        "total_experts": len(records),
        "eligible_experts": len(eligible),
        "selected_experts": len(chosen),
        "budget_bytes": budget_bytes,
        "used_bytes": used,
        "used_gib": used / 2**30,
        "estimated_q2_weighted_error": total_q2_error,
        "estimated_error_recovered": recovered,
        "estimated_recovery_fraction": recovered / max(total_q2_error, 1e-30),
        "experts_by_layer": {key: sorted(value) for key, value in sorted(by_layer.items(), key=lambda x: int(x[0]))},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
