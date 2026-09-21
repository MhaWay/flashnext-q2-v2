#!/usr/bin/env python3
"""Audit seeded output fidelity between target-only and MTP quality runs."""

from __future__ import annotations

import argparse
import json
import pathlib
import re


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip()).casefold()


def load(run_dir: pathlib.Path) -> tuple[dict, dict[tuple[str, int, int], dict]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = {}
    for line in (run_dir / "results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["task_id"], int(row["repeat"]), int(row["seed"]))
        rows[key] = row
    return manifest, rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-only", required=True, type=pathlib.Path)
    ap.add_argument("--mtp", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--require-exact", action="store_true",
                    help="fail unless every matched response is byte-identical")
    args = ap.parse_args()
    target_manifest, target = load(args.target_only)
    mtp_manifest, mtp = load(args.mtp)
    if target_manifest["taskset_sha256"] != mtp_manifest["taskset_sha256"]:
        raise SystemExit("taskset SHA mismatch; fidelity audit refused")
    common = sorted(set(target) & set(mtp))
    if not common:
        raise SystemExit("no rows matched by task/repeat/seed")
    missing_target = sorted(set(mtp) - set(target))
    missing_mtp = sorted(set(target) - set(mtp))
    rows = []
    for key in common:
        left, right = target[key], mtp[key]
        exact = left.get("response", "") == right.get("response", "")
        normalized = normalize(left.get("response", "")) == normalize(right.get("response", ""))
        rows.append({
            "task_id": key[0],
            "repeat": key[1],
            "seed": key[2],
            "exact_match": exact,
            "normalized_match": normalized,
            "target_score": left.get("score"),
            "mtp_score": right.get("score"),
            "score_delta": float(right.get("score", 0)) - float(left.get("score", 0)),
            "target_response_sha256": left.get("response_sha256"),
            "mtp_response_sha256": right.get("response_sha256"),
        })
    report = {
        "schema_version": 1,
        "target_only": target_manifest["implementation_id"],
        "mtp": mtp_manifest["implementation_id"],
        "taskset_sha256": target_manifest["taskset_sha256"],
        "matched_rows": len(rows),
        "missing_target_rows": len(missing_target),
        "missing_mtp_rows": len(missing_mtp),
        "exact_match_rate": sum(row["exact_match"] for row in rows) / len(rows),
        "normalized_match_rate": sum(row["normalized_match"] for row in rows) / len(rows),
        "mean_score_delta": sum(row["score_delta"] for row in rows) / len(rows),
        "note": "Seeded equality is a smoke test, not proof of stochastic distribution equivalence.",
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.require_exact and (report["exact_match_rate"] != 1.0 or missing_target or missing_mtp):
        raise SystemExit(4)


if __name__ == "__main__":
    main()
