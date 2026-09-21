#!/usr/bin/env python3
"""Compare two quality-phase0 result directories and enforce promotion gates."""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict


def load_rows(run_dir: pathlib.Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    rows = [json.loads(line) for line in (run_dir / "results.jsonl").read_text().splitlines() if line.strip()]
    return manifest, rows


def aggregate(rows: list[dict]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if not row.get("valid_for_quality", True):
            continue
        grouped[row["task_id"]].append(row)
    return {
        task_id: {
            "score": sum(r["score"] for r in values) / len(values),
            "pass_rate": sum(bool(r["passed"]) for r in values) / len(values),
            "critical": any(bool(r.get("critical")) for r in values),
            "category": values[0]["category"],
        }
        for task_id, values in grouped.items()
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=pathlib.Path)
    ap.add_argument("--candidate", required=True, type=pathlib.Path)
    ap.add_argument("--out-json", required=True, type=pathlib.Path)
    ap.add_argument("--out-md", required=True, type=pathlib.Path)
    ap.add_argument("--max-mean-regression", type=float, default=0.01)
    ap.add_argument("--allow-taskset-mismatch", action="store_true")
    args = ap.parse_args()

    base_manifest, base_rows = load_rows(args.baseline)
    cand_manifest, cand_rows = load_rows(args.candidate)
    same_taskset = base_manifest["taskset_sha256"] == cand_manifest["taskset_sha256"]
    if not same_taskset and not args.allow_taskset_mismatch:
        raise SystemExit("taskset SHA mismatch; comparison refused")
    base = aggregate(base_rows)
    cand = aggregate(cand_rows)
    task_ids = sorted(set(base) & set(cand))
    if not task_ids:
        raise SystemExit("no common task ids")

    tasks = []
    for task_id in task_ids:
        b, c = base[task_id], cand[task_id]
        tasks.append({
            "task_id": task_id,
            "category": b["category"],
            "critical": b["critical"],
            "baseline_score": b["score"],
            "candidate_score": c["score"],
            "delta": c["score"] - b["score"],
            "baseline_pass_rate": b["pass_rate"],
            "candidate_pass_rate": c["pass_rate"],
        })
    base_mean = sum(t["baseline_score"] for t in tasks) / len(tasks)
    cand_mean = sum(t["candidate_score"] for t in tasks) / len(tasks)
    critical_regressions = [
        t["task_id"] for t in tasks
        if t["critical"] and t["candidate_pass_rate"] < t["baseline_pass_rate"]
    ]
    passed = not critical_regressions and cand_mean >= base_mean - args.max_mean_regression
    report = {
        "schema_version": 1,
        "baseline": base_manifest["implementation_id"],
        "candidate": cand_manifest["implementation_id"],
        "same_taskset": same_taskset,
        "common_tasks": len(tasks),
        "baseline_mean_score": base_mean,
        "candidate_mean_score": cand_mean,
        "mean_delta": cand_mean - base_mean,
        "max_mean_regression": args.max_mean_regression,
        "critical_regressions": critical_regressions,
        "promotion_gate_passed": passed,
        "tasks": tasks,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n")
    lines = [
        "# Quality comparison",
        "",
        f"- Baseline: `{report['baseline']}`",
        f"- Candidate: `{report['candidate']}`",
        f"- Mean score: {base_mean:.4f} -> {cand_mean:.4f} ({cand_mean - base_mean:+.4f})",
        f"- Promotion gate: **{'PASS' if passed else 'FAIL'}**",
        "",
        "| Task | Category | Baseline | Candidate | Delta | Critical |",
        "|---|---|---:|---:|---:|:---:|",
    ]
    for task in tasks:
        lines.append(
            f"| {task['task_id']} | {task['category']} | {task['baseline_score']:.3f} | "
            f"{task['candidate_score']:.3f} | {task['delta']:+.3f} | "
            f"{'yes' if task['critical'] else 'no'} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if passed else 3)


if __name__ == "__main__":
    main()
