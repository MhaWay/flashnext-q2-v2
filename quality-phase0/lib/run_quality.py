#!/usr/bin/env python3
"""Run deterministic, locally scored quality tasks against an OpenAI API.

The runner intentionally uses only the Python standard library.  A run emits
one JSONL row per task/repeat plus a summary and an immutable manifest.  API
keys are read from an environment variable and are never written to disk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import subprocess
import time
import urllib.request
from collections import defaultdict
from typing import Any


SCHEMA_VERSION = 1


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_text(value: str, *, case_sensitive: bool = False) -> str:
    value = re.sub(r"\s+", " ", value.strip())
    return value if case_sensitive else value.casefold()


def extract_json(text: str) -> Any:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_candidates = [p for p in (text.find("{"), text.find("[")) if p >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        for end in range(len(text), start, -1):
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                continue
        raise


def score_response(response: str, spec: dict[str, Any]) -> tuple[float, str]:
    kind = spec["type"]
    if kind == "exact":
        got = canonical_text(response, case_sensitive=spec.get("case_sensitive", False))
        expected = canonical_text(str(spec["value"]), case_sensitive=spec.get("case_sensitive", False))
        return (1.0 if got == expected else 0.0, f"exact expected={expected!r}")
    if kind == "numeric":
        matches = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", response.replace(",", "."))
        if not matches:
            return 0.0, "no numeric value found"
        got = float(matches[0])
        expected = float(spec["value"])
        tolerance = float(spec.get("tolerance", 0.0))
        return (1.0 if abs(got - expected) <= tolerance else 0.0,
                f"numeric got={got} expected={expected} tolerance={tolerance}")
    if kind == "contains_all":
        haystack = canonical_text(response)
        missing = [str(v) for v in spec["values"] if canonical_text(str(v)) not in haystack]
        return (1.0 if not missing else 0.0, f"missing={missing}")
    if kind == "regex":
        matched = re.search(spec["pattern"], response, flags=re.MULTILINE | re.DOTALL) is not None
        return (1.0 if matched else 0.0, f"pattern={spec['pattern']!r}")
    if kind == "json_equal":
        try:
            got = extract_json(response)
        except (json.JSONDecodeError, ValueError) as exc:
            return 0.0, f"invalid JSON: {exc}"
        expected = spec["value"]
        return (1.0 if got == expected else 0.0,
                f"json got={got!r} expected={expected!r}")
    raise ValueError(f"unknown scorer type: {kind}")


def load_tasks(path: pathlib.Path) -> list[dict[str, Any]]:
    tasks = []
    seen = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        task = json.loads(line)
        missing = {"id", "category", "prompt", "scorer"} - task.keys()
        if missing:
            raise ValueError(f"{path}:{lineno}: missing fields {sorted(missing)}")
        if task["id"] in seen:
            raise ValueError(f"{path}:{lineno}: duplicate task id {task['id']!r}")
        score_response(str(task["scorer"].get("value", "")), task["scorer"])
        seen.add(task["id"])
        tasks.append(task)
    if not tasks:
        raise ValueError(f"no tasks in {path}")
    return tasks


def git_revision(repo: pathlib.Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def request_completion(args: argparse.Namespace, task: dict[str, Any], seed: int) -> tuple[dict, float]:
    max_tokens = (args.force_max_tokens if args.force_max_tokens is not None
                  else int(task.get("max_tokens", args.max_tokens)))
    body: dict[str, Any] = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": task.get("system", "Follow the user instruction exactly.")},
            {"role": "user", "content": task["prompt"]},
        ],
        "max_tokens": max_tokens,
        "temperature": float(task.get("temperature", args.temperature)),
        "seed": seed,
        "stream": False,
    }
    if args.no_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    if args.extra_body:
        body.update(args.extra_body)
    headers = {"Content-Type": "application/json"}
    token = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers=headers,
    )
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=args.timeout) as response:
        result = json.load(response)
    return result, time.perf_counter() - started


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--implementation-id", required=True,
                        help="stable label, e.g. q2-v0114-k0 or bf16-reference")
    parser.add_argument("--tasks", required=True, type=pathlib.Path)
    parser.add_argument("--out-dir", required=True, type=pathlib.Path)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--force-max-tokens", type=int,
                        help="override every per-task max_tokens value (useful for thinking runs)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--no-thinking", type=int, choices=(0, 1), default=1)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--extra-body-json", default="{}")
    parser.add_argument("--baseline-sha256", default="")
    parser.add_argument("--sidecar-sha256", default="")
    parser.add_argument("--runtime-image-id", default="")
    parser.add_argument("--fail-on-task-error", action="store_true")
    args = parser.parse_args()
    args.no_thinking = bool(args.no_thinking)
    args.extra_body = json.loads(args.extra_body_json)
    if not isinstance(args.extra_body, dict):
        raise SystemExit("--extra-body-json must decode to an object")
    if args.repeats < 1:
        raise SystemExit("--repeats must be >= 1")
    if args.force_max_tokens is not None and args.force_max_tokens < 1:
        raise SystemExit("--force-max-tokens must be >= 1")

    tasks_raw = args.tasks.read_bytes()
    tasks = load_tasks(args.tasks)
    args.out_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "implementation_id": args.implementation_id,
        "base_url": args.base_url,
        "model": args.model,
        "taskset": str(args.tasks),
        "taskset_sha256": sha256_bytes(tasks_raw),
        "task_count": len(tasks),
        "repeats": args.repeats,
        "seed_base": args.seed,
        "temperature_default": args.temperature,
        "max_tokens_default": args.max_tokens,
        "force_max_tokens": args.force_max_tokens,
        "no_thinking": args.no_thinking,
        "extra_body": args.extra_body,
        "baseline_sha256": args.baseline_sha256 or None,
        "sidecar_sha256": args.sidecar_sha256 or None,
        "runtime_image_id": args.runtime_image_id or None,
        "repo_revision": git_revision(pathlib.Path(__file__).resolve().parents[2]),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    rows = []
    failures = 0
    result_path = args.out_dir / "results.jsonl"
    with result_path.open("w", encoding="utf-8") as sink:
        for repeat in range(args.repeats):
            for index, task in enumerate(tasks):
                seed = args.seed + repeat * len(tasks) + index
                row = {
                    "schema_version": SCHEMA_VERSION,
                    "implementation_id": args.implementation_id,
                    "taskset_sha256": manifest["taskset_sha256"],
                    "task_id": task["id"],
                    "category": task["category"],
                    "critical": bool(task.get("critical", False)),
                    "repeat": repeat + 1,
                    "seed": seed,
                }
                try:
                    response, latency = request_completion(args, task, seed)
                    choice = response["choices"][0]
                    message = choice.get("message") or {}
                    text = message.get("content") or ""
                    reasoning_text = message.get("reasoning_content") or ""
                    score, detail = score_response(text, task["scorer"])
                    finish_reason = choice.get("finish_reason")
                    truncated = finish_reason == "length"
                    row.update({
                        "ok": True,
                        "score": score,
                        "passed": score >= float(task.get("pass_score", 1.0)),
                        "valid_for_quality": not truncated,
                        "truncated": truncated,
                        "score_detail": detail,
                        "latency_s": round(latency, 6),
                        "finish_reason": finish_reason,
                        "response": text,
                        "response_sha256": sha256_bytes(text.encode("utf-8")),
                        "reasoning_content": reasoning_text,
                        "reasoning_sha256": (sha256_bytes(reasoning_text.encode("utf-8"))
                                             if reasoning_text else None),
                        "usage": response.get("usage") or {},
                    })
                except Exception as exc:  # all request failures belong in the artifact
                    failures += 1
                    row.update({"ok": False, "score": 0.0, "passed": False,
                                "valid_for_quality": False, "truncated": False,
                                "error": f"{type(exc).__name__}: {exc}"})
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                rows.append(row)
                print(f"{task['id']} repeat={repeat + 1} score={row['score']:.3f} ok={row['ok']}")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    categories = {}
    for category, values in sorted(grouped.items()):
        valid = [value for value in values if value.get("valid_for_quality", True)]
        categories[category] = {
            "rows": len(values),
            "valid_rows": len(valid),
            "truncated_rows": sum(bool(value.get("truncated")) for value in values),
            "mean_score": (sum(value["score"] for value in valid) / len(valid) if valid else None),
            "pass_rate": (sum(bool(value["passed"]) for value in valid) / len(valid) if valid else None),
        }
    valid_rows = [row for row in rows if row.get("valid_for_quality", True)]
    summary = {
        **manifest,
        "row_count": len(rows),
        "valid_row_count": len(valid_rows),
        "invalid_row_count": len(rows) - len(valid_rows),
        "truncated_row_count": sum(bool(row.get("truncated")) for row in rows),
        "request_failures": failures,
        "mean_score": (sum(row["score"] for row in valid_rows) / len(valid_rows)
                       if valid_rows else None),
        "pass_rate": (sum(bool(row["passed"]) for row in valid_rows) / len(valid_rows)
                      if valid_rows else None),
        "critical_failures": sorted({
            row["task_id"] for row in valid_rows if row["critical"] and not row["passed"]
        }),
        "critical_invalid": sorted({
            row["task_id"] for row in rows
            if row["critical"] and not row.get("valid_for_quality", True)
        }),
        "categories": categories,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if args.fail_on_task_error and failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
