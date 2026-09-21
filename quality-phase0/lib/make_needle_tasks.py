#!/usr/bin/env python3
"""Generate tokenizer-calibrated long-context retrieval tasks.

Large prompts are generated into the chosen output directory and intentionally
remain outside git.  Each prompt carries a deterministic needle at an early,
middle or late position and a JSONL taskset references the prompt content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import urllib.request


GENERATOR_VERSION = 1
FILLER = (
    "cedar orbit marble quiet lantern river copper meadow winter violet "
    "harbor compass silver orchard cloud amber valley quartz"
).split()
POSITIONS = {"early": 0.10, "middle": 0.50, "late": 0.90}


def content_for(words: int, marker: str, position: float) -> str:
    body = [FILLER[i % len(FILLER)] for i in range(words)]
    at = min(len(body), max(0, round(len(body) * position)))
    body[at:at] = ["IMPORTANT", "RECORD:", marker]
    return (
        "Read the synthetic archive below. Ignore repeated filler words. "
        "Remember the exact value following IMPORTANT RECORD.\n\n"
        + " ".join(body)
        + "\n\nQuestion: What exact value followed IMPORTANT RECORD? "
          "Return only that value."
    )


def tokenize(base_url: str, model: str, prompt: str, no_thinking: bool, timeout: float) -> int:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "chat_template_kwargs": {"enable_thinking": not no_thinking},
    }
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/tokenize",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(json.load(response)["count"])


def calibrate(base_url: str, model: str, target: int, marker: str, position: float,
              no_thinking: bool, timeout: float) -> tuple[str, int, int]:
    base = tokenize(base_url, model, content_for(0, marker, position), no_thinking, timeout)
    goal = max(target - 16, base)
    words = max(goal - base, 1)
    candidates = []
    for _ in range(7):
        prompt = content_for(words, marker, position)
        count = tokenize(base_url, model, prompt, no_thinking, timeout)
        candidates.append((abs(count - goal), prompt, count, words))
        if abs(count - goal) <= max(8, goal // 1000):
            break
        variable = max(count - base, 1)
        words = max(0, round(words * max(goal - base, 0) / variable))
    _, prompt, count, words = min(candidates, key=lambda item: item[0])
    return prompt, count, words


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--targets", default="32768,131072,250000")
    ap.add_argument("--positions", default="early,middle,late")
    ap.add_argument("--out-dir", required=True, type=pathlib.Path)
    ap.add_argument("--no-thinking", type=int, choices=(0, 1), default=1)
    ap.add_argument("--timeout", type=float, default=900)
    args = ap.parse_args()
    targets = [int(value) for value in args.targets.split(",") if value.strip()]
    positions = [value.strip() for value in args.positions.split(",") if value.strip()]
    unknown = set(positions) - POSITIONS.keys()
    if unknown:
        raise SystemExit(f"unknown positions: {sorted(unknown)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    taskset = args.out_dir / "long-context-v1.jsonl"
    metadata = []
    with taskset.open("w", encoding="utf-8") as tasks:
        for target in targets:
            for position_name in positions:
                marker = f"FNQ2-{target}-{position_name.upper()}-7C9A"
                prompt, calibrated, words = calibrate(
                    args.base_url, args.model, target, marker, POSITIONS[position_name],
                    bool(args.no_thinking), args.timeout,
                )
                prompt_path = args.out_dir / f"needle-{target}-{position_name}.txt"
                prompt_path.write_text(prompt, encoding="utf-8")
                digest = hashlib.sha256(prompt.encode()).hexdigest()
                task = {
                    "id": f"needle_{target}_{position_name}",
                    "category": "long_context_retrieval",
                    "critical": True,
                    "prompt": prompt,
                    "max_tokens": 32,
                    "temperature": 0.0,
                    "scorer": {"type": "exact", "value": marker, "case_sensitive": True},
                    "metadata": {
                        "generator_version": GENERATOR_VERSION,
                        "target_tokens": target,
                        "calibrated_tokens": calibrated,
                        "needle_position": position_name,
                        "word_count": words,
                        "prompt_sha256": digest,
                    },
                }
                tasks.write(json.dumps(task, ensure_ascii=False) + "\n")
                metadata.append({"id": task["id"], **task["metadata"]})
                print(json.dumps(metadata[-1]))
    (args.out_dir / "long-context-v1.meta.json").write_text(
        json.dumps({
            "generator_version": GENERATOR_VERSION,
            "base_url": args.base_url,
            "model": args.model,
            "no_thinking": bool(args.no_thinking),
            "tasks": metadata,
        }, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
