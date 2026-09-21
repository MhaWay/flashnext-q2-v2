#!/usr/bin/env python3
"""Send a deterministic, no-thinking prefill corpus to the calibration server."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request


SEEDS = (
    "Explain the invariant and edge cases of this Python function: def merge(xs, ys): return sorted(xs + ys).",
    "Risolvi con passaggi espliciti un problema di percentuali, poi verifica il risultato con un metodo diverso.",
    "Design a fault-tolerant queue consumer with idempotency, retries, observability, and bounded memory.",
    "Confronta memoria virtuale, cache, DMA e coerenza in una CPU moderna con esempi concreti.",
    "Given four people and ordering constraints, derive the unique schedule and state every implication.",
    "Translate this technical incident report between English and Italian while preserving identifiers.",
    "Write JSON schemas for a tool call, validate malformed cases, and explain security boundaries.",
    "Analyze a CUDA kernel for coalescing, occupancy, divergence, launch overhead, and numerical error.",
)


def build_prompt(words: int) -> str:
    blocks, current = [], 0
    index = 0
    while current < words:
        seed = SEEDS[index % len(SEEDS)]
        block = f"Document {index:05d}. {seed} Nonce {index * 104729 % 1000003}."
        blocks.append(block)
        current += len(block.split())
        index += 1
    return "\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8012")
    parser.add_argument("--model", default="qwen3.8-flash-next-q2")
    parser.add_argument("--target-words", type=int, default=30000,
                        help="Approximate corpus size; 30k words is normally >32k tokens")
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    prompt = build_prompt(args.target_words)
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 1,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    raw = json.dumps(body).encode()
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/v1/chat/completions", data=raw,
        headers={"Content-Type": "application/json"}, method="POST")
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        result = json.loads(response.read())
    usage = result.get("usage", {})
    record = {
        "schema_version": 1, "no_thinking": True,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "target_words": args.target_words, "prompt_bytes": len(prompt.encode()),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "elapsed_s": time.monotonic() - started,
        "finish_reason": result.get("choices", [{}])[0].get("finish_reason"),
    }
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
