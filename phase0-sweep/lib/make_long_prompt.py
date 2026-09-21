#!/usr/bin/env python3
"""Build and verify a deterministic long-context prompt with vLLM /tokenize."""

import argparse
import hashlib
import json
import pathlib
import urllib.request


GENERATOR_VERSION = 2
NONCE = "0123456789abcdef0123456789abcdef"
WORDS = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega"
).split()
INTRO = (
    "The following is a long synthetic reference document assembled for "
    "context-retrieval benchmarking. Reference document begins: "
)
OUTRO = (
    " Reference document ends. Question: According to the reference document "
    "above, what is the final marker word? Answer concisely. "
    "Marker word: ZQX-7714-DELTA."
)


def make_prompt(word_count: int) -> str:
    body = " ".join(WORDS[i % len(WORDS)] for i in range(word_count))
    return INTRO + body + OUTRO


def token_count(base_url: str, model: str, prompt: str, no_thinking: bool) -> int:
    content = f"Unique benchmark nonce: {NONCE}\n{prompt}"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "chat_template_kwargs": {"enable_thinking": not no_thinking},
    }
    req = urllib.request.Request(
        f"{base_url}/tokenize",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        result = json.load(resp)
    return int(result["count"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--target-tokens", required=True, type=int)
    ap.add_argument("--no-thinking", default="1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta", required=True)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    meta = pathlib.Path(args.meta)
    goal = max(args.target_tokens - 32, 1)  # nonce/tokenizer variation reserve
    no_thinking = str(args.no_thinking) == "1"

    if out.is_file() and meta.is_file():
        old = json.loads(meta.read_text())
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        if (
            old.get("generator_version") == GENERATOR_VERSION
            and old.get("target_tokens") == args.target_tokens
            and old.get("model") == args.model
            and old.get("no_thinking") == no_thinking
            and old.get("prompt_sha256") == digest
        ):
            print(json.dumps(old))
            return

    base_prompt = make_prompt(0)
    base_count = token_count(args.base_url, args.model, base_prompt, no_thinking)
    candidates = [(base_count, 0, base_prompt)]
    words = max(goal - base_count, 1)

    # Token count is almost linear in word count.  Proportional correction
    # reaches <0.1% error in a few calls without downloading dozens of large
    # token-id arrays as a full binary search would.
    for _ in range(6):
        prompt = make_prompt(words)
        count = token_count(args.base_url, args.model, prompt, no_thinking)
        candidates.append((count, words, prompt))
        if abs(count - goal) <= max(8, goal // 1000):
            break
        variable = max(count - base_count, 1)
        words = max(0, round(words * max(goal - base_count, 0) / variable))

    chosen = min(candidates, key=lambda x: abs(x[0] - goal))
    count, words, prompt = chosen
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(prompt)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    record = {
        "generator_version": GENERATOR_VERSION,
        "model": args.model,
        "no_thinking": no_thinking,
        "target_tokens": args.target_tokens,
        "calibrated_tokens": count,
        "word_count": words,
        "prompt_bytes": out.stat().st_size,
        "prompt_sha256": digest,
    }
    meta.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
