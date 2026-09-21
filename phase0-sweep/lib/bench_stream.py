#!/usr/bin/env python3
"""Concurrent streaming benchmark against an OpenAI-compatible vLLM endpoint.

Fire --streams simultaneous streaming chat/completions requests, measure per
stream TTFT / elapsed / completion tokens, and emit an aggregate JSON:
  wall_time_s, output_tokens, aggregate_tps, per_stream_tps_mean,
  ttft_mean, ttft_p95, prompt_tokens, ok_streams, err_streams, errors[].
Exit code 0 only if every stream completed without error.
"""
import argparse
import json
import threading
import time
import urllib.request


def stream_one(base, model, prompt, args, slot):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if str(args.no_thinking) == "1":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    usage = None
    chunk_count = 0
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta") or {}
                    piece = delta.get("content") or delta.get("reasoning_content") or ""
                    if piece and ttft is None:
                        ttft = time.time() - t0
                if chunk.get("usage"):
                    usage = chunk["usage"]
                chunk_count += 1
        t_end = time.time()
        slot.update(
            ok=True,
            ttft=ttft,
            elapsed=t_end - t0,
            end=t_end,
            completion_tokens=(usage or {}).get("completion_tokens") or chunk_count,
            prompt_tokens=(usage or {}).get("prompt_tokens"),
        )
    except Exception as exc:  # noqa: BLE001 - any stream failure is reportable
        slot.update(ok=False, error=str(exc), elapsed=time.time() - t0, end=time.time())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--streams", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=float, default=5400)
    ap.add_argument("--no-thinking", default="1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompt = open(args.prompt_file, encoding="utf-8").read()
    slots = [{} for _ in range(args.streams)]
    threads = []
    t_start = time.time()
    for i in range(args.streams):
        th = threading.Thread(target=stream_one, args=(args.base_url, args.model, prompt, args, slots[i]))
        th.start()
        threads.append(th)
    for th in threads:
        th.join()

    ok = [s for s in slots if s.get("ok")]
    err = [s for s in slots if not s.get("ok")]
    wall = (max((s.get("end", t_start) for s in slots), default=t_start)) - t_start
    total_tokens = sum(int(s.get("completion_tokens") or 0) for s in ok)
    ttf_ts = sorted(s["ttft"] for s in ok if s.get("ttft") is not None)
    per_stream = [s["completion_tokens"] / s["elapsed"] for s in ok if s.get("elapsed", 0) > 0 and s.get("completion_tokens")]

    result = {
        "wall_time_s": round(wall, 3),
        "output_tokens": total_tokens,
        "aggregate_tps": round(total_tokens / wall, 3) if wall > 0 and total_tokens else 0.0,
        "per_stream_tps_mean": round(sum(per_stream) / len(per_stream), 3) if per_stream else 0.0,
        "ttft_mean": round(sum(ttf_ts) / len(ttf_ts), 3) if ttf_ts else None,
        "ttft_p95": round(ttf_ts[min(int(len(ttf_ts) * 0.95), len(ttf_ts) - 1)], 3) if ttf_ts else None,
        "prompt_tokens": next((s["prompt_tokens"] for s in ok if s.get("prompt_tokens")), None),
        "ok_streams": len(ok),
        "err_streams": len(err),
        "errors": [s.get("error") for s in err][:5],
        "per_stream_detail": [
            {"ttft": s.get("ttft"), "elapsed": round(s.get("elapsed", 0), 3),
             "completion_tokens": s.get("completion_tokens"), "prompt_tokens": s.get("prompt_tokens")}
            for s in ok
        ],
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not err else 2)


if __name__ == "__main__":
    main()
