#!/usr/bin/env python3
"""Concurrent streaming benchmark against an OpenAI-compatible vLLM endpoint.

Fire --streams simultaneous streaming chat/completions requests, measure per
stream TTFT / decode window / completion tokens, and emit an aggregate JSON.
Decode rates exclude TTFT and the first emitted token:
  wall_time_s, output_tokens, decode_tokens, aggregate_tps, per_stream_tps_mean,
  ttft_mean, ttft_p95, prompt_tokens, ok_streams, err_streams, errors[].
Exit code 0 only if every stream completed without error.
"""
import argparse
import json
import threading
import time
import urllib.request
import uuid


def stream_one(base, model, prompt, args, slot, start_barrier):
    nonce = uuid.uuid4().hex
    body = {
        "model": model,
        # Prefix caching stops at the first differing token.  A nonce at the
        # end leaves the whole long prompt cacheable, so it must come first.
        "messages": [{"role": "user", "content": f"Unique benchmark nonce: {nonce}\n{prompt}"}],
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
    start_barrier.wait(timeout=args.timeout)
    t0 = time.perf_counter()
    first_visible = None
    last_visible = None
    usage = None
    visible_chunks = 0
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
                    piece = (delta.get("content") or delta.get("reasoning_content")
                             or delta.get("tool_calls") or "")
                    if piece:
                        visible_chunks += 1
                        now = time.perf_counter()
                        if first_visible is None:
                            first_visible = now
                        last_visible = now
                if chunk.get("usage"):
                    usage = chunk["usage"]
        t_end = time.perf_counter()
        first_visible = first_visible or t_end
        last_visible = last_visible or t_end
        completion_tokens = int((usage or {}).get("completion_tokens") or visible_chunks)
        decode_tokens = max(completion_tokens - 1, 0)
        # Use the completed SSE response as the decode boundary.  With
        # ignore_eos=true, vLLM can account generated EOS/control tokens that
        # have no visible delta; last_visible would then overstate tok/s.
        decode_s = max(t_end - first_visible, 0.0)
        slot.update(
            ok=True,
            sent=t0,
            first=first_visible,
            last=last_visible,
            ttft=first_visible - t0,
            elapsed=t_end - t0,
            decode_s=decode_s,
            decode_tokens=decode_tokens,
            decode_tps=(decode_tokens / decode_s if decode_s > 0 else None),
            end=t_end,
            completion_tokens=completion_tokens,
            visible_chunks=visible_chunks,
            prompt_tokens=(usage or {}).get("prompt_tokens"),
        )
    except Exception as exc:  # noqa: BLE001 - any stream failure is reportable
        end = time.perf_counter()
        slot.update(ok=False, error=str(exc), sent=t0, elapsed=end - t0, end=end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--streams", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=5400)
    ap.add_argument("--no-thinking", default="1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prompt = open(args.prompt_file, encoding="utf-8").read()
    slots = [{} for _ in range(args.streams)]
    threads = []
    start_barrier = threading.Barrier(args.streams)
    for i in range(args.streams):
        th = threading.Thread(
            target=stream_one,
            args=(args.base_url, args.model, prompt, args, slots[i], start_barrier),
        )
        th.start()
        threads.append(th)
    for th in threads:
        th.join()

    ok = [s for s in slots if s.get("ok")]
    err = [s for s in slots if not s.get("ok")]
    sent = [s["sent"] for s in ok]
    done = [s["end"] for s in ok]
    wall = max(done) - min(sent) if sent and done else 0.0
    total_tokens = sum(int(s.get("completion_tokens") or 0) for s in ok)
    decode_tokens = sum(int(s.get("decode_tokens") or 0) for s in ok)
    ttf_ts = sorted(s["ttft"] for s in ok if s.get("ttft") is not None)
    per_stream = [s["decode_tps"] for s in ok if s.get("decode_tps") is not None]
    decode_window = 0.0
    if ok:
        decode_window = max(s["end"] for s in ok) - min(s["first"] for s in ok)

    result = {
        "wall_time_s": round(wall, 3),
        "output_tokens": total_tokens,
        "visible_chunks": sum(int(s.get("visible_chunks") or 0) for s in ok),
        "decode_tokens": decode_tokens,
        "decode_window_s": round(decode_window, 6),
        "aggregate_tps": round(decode_tokens / decode_window, 3) if decode_window > 0 and decode_tokens else 0.0,
        "request_total_tps": round(total_tokens / wall, 3) if wall > 0 and total_tokens else 0.0,
        "per_stream_tps_mean": round(sum(per_stream) / len(per_stream), 3) if per_stream else 0.0,
        "ttft_mean": round(sum(ttf_ts) / len(ttf_ts), 3) if ttf_ts else None,
        "ttft_p95": round(ttf_ts[min(int(len(ttf_ts) * 0.95), len(ttf_ts) - 1)], 3) if ttf_ts else None,
        "prompt_tokens": next((s["prompt_tokens"] for s in ok if s.get("prompt_tokens")), None),
        "ok_streams": len(ok),
        "err_streams": len(err),
        "errors": [s.get("error") for s in err][:5],
        "per_stream_detail": [
            {"ttft": s.get("ttft"), "elapsed": round(s.get("elapsed", 0), 3),
             "decode_s": round(s.get("decode_s", 0), 6),
             "decode_tokens": s.get("decode_tokens"), "decode_tps": s.get("decode_tps"),
             "completion_tokens": s.get("completion_tokens"),
             "visible_chunks": s.get("visible_chunks"),
             "prompt_tokens": s.get("prompt_tokens")}
            for s in ok
        ],
    }
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if not err else 2)


if __name__ == "__main__":
    main()
