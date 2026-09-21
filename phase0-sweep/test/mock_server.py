#!/usr/bin/env python3
"""Mock vLLM-compatible endpoint for phase0-sweep smoke tests.

Implements: GET /v1/models, GET /metrics (cumulative counters), POST /tokenize,
and POST /v1/chat/completions (SSE, fixed 8 accounted tokens, four visible
chunks). Advances speculative-decode counters on every completion when --k>0.
"""
import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

LOCK = threading.Lock()
C = {"requests": 0, "prompt_tok": 0, "gen_tok": 0, "drafts": 0, "draft_tok": 0, "accepted": 0}
POS = [0, 0, 0, 0]  # accepted token count at draft position, matching vLLM
PATTERN = [2, 1, 3, 0, 2, 2, 1, 3]
TURN = [0]
ARGS = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            body = json.dumps({"data": [{"id": ARGS.model}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/metrics"):
            lines = [
                "vllm:num_requests_total %d" % C["requests"],
                "vllm:prompt_tokens_total %d" % C["prompt_tok"],
                "vllm:generation_tokens_total %d" % C["gen_tok"],
            ]
            if ARGS.k > 0:
                lines += [
                    'vllm:spec_decode_num_drafts_total{engine="0",model_name="mock"} %d' % C["drafts"],
                    'vllm:spec_decode_num_draft_tokens_total{engine="0",model_name="mock"} %d' % C["draft_tok"],
                    'vllm:spec_decode_num_accepted_tokens_total{engine="0",model_name="mock"} %d' % C["accepted"],
                ]
                for i in range(ARGS.k):
                    lines.append(
                        'vllm:spec_decode_num_accepted_tokens_per_pos_total'
                        '{engine="0",model_name="mock",position="%d"} %d' % (i, POS[i])
                    )
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            body = ("\n".join(lines) + "\n").encode()
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        prompt = json.dumps(req.get("messages", ""))
        prompt_tokens = max(1, len(prompt) // 4)
        if self.path.startswith("/tokenize"):
            body = json.dumps({
                "count": prompt_tokens,
                "max_model_len": 262144,
                "tokens": [1] * prompt_tokens,
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        with LOCK:
            C["requests"] += 1
            C["prompt_tok"] += prompt_tokens
            C["gen_tok"] += 8
            if ARGS.k > 0:
                acc = min(PATTERN[TURN[0] % len(PATTERN)], ARGS.k)
                TURN[0] += 1
                C["drafts"] += 1
                C["draft_tok"] += ARGS.k
                C["accepted"] += acc
                for pos in range(acc):
                    POS[pos] += 1

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        for _ in range(4):
            time.sleep(0.002)
            chunk = {"choices": [{"index": 0, "delta": {"content": " tok"}}]}
            self.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode())
            self.wfile.flush()
        # Account four additional invisible tokens before DONE.  This catches
        # clients that incorrectly end the decode window at last visible text.
        time.sleep(0.008)
        usage = {"choices": [], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 8}}
        self.wfile.write(("data: " + json.dumps(usage) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18123)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--model", default="mock/qwen3.8-flash-next")
    ARGS = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
