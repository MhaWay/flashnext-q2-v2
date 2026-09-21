#!/usr/bin/env python3
"""Small OpenAI-compatible endpoint for quality-phase0 CI."""

import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def answer(prompt: str) -> str:
    pairs = [
        ("37 × 29", "947"),
        ("Who arrived second", "Ada"),
        ("esattamente tre parole", "blu caldo otto"),
        ("Q2-CHECK-71B", "Q2-CHECK-71B"),
        ("weather be tomorrow", '{"tool":"weather","location":"Rome, Italy"}'),
        ("name=Ada", '{"name":"Ada","scores":[3,5,8],"active":true}'),
        ("single Python expression", "[x for x in xs if x % 2 == 0]"),
        ("Project Beta", "9443"),
        ("Object X", "no"),
        ("The service is ready", "Il servizio è pronto (ZX-418)"),
    ]
    for needle, response in pairs:
        if needle in prompt:
            return response
    marker = re.search(r"IMPORTANT RECORD:\s*(FNQ2-[A-Z0-9-]+)", prompt)
    return marker.group(1) if marker else "unknown"


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            self.reply({"data": [{"id": "mock-quality"}]})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        messages = request.get("messages") or []
        prompt = "\n".join(str(message.get("content") or "") for message in messages)
        if self.path == "/tokenize":
            self.reply({"count": max(1, len(prompt.split()) + 12), "tokens": []})
            return
        if self.path == "/v1/chat/completions":
            response = answer(prompt)
            self.reply({
                "id": "mock-completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": response},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": len(response.split())},
            })
            return
        self.send_response(404)
        self.end_headers()

    def reply(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18124)
    args = parser.parse_args()
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
