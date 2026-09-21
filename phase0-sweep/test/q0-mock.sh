#!/usr/bin/env bash
# Mock of the frozen flashnext-quat.sh for phase0-sweep smoke tests.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${MOCK_PORT:-18123}"
k=0
case "${1:-}" in
  serve)
    if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
      k=$(printf '%s' "$EXTRA_VLLM_ARGS" | grep -o '"num_speculative_tokens":[0-9]*' | grep -o '[0-9]*$' || true)
      k=${k:-0}
    fi
    exec python3 "$HERE/mock_server.py" --port "$PORT" --k "$k"
    ;;
  serve-decode-mtp3)
    exec python3 "$HERE/mock_server.py" --port "$PORT" --k 3
    ;;
  stop)
    pkill -f "mock_server.py --port $PORT" || true
    ;;
  *)
    echo "mock baseline: unsupported action ${1:-}" >&2
    exit 1
    ;;
esac
