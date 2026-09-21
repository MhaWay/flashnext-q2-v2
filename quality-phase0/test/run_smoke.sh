#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
TMP="$(mktemp -d)"
SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" ]]; then kill "$SERVER_PID" 2>/dev/null || true; fi
  rm -rf "$TMP"
}
trap cleanup EXIT

python3 "$HERE/mock_server.py" --port 18124 &
SERVER_PID=$!
for _ in $(seq 1 50); do
  if curl -fsS http://127.0.0.1:18124/v1/models >/dev/null 2>&1; then break; fi
  sleep 0.1
done

python3 "$ROOT/quality-phase0/lib/run_quality.py" \
  --base-url http://127.0.0.1:18124 --model mock-quality \
  --implementation-id mock-baseline --tasks "$ROOT/quality-phase0/tasks/core-v1.jsonl" \
  --out-dir "$TMP/baseline" --fail-on-task-error
python3 "$ROOT/quality-phase0/lib/run_quality.py" \
  --base-url http://127.0.0.1:18124 --model mock-quality \
  --implementation-id mock-candidate --tasks "$ROOT/quality-phase0/tasks/core-v1.jsonl" \
  --out-dir "$TMP/candidate" --force-max-tokens 128 --fail-on-task-error
python3 "$ROOT/quality-phase0/lib/run_quality.py" \
  --base-url http://127.0.0.1:18124 --model mock-quality \
  --implementation-id mock-truncated --tasks "$ROOT/quality-phase0/tasks/core-v1.jsonl" \
  --out-dir "$TMP/truncated" --force-max-tokens 7 --fail-on-task-error
python3 "$ROOT/quality-phase0/lib/compare_quality.py" \
  --baseline "$TMP/baseline" --candidate "$TMP/candidate" \
  --out-json "$TMP/comparison.json" --out-md "$TMP/comparison.md"
python3 "$ROOT/quality-phase0/lib/compare_fidelity.py" \
  --target-only "$TMP/baseline" --mtp "$TMP/candidate" \
  --out "$TMP/fidelity.json" --require-exact

python3 "$ROOT/quality-phase0/lib/make_needle_tasks.py" \
  --base-url http://127.0.0.1:18124 --model mock-quality \
  --targets 512 --positions middle --out-dir "$TMP/long"
python3 "$ROOT/quality-phase0/lib/run_quality.py" \
  --base-url http://127.0.0.1:18124 --model mock-quality \
  --implementation-id mock-long --tasks "$TMP/long/long-context-v1.jsonl" \
  --out-dir "$TMP/long-run" --fail-on-task-error

python3 - "$TMP" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
summary = json.loads((root / "baseline/summary.json").read_text())
comparison = json.loads((root / "comparison.json").read_text())
fidelity = json.loads((root / "fidelity.json").read_text())
long_summary = json.loads((root / "long-run/summary.json").read_text())
truncated_summary = json.loads((root / "truncated/summary.json").read_text())
assert summary["task_count"] == 10
assert summary["pass_rate"] == 1.0
assert summary["valid_row_count"] == 10
assert not summary["critical_failures"]
assert comparison["promotion_gate_passed"]
assert fidelity["exact_match_rate"] == 1.0
assert long_summary["pass_rate"] == 1.0
assert truncated_summary["valid_row_count"] == 0
assert truncated_summary["truncated_row_count"] == 10
assert truncated_summary["pass_rate"] is None
candidate_manifest = json.loads((root / "candidate/manifest.json").read_text())
assert candidate_manifest["force_max_tokens"] == 128
PY
echo "QUALITY SMOKE OK"
