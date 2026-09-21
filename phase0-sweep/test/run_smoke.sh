#!/usr/bin/env bash
# Reproducible smoke test of the phase0 harness against a mock endpoint.
# Verifies: parity-controlled server lifecycle for k=0/1/2/3, labeled
# metrics-delta parsing, short/long phases, run schema and summary generation.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(dirname "$HERE")"
export PATH="$HERE/fake-bin:$PATH"
export Q0_SCRIPT="$HERE/q0-mock.sh"
export BASE_URL="http://127.0.0.1:18123"
export RESULTS_DIR="$HERE/results"
export MAX_TOKENS=32 REQ_TIMEOUT=30 READY_TIMEOUT=60 INTER_DELAY=0 NCU_TEST=0
export WARMUP_REPEATS=1 MEASURED_REPEATS=1 LONG_REPEATS=1
export SHORT_STREAMS_LIST="1 4" SHORT_WORKLOADS_LIST="prose"
export SHORT_KS_LIST="${SHORT_KS_LIST:-0 1 2 3}"
export LONG_CONTEXTS_LIST="64"
rm -rf "$RESULTS_DIR"
bash "$BASE/phase0-sweep.sh" full
python3 - "$RESULTS_DIR/results.jsonl" <<'PY'
import json, sys
rows = [json.loads(line) for line in open(sys.argv[1])]
measured = [r for r in rows if r["phase"] == "short" and not r["warmup"]]
assert {r["k"] for r in measured} == {0, 1, 2, 3}
assert {r["streams"] for r in measured} == {1, 4}
assert all(r["repeat"] == 1 for r in measured)
assert all(r["decode_tokens"] == 7 * r["streams"] for r in measured)
assert all(r["draft_cycles"] > 0 for r in measured if r["k"] > 0)
assert all(r["accepted_by_position"] for r in measured if r["k"] > 0)
assert all(r["memory_peak_gib"] is None for r in rows)
assert any(r["phase"] == "long" and not r["warmup"] for r in rows)
PY
echo "SMOKE OK: $(wc -l < "$RESULTS_DIR/results.jsonl") rows"
