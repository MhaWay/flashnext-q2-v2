#!/usr/bin/env bash
# Reproducible smoke test of the phase0 harness against a mock endpoint.
# Verifies: server lifecycle for all three command paths (k=0, k=1/2, k=3),
# metrics-delta parsing, run.json schema keys, summary generation.
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="$(dirname "$HERE")"
export Q0_SCRIPT="$HERE/q0-mock.sh"
export BASE_URL="http://127.0.0.1:18123"
export RESULTS_DIR="$HERE/results"
export MAX_TOKENS=32 REQ_TIMEOUT=30 READY_TIMEOUT=60 INTER_DELAY=0 NCU_TEST=0
export WARMUP_REPEATS=1 MEASURED_REPEATS=1 LONG_REPEATS=1
export SHORT_STREAMS_LIST="1" SHORT_WORKLOADS_LIST="prose"
export SHORT_KS_LIST="${SHORT_KS_LIST:-0 1 2 3}"
rm -rf "$RESULTS_DIR"
bash "$BASE/phase0-sweep.sh" sweep-short
export SHORT_KS_LIST=""  # no-op matrix for report-only pass below
bash "$BASE/phase0-sweep.sh" report
echo "SMOKE OK: $(wc -l < "$RESULTS_DIR/results.jsonl") rows"
