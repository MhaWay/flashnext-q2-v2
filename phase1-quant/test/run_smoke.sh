#!/usr/bin/env bash
set -Eeuo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

python3 -m unittest discover -s "$HERE" -p 'test_*.py' -v

mkdir -p "$TMP/analysis"
python3 - "$TMP/analysis" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for layer in range(2):
    path = root / f"layer-{layer:02d}.jsonl"
    with path.open("w") as out:
        for expert in range(10):
            recovery = float((layer + 1) * (expert + 1))
            out.write(json.dumps({
                "layer": layer, "expert": expert, "route_count": 100,
                "q2_weighted_error": 100.0, "q4_weighted_error": 100.0 - recovery,
                "weighted_error_recovery": recovery, "q4_added_bytes": 1000,
                "recovery_per_added_byte": recovery / 1000,
            }) + "\n")
PY
python3 "$ROOT/select_islands.py" --analysis-dir "$TMP/analysis" \
  --budget-fraction 0.10 --out "$TMP/precision-map.json" >/dev/null
python3 - "$TMP/precision-map.json" <<'PY'
import json, sys
result = json.load(open(sys.argv[1]))
assert result["selected_experts"] == 2
assert result["experts_by_layer"] == {"1": [8, 9]}
assert result["used_bytes"] == 2000
PY
echo "PHASE1 QUANT SMOKE OK"
