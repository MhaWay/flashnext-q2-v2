# Phase 1 — adaptive quaternary weight generation

Phase 1 creates a **new, separate** weight set. It does not replace the
v0.11.4 sidecars and it does not claim runtime compatibility yet.

The format keeps the proven H128/group-128/2-bit base, but chooses `alpha`
per weight group from a small table using expert-conditioned activation
second moments. A global byte budget then promotes the experts with the best
measured error recovery to whole-expert Q4 companions.

## Contract

- Evaluation mode remains `no_thinking=true`.
- Frozen control SHA-256:
  `006a19accf3067f51cd2ed8409af64b569494dbf1dd049f4d0f148f3c33eda58`.
- Existing Q2 sidecars are read-only and remain the rollback set.
- Calibration, source index, spec and precision map hashes are embedded in
  every generated layer. A mixed set fails verification.
- `flashnext-q2-adaptive-v1` requires the Phase 2 runtime. Do **not** point
  v0.11.4 at the generated directory.

## Spark workflow

Set paths once:

```bash
cd ~/flashnext-q2-v2
BASELINE=/home/mhaway/flashnext-quat.sh
SOURCE=$(sudo cat /media/mhaway/XTREME1/flashnext-q2/state/source_snapshot.txt)
WORK=/media/mhaway/XTREME1/flashnext-q2/phase1
IMAGE=vllm/vllm-openai:qwen38-flash-next
mkdir -p "$WORK"
```

`SOURCE` must resolve to the directory containing
`model.safetensors.index.json`.

### 1. Build the isolated calibration launcher

```bash
python3 phase1-quant/build_calibration_launcher.py \
  --baseline "$BASELINE" \
  --out "$WORK/flashnext-phase1-calibrate.sh"
```

The command refuses every baseline except the frozen SHA above. Start it with
the expert-major prefill path disabled so the deliberately narrow route-direct
instrumentation is guaranteed to run:

```bash
sudo "$BASELINE" stop
sudo env \
  Q2_PREFILL_V8_ENABLE=0 \
  FLASHNEXT_PHASE1_CALIBRATION_FLUSH_EVERY=1 \
  "$WORK/flashnext-phase1-calibrate.sh" serve
```

The default flush of one is intentionally slow but crash-safe. It overwrites
the same cumulative 48 files after each scheduler chunk; increase it only
when the request count is controlled.

### 2. Capture a deterministic no-thinking corpus

In a second terminal:

```bash
python3 phase1-quant/collect_calibration.py \
  --base-url http://127.0.0.1:8012 \
  --model qwen3.8-flash-next-q2 \
  --target-words 30000 \
  | tee "$WORK/calibration-request.json"
```

The server prints its profile directory at boot. The 48 calibration files are
under `<profile-dir>/phase1-calibration`. Preserve that directory as
`$WORK/calibration` and stop the calibration server:

```bash
sudo cp -a /media/mhaway/XTREME1/flashnext-q2/profiles/serve-YYYYMMDD-HHMMSS/phase1-calibration \
  "$WORK/calibration"
sudo "$WORK/flashnext-phase1-calibrate.sh" stop
```

Do not combine files from separate runs. A future corpus expansion must be a
new named calibration set.

Validate all 48 files before doing any weight work:

```bash
sudo docker run --rm \
  -v "$PWD:/work:ro" -v "$WORK:/phase1:rw" \
  --entrypoint python3 "$IMAGE" \
  /work/phase1-quant/verify_calibration.py \
    --calibration-dir /phase1/calibration \
    --out /phase1/calibration/verification.json
```

### 3. Measure every expert

This is the long GPU pass: 48 layers × 512 experts × 3 projections, Q2 and Q4.
It is resumable per completed layer.

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v "$PWD:/work:ro" \
  -v "$SOURCE:/source:ro" \
  -v "$WORK:/phase1:rw" \
  --entrypoint python3 \
  "$IMAGE" \
  /work/phase1-quant/analyze_sensitivity.py \
    --snapshot /source \
    --calibration-dir /phase1/calibration \
    --out-dir /phase1/sensitivity
```

Use the exact runtime image/digest already frozen by Phase 0 if the local tag
differs. Run a layer smoke first with `--end-layer 1`.

### 4. Allocate the precision-island budget

```bash
python3 phase1-quant/select_islands.py \
  --analysis-dir "$WORK/sensitivity" \
  --budget-fraction 0.05 \
  --minimum-routes 32 \
  --out "$WORK/precision-map.json" \
  | tee "$WORK/precision-map-summary.log"
```

Selection is global, not “5% per layer”: experts compete on expected weighted
error recovered per extra byte. An explicit `--budget-gib` can replace the
fraction.

### 5. Generate the new weights

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v "$PWD:/work:ro" \
  -v "$SOURCE:/source:ro" \
  -v "$WORK:/phase1:rw" \
  --entrypoint python3 \
  "$IMAGE" \
  /work/phase1-quant/convert_adaptive.py \
    --snapshot /source \
    --calibration-dir /phase1/calibration \
    --precision-map /phase1/precision-map.json \
    --out-dir /phase1/sidecars-adaptive-v1
```

The converter saves each layer atomically and resumes only when both the base
and selected Q4 companion match all hashes and shapes.

### 6. Fail-closed verification

```bash
sudo docker run --rm --gpus all --ipc=host \
  -v "$PWD:/work:ro" \
  -v "$WORK:/phase1:rw" \
  --entrypoint python3 \
  "$IMAGE" \
  /work/phase1-quant/verify_sidecars.py \
    --sidecar-dir /phase1/sidecars-adaptive-v1 \
    --calibration-dir /phase1/calibration \
    --precision-map /phase1/precision-map.json \
    --out /phase1/sidecars-adaptive-v1/verification.json
```

Promotion to Phase 2 requires 48/48 verified layers, retained raw sensitivity
rows and an explicit quality A/B against both the current Q2 and a BF16 or
identified higher-precision reference.

## Storage model

- Old fixed-alpha Q2: `0.265625 B/weight`.
- Adaptive Q2: `0.2734375 B/weight`; the uint8 alpha index adds one byte per
  128 weights (+2.94%, roughly +0.88 GiB over the 29.883 GiB routed set).
- One whole-expert Q4 companion: 2,534,400 bytes (about 2.417 MiB).
- A 5% global island budget is about 1,229 experts and 2.90 GiB, so the routed
  set is expected around 33.7 GiB before safetensors metadata.

These are storage figures, not runtime bandwidth claims. Phase 2 must measure
the cost of alpha lookup and mixed Q2/Q4 dispatch.
