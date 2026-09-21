# Quality Phase 0

This track establishes the missing end-to-end quality baseline before any
adaptive codebook, precision-island or pentary experiment. It supports any
OpenAI-compatible reference: BF16, NVFP4, a remote reference deployment, or a
previous frozen build. The implementation identity must describe what was
actually served; it is never inferred from the model name.

## What is included

- `tasks/core-v1.jsonl`: small, redistributable objective sanity suite;
- `lib/run_quality.py`: deterministic runner and local scoring;
- `lib/compare_quality.py`: paired comparison and promotion gate;
- `lib/compare_fidelity.py`: seeded k=0 versus MTP output audit;
- `lib/make_needle_tasks.py`: tokenizer-calibrated retrieval prompts at early,
  middle and late positions;
- `test/run_smoke.sh`: CPU-only end-to-end test.

The bundled ten tasks test the harness, not the model's overall intelligence.
Publishable retention claims require the external benchmark tiers in
`docs/quality-phase0-contract.md`.

## Run the objective core

```bash
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
python3 quality-phase0/lib/run_quality.py \
  --base-url http://127.0.0.1:8012 \
  --model qwen3.8-flash-next-q2 \
  --implementation-id q2-v0114-k0 \
  --tasks quality-phase0/tasks/core-v1.jsonl \
  --out-dir quality-phase0/results/q2-$STAMP \
  --repeats 3 \
  --baseline-sha256 006a19accf3067f51cd2ed8409af64b569494dbf1dd049f4d0f148f3c33eda58 \
  --sidecar-sha256 968c509f6c9333db04228a4138bab83d4aa1f9f950aafcf4713442e5ffc9b432 \
  --runtime-image-id d464f3b466fa \
  --fail-on-task-error
```

Use the same taskset, temperatures, repeats and seeds for the reference. Then:

```bash
python3 quality-phase0/lib/compare_quality.py \
  --baseline quality-phase0/results/reference-$STAMP \
  --candidate quality-phase0/results/q2-$STAMP \
  --out-json quality-phase0/results/compare-$STAMP.json \
  --out-md quality-phase0/results/compare-$STAMP.md
```

## Generate long-context tasks

Start the implementation whose tokenizer/chat template is the comparison
standard, then run:

```bash
python3 quality-phase0/lib/make_needle_tasks.py \
  --base-url http://127.0.0.1:8012 \
  --model qwen3.8-flash-next-q2 \
  --targets 32768,131072,250000 \
  --positions early,middle,late \
  --out-dir quality-phase0/results/needles-v1
```

Run both implementations against the exact generated JSONL file. Its SHA-256
is recorded automatically. Generated prompts can be hundreds of megabytes and
must remain under ignored `results/`.

## MTP comparisons

For quality, run identical seeds once with k=0 and once with k>0. Do not change
temperature, chat template, graph mode, output length or scheduler policy.
Exact answer equality at temperature zero is a useful smoke check; stochastic
distribution equivalence requires repeated samples and is a separate Phase
0 deliverable described in the contract.

```bash
python3 quality-phase0/lib/compare_fidelity.py \
  --target-only quality-phase0/results/k0-$STAMP \
  --mtp quality-phase0/results/k1-$STAMP \
  --out quality-phase0/results/mtp-fidelity-$STAMP.json \
  --require-exact
```

For a diagnostic thinking run, override the deliberately small no-thinking
budgets and identify the profile explicitly:

```bash
python3 quality-phase0/lib/run_quality.py \
  --base-url http://127.0.0.1:8012 \
  --model qwen3.8-flash-next-q2 \
  --implementation-id q2-v0114-k0-thinking-2048 \
  --tasks quality-phase0/tasks/core-v1.jsonl \
  --out-dir quality-phase0/results/q2-k0-thinking-$STAMP \
  --repeats 3 --no-thinking 0 --force-max-tokens 2048
```

Rows ending with `finish_reason=length` are retained as evidence but marked
`valid_for_quality=false` and excluded from headline scores. The manifest
records the forced token budget and complete `extra_body` object.
