# FlashNext-Q2 v2 — Phase 0 measurement harness

Public workspace for the runtime **v2 rewrite** of the FlashNext-Q2 pipeline
(offline PTQ to Q2 quaternary format + GPU runtime) targeting **Qwen3.8-Flash-Next
MoE on DGX Spark / GB10**.

**Status: Phase 0 baseline captured; Phase 1 profiling next.** The measurement
sweep separates MoE-kernel cost from MTP cost, QSA-metadata cost and scheduler
cost, so the v2 kernel design starts from numbers instead of assumptions.

## Safety contract

- This harness **never modifies** the frozen `flashnext-quat.sh` v0.11.4
  baseline. It is treated as a black box (`serve` / `stop`) via `Q0_SCRIPT`.
  The baseline itself is **not** part of this repo.
- A lock file prevents two simultaneous runs.
- Every run is recorded: config, raw Prometheus metrics before/after, stream
  detail, run.json, append-only results.jsonl.

## Layout

    flashnext-q2-v2/
    └── phase0-sweep/
        ├── phase0-sweep.sh        # orchestrator: check | sweep-short | sweep-long | full | report
        ├── lib/bench_stream.py    # N-way concurrent SSE streaming bench (TTFT, tok/s)
        ├── lib/metrics_delta.py   # Prometheus before->after delta parser (counters, histograms)
        └── test/                  # mock endpoint + mock baseline + smoke test (no GPU needed)

## Quick start (on the Spark)

    cd phase0-sweep
    Q0_SCRIPT=~/flashnext-quat.sh SIDECAR_DIR=/path/to/sidecar ./phase0-sweep.sh check
    # then, in tmux:
    Q0_SCRIPT=/path/to/flashnext-quat.sh SIDECAR_DIR=/path/to/sidecar ./phase0-sweep.sh full

Matrix (defaults): k=0..3 x streams=1/2/4/8 x {prose, code} x (1 warmup + 3
measured) at 512 forced output tokens; then long contexts
64/4096/32768/131072/250000 on k=0 + best k (+ second if within 3% margin).
Piece overrides: `SHORT_KS_LIST`, `SHORT_STREAMS_LIST`, `SHORT_WORKLOADS_LIST`,
`LONG_CONTEXTS_LIST`, `LONG_KS_LIST`, `MTP_BATCHED_TOKENS`, `MAX_TOKENS`,
`MEASURED_REPEATS`, `INTER_DELAY`, `NCU_TEST`.

The stable GB10 default is a 4096-token scheduler chunk. The 8192-token real
prefill path showed a first-use JIT/Inductor stall and is retained as an
explicit diagnostic override, not as the measurement default.

## Before the first real run

1. The default launcher calls the baseline `serve` action for every k and pins
   the same piecewise-graph, KV, batching, Q2-block and QSA policy. Only
   `MTP_ENABLE` and `MTP_NUM_SPECULATIVE_TOKENS` change. `SERVER_CMD_K0`,
   `SERVER_CMD_KN` (`__K__` placeholder) and `SERVER_CMD_K3` remain escape
   hatches for a baseline with a different interface.
2. `check` writes `results/profile_check.json`: it answers, with a *real* kernel
   counter read (not `which ncu`), whether Nsight Compute works on GB10, or
   which fallback to use (nsys / CUPTI / CUDA events / host timing). It never
   pulls the large runtime image implicitly; use `PROFILE_RECHECK=1` to rerun
   the probe after changing profiler permissions or the image.
3. `counters_k<N>.json` records the exact spec_decode metric names exposed by
   the running server; acceptance columns fall back to zero (with a warning)
   if absent.
4. Long prompts are generated against the live `/tokenize` endpoint and carry
   calibration metadata. Delete neither `ctx*.txt` nor its `.meta.json` alone;
   a missing or mismatched pair is regenerated automatically.

## Smoke test (no GPU, no server)

    cd phase0-sweep && test/run_smoke.sh

Drives all three server-command paths and the long-context phase against a
built-in mock vLLM endpoint.

## Roadmap (after Phase 0 data)

1. NCU profile of the current Q2 kernel at realistic forms M=1/4/8/32.
2. Kernel v2: CUDA GEMV for M=1; GPU planner + grouped GEMM for M=2..32;
   expert-major path for M>=64; graph-safe custom op.
3. Offline dual-layout decision (packed-2 interleaved/swizzled vs signed
   bitplanes) decided by microbenchmark, not by taste.
4. Route simulation with the byte-ledger formula (tempo_route =
   byte_route/banda + dispatch + partition) to evaluate hybrid/adaptive-K and
   the NVFP4 hot-cache layer (break-even: NVFP4 wins only if
   BW_NVFP4 > ~2.12 x BW_Q2).

The exact Phase 0 measurement contract, MTP break-even model and promotion
gates are in [`docs/phase0-contract.md`](docs/phase0-contract.md).

The first GB10 hardware bundle and the measurement corrections it exposed are
reviewed in
[`docs/results/phase0-gb10-2026-09-21.md`](docs/results/phase0-gb10-2026-09-21.md).
