# Multi-agent handoff

## Goal

Deliver a v2 runtime that improves single-stream decode and TTFT while
preserving useful model quality and the memory advantage of the Q2 routed
experts. The immediate work is measurement, not a format rewrite.

## Evidence packages

- Performance contract: `docs/phase0-contract.md`
- First hardware review: `docs/results/phase0-gb10-2026-09-21.md`
- Quality contract: `docs/quality-phase0-contract.md`
- Machine-readable state: `docs/data/project-state-v1.json`
- Frozen baseline entry point: external `flashnext-quat.sh` v0.11.4, hash in
  the state file

## Ready parallel tasks

### A. Reference-quality runner

Connect `quality-phase0/lib/run_quality.py` to the best available reference,
execute Q0/Q1, and publish a compact manifest plus summary. Do not call an
NVFP4 proxy “BF16”.

### B. Teacher-forced adapter

Add an adapter that emits per-token NLL, logit KL and top-k agreement for a
fixed calibration corpus. It must support sampling layers/components for
ablation without storing full-vocabulary logits for every token.

### C. Sensitivity capture

Capture activation statistics at routed W13/W2, router, shared expert,
attention and lm-head boundaries. Output a `[layer, expert, component]` table
with activation-weighted reconstruction error and recovery from selective
higher precision.

### D. MTP fidelity and cost

Run k=0/1/2/3 with matched seeds and configs. Record answer equality,
acceptance by position, draft cycles, total throughput and the inferred
incremental draft cost. Add a repeated-sampling distribution test before
claiming the block/probabilistic verifier is exact.

### E. Adaptive codebook simulator

Offline only: sweep alpha per layer/expert/group for
`{-1,-alpha,+alpha,+1}` using activation-weighted error. Compare one global
alpha, per-layer alpha, per-expert alpha and a small shared codebook set. Report
quality proxy, metadata bytes and anticipated kernel branches.

### F. Precision-island optimizer

After B/C: solve a byte-budgeted selection problem over BF16/FP8/NVFP4/Q4/Q3/Q2
components. The objective is measured quality recovery, not hit rate. Include
the runtime cost of routing between kernels.

## Definition of done for each task

1. Reproduction command and environment.
2. Input hashes and model/runtime identity.
3. Raw JSONL artifact and compact JSON summary.
4. Test or mock fixture where applicable.
5. Interpretation separating measurement from inference.
6. Explicit next decision or a documented negative result.

