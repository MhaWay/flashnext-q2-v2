# Multi-agent handoff

## Goal

Deliver a v2 runtime that improves single-stream decode and TTFT while
preserving useful model quality and the memory advantage of the Q2 routed
experts. Phase 0 stays frozen; Phase 1 now generates an isolated adaptive
candidate and must not change serving defaults.

## Evidence packages

- Performance contract: `docs/phase0-contract.md`
- First hardware review: `docs/results/phase0-gb10-2026-09-21.md`
- Quality contract: `docs/quality-phase0-contract.md`
- Machine-readable state: `docs/data/project-state-v1.json`
- Phase 1 design: `docs/phase1-architecture.md`
- Phase 1 Spark workflow: `phase1-quant/README.md`
- Frozen baseline entry point: external `flashnext-quat.sh` v0.11.4, hash in
  the state file

## Implemented Phase 1 path

- Hash-locked derivation of a calibration-only v0.11.4 launcher.
- Expert-conditioned W13/W2 diagonal activation moments.
- Adaptive per-group alpha from a six-value shared table.
- Full Q2-versus-Q4 sensitivity scan and global byte-budget selector.
- Atomic/resumable sidecar writer and fail-closed verifier.

The runtime does not yet load `flashnext-q2-adaptive-v1`; that is the Phase 2
boundary, not unfinished behavior hidden behind the current defaults.

## Ready tasks

### A. Reference-quality runner

Connect `quality-phase0/lib/run_quality.py` to the best available reference,
execute Q0/Q1, and publish a compact manifest plus summary. Do not call an
NVFP4 proxy “BF16”.

### B. Teacher-forced adapter

Add an adapter that emits per-token NLL, logit KL and top-k agreement for a
fixed calibration corpus. It must support sampling layers/components for
ablation without storing full-vocabulary logits for every token.

### C. Execute sensitivity capture

Run and review the implemented W13/W2 capture on GB10. Extend router, shared
expert, attention and lm-head boundaries as separate candidates; do not mix
their statistics into the routed sidecar manifest.

### D. MTP fidelity and cost

Run k=0/1/2/3 with matched seeds and configs. Record answer equality,
acceptance by position, draft cycles, total throughput and the inferred
incremental draft cost. Add a repeated-sampling distribution test before
claiming the block/probabilistic verifier is exact.

### E. Adaptive codebook hardware run

Execute the committed per-group shared-table candidate. Add global/per-layer/
per-expert ablations only if they reuse the same frozen calibration and report
metadata bytes plus anticipated kernel branches.

### F. Precision-island review

Review the committed Q4 selector at 0/2/5/10% budgets. Later extend it to
BF16/FP8/NVFP4/Q3 only with measured runtime dispatch costs; hit rate alone is
not an objective.

## Definition of done for each task

1. Reproduction command and environment.
2. Input hashes and model/runtime identity.
3. Raw JSONL artifact and compact JSON summary.
4. Test or mock fixture where applicable.
5. Interpretation separating measurement from inference.
6. Explicit next decision or a documented negative result.
