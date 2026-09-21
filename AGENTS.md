# Agent contract

This repository is the shared, evidence-first workspace for FlashNext-Q2 v2.
Read this file, `docs/phase0-contract.md`, and
`docs/quality-phase0-contract.md` before changing runtime or quantization code.

## Non-negotiable boundaries

- `flashnext-quat.sh` v0.11.4 is the frozen control. It is external to this
  repository and must never be patched by Phase 0 tooling.
- A performance comparison is valid only when baseline script, harness,
  sidecar, prompt/taskset and runtime-image fingerprints are recorded.
- Weight relMSE is not model quality. Do not state a retention percentage
  without an end-to-end task suite and an identified reference model.
- MTP acceptance is not throughput. Select MTP depth/precision using emitted
  tokens per total target+draft+runtime cost.
- Exact speculative verification should preserve the target distribution;
  the current block/probabilistic implementation has not yet been proven
  exact and requires a k=0 versus k>0 seeded output audit.
- Never commit API keys, Hugging Face tokens, generated 250K prompts, raw model
  weights or large result bundles.

## Current verified baseline

- Target: Qwen3.8-Flash-Next MoE on one DGX Spark / GB10, 128 GB unified RAM.
- Routed format: H128 rotation and four-state codebook
  `{-1, -1/3, +1/3, +1}`, group size 128, FP16 scale.
- Storage: `0.25 + 2/128 = 0.265625` bytes/weight; 48 routed layers occupy
  29.883 GiB in the sidecar used for Phase 0.
- The current GB10 measurements and their provenance live in
  `docs/data/project-state-v1.json`; do not copy numbers from chat logs.
- End-to-end BF16-versus-Q2 quality is **unknown**. Establishing it is the
  first quality gate.

## Work tracks

Agents may work independently on these tracks if their outputs respect the
contracts above:

1. `perf/`: kernel/runtime timing ledger and profiler fallback.
2. `quality/`: objective BF16/reference versus Q2 evaluation.
3. `calibration/`: activation capture, sensitivity maps and adaptive alpha.
4. `mtp/`: draft precision sweep and exactness/acceptance/cost audit.
5. `mixed-precision/`: precision-island simulation after sensitivity data.

Each contribution must state: hypothesis, controlled variables, input hashes,
commands, raw artifact location, result, and decision. Negative results are
first-class results.

## Development and review

- Keep runtime experiments behind new commands or scripts; do not silently
  change defaults.
- Prefer Python modules plus tests over Python embedded in shell heredocs.
- Use JSONL for raw rows, JSON for manifests/summaries and Markdown only for
  human-readable interpretation.
- Tests must run without a GPU using the mock endpoints.
- Run `phase0-sweep/test/run_smoke.sh` and
  `quality-phase0/test/run_smoke.sh` before requesting review.
- Put large artifacts in a release, external bundle or ignored `results/`
  directory; commit only compact summaries with hashes and provenance.

