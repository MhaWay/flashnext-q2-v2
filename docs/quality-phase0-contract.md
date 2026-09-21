# Quality Phase 0 contract

## Decision being made

Quality Phase 0 decides whether the current Q2 quaternary target is an
acceptable base and which components need more precision. It does not attempt
to prove that a new kernel is faster.

The control should be the highest-quality practical instance of the same model
and chat template. Preference order:

1. BF16 target model;
2. an official high-quality FP8/NVFP4 target with demonstrated parity;
3. a frozen external endpoint with fully recorded identity;
4. the best prior local runtime, explicitly labelled as a proxy rather than
   BF16 ground truth.

Comparisons between different model weights or chat templates are diagnostic,
not quantization-retention measurements.

## Required tiers

| Tier | Purpose | Minimum evidence |
|---|---|---|
| Q0 | Harness sanity | bundled core-v1, three repeats |
| Q1 | Capability | coding, math/reasoning, instruction following, Italian and English |
| Q2 | Model likelihood | perplexity or teacher-forced logit/KL data on a fixed calibration corpus |
| Q3 | Long context | needle/retrieval at 32K, 128K and 250K, early/middle/late |
| Q4 | Agentic behavior | structured tool selection, argument validity and multi-step completion |
| Q5 | MTP fidelity | k=0 versus k>0 seeded equality plus sampled distribution audit |

The repository intentionally does not vendor third-party benchmark datasets.
Adapters must record dataset name, revision, split, license, sample IDs and
SHA-256. Never record only an aggregate score.

## Controlled variables

Candidate and reference must use the same:

- base model revision and tokenizer;
- system/user prompts and chat template;
- `enable_thinking` value and reasoning effort;
- generation parameters, seeds and output limits;
- task selection and scoring revision.

Record separately: routed format, dense/shared precision, router precision,
embedding/lm-head precision, KV format, MTP depth and MTP precision.

## Metrics

- Objective tasks: per-task score, pass rate and critical failures.
- Paired generation: exact match, normalized match and task-specific score.
- Teacher-forced evaluation: NLL/perplexity, logit KL, top-1/top-5 agreement.
- Long context: retrieval accuracy by context length and needle position.
- MTP: accepted tokens per draft cycle, draft cost, target-output fidelity and
  end-to-end tok/s.

Rows truncated at the generation limit are invalid for quality scoring. They
must remain in the raw artifact with `finish_reason`, usage and partial output,
but are excluded from aggregate pass rates. Thinking and no-thinking profiles
require separate, explicitly recorded output budgets.

Never collapse all domains into one unexplained “retention” percentage. If a
summary is necessary, publish the per-domain table next to it.

## Promotion gates

The default harness gate allows no new critical-task regression and at most a
1 percentage-point mean-score loss against the selected reference. This is a
development guard, not a final scientific threshold.

A quantization variant can enter runtime optimization only when:

1. no Q0/Q3 critical task regresses;
2. Q1/Q4 results are available by domain;
3. Q2 logit or perplexity data exists for the affected component;
4. all inputs and outputs are fingerprinted;
5. any quality gain is reported with its resident-byte and hot-path cost.

## Experiments enabled after the baseline

Run these in order because each isolates a different source of loss:

1. BF16 versus current full Q2/FP8 stack.
2. Restore only `lm_head` to BF16.
3. Restore only router logits/routing decision to BF16.
4. Sweep MTP Q2/Q3/Q4/FP8 with the target fixed.
5. Capture activation-weighted error per layer/expert and sweep
   `alpha` in `{-1, -alpha, +alpha, +1}`.
6. Promote only the most sensitive layers/experts to Q3/Q4/NVFP4.

Pentary is deferred until these experiments show that four code values, rather
than calibration or a small number of sensitive components, limit quality.

## MTP decision rule

For one draft step, a useful simplified throughput model is

`R1 / R0 = (1 + a) / (1 + d)`,

where `a` is accepted draft tokens per cycle and `d` is draft/runtime cost as a
fraction of one target step. MTP wins only when `a > d`. The corrected
long-context observations estimate `d` near 0.50 for the tested k=1 runtime,
so a practical policy should use hysteresis around that break-even rather than
enable MTP unconditionally. This estimate must be recomputed after changing
draft precision, QSA metadata or graph capture.
