# Phase 0 measurement contract

Phase 0 answers one question before any v2 kernel rewrite: where does the
decode time go when MTP depth and effective decode batch change? The frozen
`flashnext-quat.sh` v0.11.4 runtime is the control. The harness must not patch
or regenerate it.

## Controlled variables

Every short-matrix cell uses the same runtime policy:

- piecewise CUDA graphs;
- 20 GiB KV budget, 8 sequences and 8192 batched tokens;
- Q2 W13/W2 block N = 16 and expert-major threshold M = 64;
- block rejection, probabilistic draft sampling, deterministic QSA top-k off;
- fixed 512-token output (`ignore_eos=true`), temperature 0.5;
- one unmeasured warmup followed by three measured repeats.

Only MTP depth k and simultaneous stream count change. The nominal decode form
is `M = streams * (k + 1)` for k > 0 and `M = streams` for k = 0. Every request
gets a unique nonce at the beginning of the user message, so prefix caching
cannot retain the long synthetic document before reaching the nonce.

Long prompts are calibrated against the running model's `/tokenize` endpoint.
The requested context size, calibrated count, prompt bytes and prompt hash are
stored with every run. Existing prompt files without matching calibration
metadata are regenerated.

## Metric definitions

- `TTFT = first visible SSE token - request send`.
- Per-stream decode throughput excludes TTFT and the first token:
  `(completion_tokens - 1) / (response_complete - first_visible)`.
- Aggregate decode throughput uses the union of the completed decode windows:
  `sum(completion_tokens - 1) / (max(response_complete) - min(first_visible))`.
  The response boundary is intentional: with `ignore_eos=true`, completion
  usage can include generated EOS/control tokens that produce no visible SSE
  delta. Ending at the last visible delta would report impossible throughput.
- `request_total_tps` includes TTFT and exists only as a diagnostic; it is not
  the headline decode number.
- vLLM counters are differenced before/after every cell. Labeled counters are
  summed across engine/model labels while the original series are retained.

The result record includes hashes of the frozen baseline, measurement harness,
layer-0 sidecar and prompt plus the runtime image ID. A comparison without
matching fingerprints is invalid.

## MTP cost model

From the measured acceptance probabilities `p_i`, expected emitted tokens per
target verification are:

`Y_k = 1 + p_1 + p_1 p_2 + ... + product(p_1..p_k)`.

For the observed acceptance 0.715 / 0.676 / 0.663, this gives approximately
`Y_1=1.715`, `Y_2=2.198`, `Y_3=2.518`. In the simplified model
`R_k = Y_k / (1 + k*d)`, the incremental break-even limits are:

- k=1 beats k=0 when `d < 0.715`;
- k=2 beats k=1 when `d < 0.392`;
- k=3 beats k=2 when `d < 0.205`.

The real model is `R_k = Y_k / (T_k + kD + H_k)`: `T_k` may change with M,
`D` is draft-forward cost and `H_k` contains QSA metadata, scheduler and graph
overhead. Phase 0 measures their combined effect; it does not assume it away.

## Existing byte ledger

The packed Q2 format costs `0.25 + 2/128 = 0.265625` bytes per weight. At
top-k 10 this is 12.451 MiB per routed layer per token, or 597.65 MiB across 48
layers. The measured routed-kernel bandwidth near 39 GB/s is only about 14% of
the GB10 bandwidth figure, so the next profiler pass must determine whether
the loss is transactions, occupancy, dependency latency or launch/dispatch.

An NVFP4 hot cache is not a traffic optimization: it moves about 2.12x more
bytes per expert. It is viable only if its effective kernel throughput exceeds
the Q2 path by more than that ratio after dispatch and partition overhead.

## Promotion gates

Phase 1 starts only after:

1. all k/stream cells have three valid runs and nonzero MTP counters for k>0;
2. NCU is proven with a real counter read, or the nsys/CUPTI/CUDA-event fallback
   is recorded;
3. the selected k remains best across prose/code and is checked at long
   contexts rather than winning only one short prompt;
4. the current sidecar and baseline hashes are frozen in the result bundle.

Quality evaluation is a separate mandatory gate before hybrid/adaptive-k or
hot-cache work: weight relMSE is not an end-to-end quality measurement.
