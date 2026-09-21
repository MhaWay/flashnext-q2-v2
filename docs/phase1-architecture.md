# Phase 1 architecture decision record

## Decision

Generate `flashnext-q2-adaptive-v1` as an immutable candidate alongside the
v0.11.4 fixed-alpha sidecars:

1. normalized H128 rotation remains unchanged;
2. every group of 128 weights chooses one alpha from
   `{0.20, 0.25, 0.30, 1/3, 0.40, 0.50}`;
3. scale and alpha minimize activation-second-moment-weighted error;
4. under-observed experts use the layer-mean moment, never a zero objective;
5. precision islands promote complete gate/up/down experts to signed Q4;
6. the global selector maximizes measured error recovery per added byte.

The output is intentionally not loadable by v0.11.4. Phase 2 must add alpha
lookup and Q4 dispatch kernels, then pass numerical and end-to-end quality
gates before the new directory can become a serving candidate.

## Why this, before a new kernel

The Phase 0 performance data shows the current decode kernel leaves most GB10
bandwidth unused, but that does not establish the model-quality margin at two
bits. Phase 1 therefore improves the information allocation without changing
the proven rotation or packing. It also produces sensitivity data that makes
the Q4 budget an evidence-based choice rather than an expert-frequency guess.

## Calibration semantics

For expert `e`, the objective for a rotated weight group is

\[
  \sum_j E[x_j^2 \mid e\text{ routed}] (w_j - \hat w_j)^2.
\]

W13 uses the rotated layer input and W2 uses the rotated SwiGLU output. Route
counts convert conditional per-expert errors back into expected workload cost
for the global island selector. Cross-coordinate covariance is deliberately
not stored: diagonal moments need about 300 MiB for all 48 layers and can be
collected online, while full covariance is impractical.

## Phase boundaries

| Phase | Produces | May change serving defaults? |
|---|---|---:|
| 0 | Frozen performance and quality evidence | No |
| 1 | Calibration, sensitivity map, adaptive sidecars | No |
| 2 | Correct adaptive-Q2/Q4 runtime kernels | Only after gates |
| 3 | Decode/prefill tuning and MTP integration | Only after A/B |

## Promotion gates

- 48 calibration files from one identified corpus/run.
- 24,576 sensitivity rows and hashes for every layer file.
- 48 base sidecars plus exactly the Q4 companions named by the precision map.
- `verify_sidecars.py` reports `status=ok` for 48 layers.
- Kernel output agrees with the Torch/reference codec within a declared bound.
- No-thinking quality suite compared against current Q2 and a named reference;
  truncations are invalid rows, not failures.
- Decode and TTFT are measured independently; MTP acceptance alone is never a
  performance result.
