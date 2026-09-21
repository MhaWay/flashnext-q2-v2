#!/usr/bin/env python3
"""NumPy reference for the Phase 1 adaptive quaternary format.

This module is deliberately slow and dependency-light.  It defines the
mathematics and golden behavior used to test the CUDA/Torch converter; it is
not the production converter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GROUP = 128


@dataclass(frozen=True)
class AdaptiveQ2:
    packed: np.ndarray
    scales: np.ndarray
    alpha_index: np.ndarray
    alpha_table: np.ndarray
    weighted_rel_mse: float
    rel_mse: float
    weighted_error: float
    weighted_reference: float


def fwht128(x: np.ndarray) -> np.ndarray:
    """Normalized block H128 along the final axis."""
    x = np.asarray(x)
    if x.shape[-1] % GROUP:
        raise ValueError("last dimension must be divisible by 128")
    original = x.shape
    y = x.astype(np.float32, copy=True).reshape(-1, GROUP)
    h = 1
    while h < GROUP:
        view = y.reshape(-1, GROUP // (2 * h), 2, h)
        a = view[:, :, 0, :].copy()
        b = view[:, :, 1, :].copy()
        view[:, :, 0, :] = a + b
        view[:, :, 1, :] = a - b
        y = view.reshape(-1, GROUP)
        h *= 2
    y *= np.float32(1.0 / np.sqrt(GROUP))
    return y.reshape(original)


def pack2(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint8)
    if codes.shape[-1] % 4:
        raise ValueError("code width must be divisible by four")
    return (codes[..., 0::4] | (codes[..., 1::4] << 2)
            | (codes[..., 2::4] << 4) | (codes[..., 3::4] << 6)).astype(np.uint8)


def unpack2(packed: np.ndarray) -> np.ndarray:
    p = np.asarray(packed, dtype=np.uint8)
    return np.stack((p & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3), axis=-1).reshape(
        *p.shape[:-1], p.shape[-1] * 4
    )


def _importance_matrix(importance: np.ndarray, rows: int, width: int) -> np.ndarray:
    importance = np.asarray(importance, dtype=np.float32)
    if importance.shape != (width,):
        raise ValueError(f"importance shape {importance.shape} != {(width,)}")
    importance = np.maximum(importance, 0)
    if not np.any(importance):
        importance = np.ones_like(importance)
    groups = importance.reshape(width // GROUP, GROUP)
    return np.broadcast_to(groups[None, :, :], (rows, width // GROUP, GROUP)).reshape(-1, GROUP)


def quantize_adaptive_q2(weight: np.ndarray, importance: np.ndarray,
                         alpha_table=(0.2, 0.25, 0.3, 1 / 3, 0.4, 0.5),
                         iterations: int = 2) -> AdaptiveQ2:
    weight = np.asarray(weight, dtype=np.float32)
    if weight.ndim != 2 or weight.shape[1] % GROUP:
        raise ValueError("weight must be [N,K] with K divisible by 128")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    alphas = np.asarray(alpha_table, dtype=np.float32)
    if alphas.ndim != 1 or not len(alphas) or np.any(alphas <= 0) or np.any(alphas >= 1):
        raise ValueError("alpha table must contain values in (0,1)")

    rotated = fwht128(weight)
    groups = rotated.reshape(-1, GROUP)
    h = _importance_matrix(importance, weight.shape[0], weight.shape[1])
    best_error = np.full(groups.shape[0], np.inf, dtype=np.float64)
    best_scale = np.zeros(groups.shape[0], dtype=np.float32)
    best_codes = np.zeros_like(groups, dtype=np.uint8)
    best_alpha = np.zeros(groups.shape[0], dtype=np.uint8)

    for alpha_i, alpha in enumerate(alphas):
        scale = np.max(np.abs(groups), axis=1)
        for _ in range(iterations):
            safe = np.where(scale > 0, scale, 1)
            high = np.abs(groups) / safe[:, None] > (1 + alpha) * 0.5
            magnitude = np.where(high, 1.0, alpha).astype(np.float32)
            base = np.where(groups < 0, -magnitude, magnitude)
            denominator = np.sum(h * base * base, axis=1)
            numerator = np.sum(h * groups * base, axis=1)
            scale = np.maximum(numerator / np.maximum(denominator, 1e-20), 0)
        safe = np.where(scale > 0, scale, 1)
        high = np.abs(groups) / safe[:, None] > (1 + alpha) * 0.5
        positive = groups >= 0
        magnitude = np.where(high, 1.0, alpha).astype(np.float32)
        quantized = np.where(positive, magnitude, -magnitude) * scale[:, None]
        error = np.sum(h * np.square(groups - quantized), axis=1, dtype=np.float64)
        codes = np.where(positive, np.where(high, 3, 2), np.where(high, 0, 1)).astype(np.uint8)
        take = error < best_error
        best_error[take] = error[take]
        best_scale[take] = scale[take]
        best_codes[take] = codes[take]
        best_alpha[take] = alpha_i

    stored_scales = best_scale.astype(np.float16)
    stored_alphas = alphas.astype(np.float16)
    dequant = dequantize_adaptive_q2(
        pack2(best_codes).reshape(weight.shape[0], weight.shape[1] // 4),
        stored_scales.reshape(weight.shape[0], weight.shape[1] // GROUP),
        best_alpha.reshape(weight.shape[0], weight.shape[1] // GROUP),
        stored_alphas,
    )
    weighted_ref = np.sum(h * groups * groups, dtype=np.float64)
    stored_error = np.sum(h * np.square(groups - dequant.reshape(-1, GROUP)), dtype=np.float64)
    weighted_rel = float(stored_error / max(weighted_ref, 1e-30))
    rel = float(np.sum(np.square(rotated - dequant), dtype=np.float64)
                / max(np.sum(np.square(rotated), dtype=np.float64), 1e-30))
    return AdaptiveQ2(
        packed=pack2(best_codes).reshape(weight.shape[0], weight.shape[1] // 4),
        scales=stored_scales.reshape(weight.shape[0], weight.shape[1] // GROUP),
        alpha_index=best_alpha.reshape(weight.shape[0], weight.shape[1] // GROUP),
        alpha_table=stored_alphas,
        weighted_rel_mse=weighted_rel,
        rel_mse=rel,
        weighted_error=float(stored_error),
        weighted_reference=float(weighted_ref),
    )


def dequantize_adaptive_q2(packed: np.ndarray, scales: np.ndarray,
                           alpha_index: np.ndarray, alpha_table: np.ndarray) -> np.ndarray:
    packed = np.asarray(packed, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.float32)
    alpha_index = np.asarray(alpha_index, dtype=np.uint8)
    alpha_table = np.asarray(alpha_table, dtype=np.float32)
    rows, packed_width = packed.shape
    width = packed_width * 4
    expected = (rows, width // GROUP)
    if scales.shape != expected or alpha_index.shape != expected:
        raise ValueError("scale/alpha-index shape mismatch")
    if np.any(alpha_index >= len(alpha_table)):
        raise ValueError("alpha index outside table")
    codes = unpack2(packed).reshape(rows, width // GROUP, GROUP)
    high = (codes == 0) | (codes == 3)
    positive = codes >= 2
    alpha = alpha_table[alpha_index][..., None]
    magnitude = np.where(high, 1.0, alpha)
    base = np.where(positive, magnitude, -magnitude)
    return (base * scales[..., None]).reshape(rows, width).astype(np.float32)


def quantize_symmetric_q4(weight_rotated: np.ndarray, importance: np.ndarray):
    """Reference signed-int4 precision-island codec (values -7..7)."""
    weight = np.asarray(weight_rotated, dtype=np.float32)
    if weight.ndim != 2 or weight.shape[1] % GROUP:
        raise ValueError("weight must be [N,K] with K divisible by 128")
    groups = weight.reshape(-1, GROUP)
    h = _importance_matrix(importance, weight.shape[0], weight.shape[1])
    scale = np.max(np.abs(groups), axis=1) / 7.0
    safe = np.where(scale > 0, scale, 1)
    q = np.clip(np.rint(groups / safe[:, None]), -7, 7).astype(np.int8)
    # One weighted LS refit with the chosen integer codes.
    den = np.sum(h * q.astype(np.float32) ** 2, axis=1)
    scale = np.maximum(
        np.sum(h * groups * q.astype(np.float32), axis=1) / np.maximum(den, 1e-20), 0
    )
    unsigned = (q.astype(np.int16) + 8).astype(np.uint8)
    packed = (unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)).astype(np.uint8)
    stored_scale = scale.astype(np.float16)
    recon = q.astype(np.float32) * stored_scale.astype(np.float32)[:, None]
    err = np.sum(h * np.square(groups - recon), dtype=np.float64)
    ref = np.sum(h * groups * groups, dtype=np.float64)
    return (
        packed.reshape(weight.shape[0], weight.shape[1] // 2),
        stored_scale.reshape(weight.shape[0], weight.shape[1] // GROUP),
        float(err / max(ref, 1e-30)),
    )
