#!/usr/bin/env python3
"""CUDA/Torch implementation of the Phase 1 adaptive Q2/Q4 codecs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


GROUP = 128


@dataclass
class TorchAdaptiveQ2:
    packed: torch.Tensor
    scales: torch.Tensor
    alpha_index: torch.Tensor
    weighted_rel_mse: float
    rel_mse: float
    weighted_error: float
    weighted_reference: float


def fwht128(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % GROUP:
        raise ValueError("last dimension must be divisible by 128")
    shape = x.shape
    y = x.reshape(-1, GROUP).float()
    h = 1
    while h < GROUP:
        y = y.reshape(-1, GROUP // (2 * h), 2, h)
        a = y[:, :, 0, :]
        b = y[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, GROUP)
        h *= 2
    return y.mul_(GROUP ** -0.5).reshape(shape)


def pack2(codes: torch.Tensor) -> torch.Tensor:
    codes = codes.to(torch.uint8)
    return (codes[..., 0::4] | (codes[..., 1::4] << 2)
            | (codes[..., 2::4] << 4) | (codes[..., 3::4] << 6)).contiguous()


def unpack2(packed: torch.Tensor) -> torch.Tensor:
    p = packed.to(torch.uint8)
    return torch.stack((p & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3), dim=-1).reshape(
        *p.shape[:-1], p.shape[-1] * 4
    )


def _importance_groups(importance: torch.Tensor, width: int) -> torch.Tensor:
    if tuple(importance.shape) != (width,):
        raise ValueError(f"importance shape {tuple(importance.shape)} != {(width,)}")
    h = importance.float().clamp_min(0)
    if not bool(torch.any(h > 0)):
        h = torch.ones_like(h)
    return h.reshape(width // GROUP, GROUP)


@torch.no_grad()
def quantize_adaptive_q2(weight: torch.Tensor, importance: torch.Tensor,
                         alpha_table: torch.Tensor, *, iterations: int = 2,
                         chunk_groups: int = 4096) -> TorchAdaptiveQ2:
    if weight.ndim != 2 or weight.shape[1] % GROUP:
        raise ValueError("weight must be [N,K] with K divisible by 128")
    if alpha_table.ndim != 1 or alpha_table.numel() > 255:
        raise ValueError("alpha table must be one-dimensional with <=255 entries")
    if bool(torch.any(alpha_table <= 0)) or bool(torch.any(alpha_table >= 1)):
        raise ValueError("alpha table entries must be in (0,1)")
    rows, width = weight.shape
    rotated = fwht128(weight)
    groups = rotated.reshape(-1, GROUP)
    h_groups = _importance_groups(importance.to(weight.device), width)
    groups_per_row = width // GROUP
    packed_out = torch.empty((groups.shape[0], GROUP // 4), dtype=torch.uint8, device=weight.device)
    scales_out = torch.empty(groups.shape[0], dtype=torch.float16, device=weight.device)
    alpha_out = torch.empty(groups.shape[0], dtype=torch.uint8, device=weight.device)
    stored_alpha_table = alpha_table.to(torch.float16).float()
    weighted_error = torch.zeros((), dtype=torch.float64, device=weight.device)
    weighted_ref = torch.zeros((), dtype=torch.float64, device=weight.device)

    for start in range(0, groups.shape[0], chunk_groups):
        stop = min(start + chunk_groups, groups.shape[0])
        g = groups[start:stop]
        group_ids = torch.arange(start, stop, device=weight.device) % groups_per_row
        h = h_groups[group_ids]
        best_error = torch.full((len(g),), torch.inf, dtype=torch.float64, device=weight.device)
        best_scale = torch.zeros(len(g), dtype=torch.float32, device=weight.device)
        best_codes = torch.zeros_like(g, dtype=torch.uint8)
        best_alpha = torch.zeros(len(g), dtype=torch.uint8, device=weight.device)
        for alpha_i in range(alpha_table.numel()):
            alpha = alpha_table[alpha_i].float()
            scale = g.abs().amax(dim=1)
            for _ in range(iterations):
                safe = torch.where(scale > 0, scale, torch.ones_like(scale))
                high = g.abs() / safe[:, None] > (1 + alpha) * 0.5
                magnitude = torch.where(high, torch.ones_like(g), alpha.expand_as(g))
                base = torch.where(g < 0, -magnitude, magnitude)
                denominator = (h * base * base).sum(dim=1).clamp_min(1e-20)
                scale = ((h * g * base).sum(dim=1) / denominator).clamp_min(0)
            safe = torch.where(scale > 0, scale, torch.ones_like(scale))
            high = g.abs() / safe[:, None] > (1 + alpha) * 0.5
            positive = g >= 0
            magnitude = torch.where(high, torch.ones_like(g), alpha.expand_as(g))
            quantized = torch.where(positive, magnitude, -magnitude) * scale[:, None]
            error = (h.double() * (g.double() - quantized.double()).square()).sum(dim=1)
            codes = torch.where(
                positive,
                torch.where(high, torch.full_like(g, 3, dtype=torch.uint8),
                            torch.full_like(g, 2, dtype=torch.uint8)),
                torch.where(high, torch.zeros_like(g, dtype=torch.uint8),
                            torch.ones_like(g, dtype=torch.uint8)),
            )
            take = error < best_error
            best_error = torch.where(take, error, best_error)
            best_scale = torch.where(take, scale, best_scale)
            best_codes = torch.where(take[:, None], codes, best_codes)
            best_alpha = torch.where(take, torch.full_like(best_alpha, alpha_i), best_alpha)
        packed_out[start:stop] = pack2(best_codes)
        stored_scale = best_scale.to(torch.float16)
        scales_out[start:stop] = stored_scale
        alpha_out[start:stop] = best_alpha
        chosen_alpha = stored_alpha_table[best_alpha.long()]
        high = (best_codes == 0) | (best_codes == 3)
        positive = best_codes >= 2
        base = torch.where(high, torch.ones_like(g), chosen_alpha[:, None])
        base = torch.where(positive, base, -base)
        stored_reconstruction = base * stored_scale.float()[:, None]
        weighted_error += (h.double() * (g.double() - stored_reconstruction.double()).square()).sum()
        weighted_ref += (h.double() * g.double().square()).sum()

    packed = packed_out.reshape(rows, width // 4)
    scales = scales_out.reshape(rows, groups_per_row)
    alpha_index = alpha_out.reshape(rows, groups_per_row)
    reconstructed = dequantize_adaptive_q2(packed, scales, alpha_index, stored_alpha_table)
    rel = ((rotated.double() - reconstructed.double()).square().sum()
           / rotated.double().square().sum().clamp_min(1e-30))
    return TorchAdaptiveQ2(
        packed, scales, alpha_index,
        float((weighted_error / weighted_ref.clamp_min(1e-30)).item()),
        float(rel.item()),
        float(weighted_error.item()),
        float(weighted_ref.item()),
    )


def dequantize_adaptive_q2(packed: torch.Tensor, scales: torch.Tensor,
                           alpha_index: torch.Tensor, alpha_table: torch.Tensor) -> torch.Tensor:
    rows, packed_width = packed.shape
    width = packed_width * 4
    expected = (rows, width // GROUP)
    if tuple(scales.shape) != expected or tuple(alpha_index.shape) != expected:
        raise ValueError("scale/alpha-index shape mismatch")
    codes = unpack2(packed).reshape(rows, width // GROUP, GROUP)
    high = (codes == 0) | (codes == 3)
    positive = codes >= 2
    alpha = alpha_table.to(packed.device).float()[alpha_index.long()].unsqueeze(-1)
    magnitude = torch.where(high, torch.ones_like(codes, dtype=torch.float32), alpha)
    base = torch.where(positive, magnitude, -magnitude)
    return (base * scales.float().unsqueeze(-1)).reshape(rows, width)


@torch.no_grad()
def quantize_symmetric_q4(rotated: torch.Tensor, importance: torch.Tensor):
    if rotated.ndim != 2 or rotated.shape[1] % GROUP:
        raise ValueError("rotated weight must be [N,K] with K divisible by 128")
    rows, width = rotated.shape
    groups = rotated.float().reshape(-1, GROUP)
    h_base = _importance_groups(importance.to(rotated.device), width)
    h = h_base[torch.arange(groups.shape[0], device=rotated.device) % (width // GROUP)]
    scale = groups.abs().amax(dim=1) / 7.0
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = torch.round(groups / safe[:, None]).clamp(-7, 7)
    den = (h * q.square()).sum(dim=1).clamp_min(1e-20)
    scale = ((h * groups * q).sum(dim=1) / den).clamp_min(0)
    unsigned = (q.to(torch.int16) + 8).to(torch.uint8)
    packed = (unsigned[:, 0::2] | (unsigned[:, 1::2] << 4)).reshape(rows, width // 2)
    stored_scale = scale.to(torch.float16)
    recon = q * stored_scale.float()[:, None]
    error = (h.double() * (groups.double() - recon.double()).square()).sum()
    ref = (h.double() * groups.double().square()).sum().clamp_min(1e-30)
    return (packed, stored_scale.reshape(rows, width // GROUP),
            float((error / ref).item()), float(error.item()), float(ref.item()))
