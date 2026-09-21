#!/usr/bin/env python3
"""Load and validate expert-conditioned activation calibration statistics."""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from source_reader import HIDDEN, INTER, NUM_EXPERTS


def _conditioned_moment(sq_sum: torch.Tensor, count: torch.Tensor, minimum_routes: int) -> torch.Tensor:
    count = count.to(torch.float32).clamp_min(0)
    observed = count >= minimum_routes
    if bool(torch.any(observed)):
        layer_mean = (sq_sum[observed].sum(dim=0)
                      / count[observed].sum().clamp_min(1.0))
    else:
        layer_mean = torch.ones(sq_sum.shape[1], dtype=torch.float32)
    moment = sq_sum.float() / count.clamp_min(1.0).unsqueeze(1)
    moment[~observed] = layer_mean
    return moment.clamp_min(0)


def load_layer_calibration(root: Path, layer: int, minimum_routes: int = 32):
    path = root / f"layer-{layer:02d}.safetensors"
    if not path.is_file():
        raise FileNotFoundError(path)
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        keys = set(handle.keys())
        required = {"w13_sq_sum", "w2_sq_sum", "route_count"}
        if not required <= keys:
            raise ValueError(f"{path}: missing {sorted(required - keys)}")
        w13_sum = handle.get_tensor("w13_sq_sum")
        w2_sum = handle.get_tensor("w2_sq_sum")
        count = handle.get_tensor("route_count")
    if tuple(w13_sum.shape) != (NUM_EXPERTS, HIDDEN):
        raise ValueError(f"{path}: bad w13_sq_sum shape {tuple(w13_sum.shape)}")
    if tuple(w2_sum.shape) != (NUM_EXPERTS, INTER):
        raise ValueError(f"{path}: bad w2_sq_sum shape {tuple(w2_sum.shape)}")
    if tuple(count.shape) != (NUM_EXPERTS,):
        raise ValueError(f"{path}: bad route_count shape {tuple(count.shape)}")
    return {
        "path": path,
        "metadata": metadata,
        "count": count.to(torch.int64),
        "w13_moment": _conditioned_moment(w13_sum, count, minimum_routes),
        "w2_moment": _conditioned_moment(w2_sum, count, minimum_routes),
    }
