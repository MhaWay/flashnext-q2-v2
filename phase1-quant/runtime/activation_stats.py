#!/usr/bin/env python3
"""Calibration-only accumulator for the FlashNext routed MoE runtime.

Integration points in `FlashNextQ2OfflineMoEMethod.apply`:

* after `xrot` and top-k `ids` exist: `stats.add_w13(xrot, ids)`;
* after route-major `zrot` exists: `stats.add_w2(zrot, ids.reshape(-1))`.

This module is not imported in normal serving.  Calibration mode trades speed
for expert-conditioned activation statistics and flushes one file per layer.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from safetensors.torch import save_file


class ExpertActivationStats:
    def __init__(self, layer: int, num_experts: int, hidden: int, intermediate: int,
                 device: torch.device | str):
        self.layer = int(layer)
        self.num_experts = int(num_experts)
        self.hidden = int(hidden)
        self.intermediate = int(intermediate)
        self.device = torch.device(device)
        self.w13_sq_sum = torch.zeros((num_experts, hidden), dtype=torch.float32, device=self.device)
        self.w2_sq_sum = torch.zeros((num_experts, intermediate), dtype=torch.float32, device=self.device)
        self.route_count = torch.zeros(num_experts, dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def add_w13(self, xrot: torch.Tensor, expert_ids: torch.Tensor) -> None:
        if expert_ids.ndim == 1:
            expert_ids = expert_ids.unsqueeze(0)
        if xrot.ndim != 2 or expert_ids.ndim != 2 or xrot.shape[0] != expert_ids.shape[0]:
            raise ValueError("w13 calibration expects xrot[M,H], expert_ids[M,K]")
        values = xrot.float().square()
        ones = torch.ones(xrot.shape[0], dtype=torch.int64, device=self.device)
        # Avoid materializing [M, topk, H]. Calibration is offline and ten
        # index_add calls are preferable to a multi-GiB temporary.
        for slot in range(expert_ids.shape[1]):
            ids = expert_ids[:, slot].long()
            valid = (ids >= 0) & (ids < self.num_experts)
            self.w13_sq_sum.index_add_(0, ids[valid], values[valid])
            self.route_count.index_add_(0, ids[valid], ones[valid])

    @torch.no_grad()
    def add_w2(self, zrot: torch.Tensor, flat_expert_ids: torch.Tensor) -> None:
        if zrot.ndim != 2 or flat_expert_ids.ndim != 1 or zrot.shape[0] != flat_expert_ids.shape[0]:
            raise ValueError("w2 calibration expects zrot[routes,I], expert_ids[routes]")
        ids = flat_expert_ids.long()
        valid = (ids >= 0) & (ids < self.num_experts)
        self.w2_sq_sum.index_add_(0, ids[valid], zrot[valid].float().square())

    @torch.no_grad()
    def flush(self, root: str | os.PathLike, metadata: dict[str, str] | None = None) -> Path:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        destination = root / f"layer-{self.layer:02d}.safetensors"
        temporary = root / f".{destination.name}.{os.getpid()}.tmp"
        state = {
            "w13_sq_sum": self.w13_sq_sum.cpu().contiguous(),
            "w2_sq_sum": self.w2_sq_sum.cpu().contiguous(),
            "route_count": self.route_count.cpu().contiguous(),
        }
        md = {
            "format": "flashnext-q2-activation-calibration-v1",
            "layer": str(self.layer),
            "num_experts": str(self.num_experts),
            "hidden": str(self.hidden),
            "intermediate": str(self.intermediate),
        }
        md.update(metadata or {})
        save_file(state, str(temporary), metadata=md)
        os.replace(temporary, destination)
        return destination
