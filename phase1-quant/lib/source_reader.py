#!/usr/bin/env python3
"""Streaming reader for native Qwen3.8-Flash-Next routed BF16 weights."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from safetensors import safe_open


NUM_LAYERS = 48
NUM_EXPERTS = 512
HIDDEN = 2560
INTER = 640
W13_OUT = 2 * INTER

PER_EXPERT_RE = re.compile(
    r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
FUSED_RE = re.compile(
    r"^(?:model\.)?(?:language_model\.)?layers\.(\d+)\.mlp\.experts\."
    r"(gate_up_proj|down_proj)(?:\.weight)?$"
)


def load_weight_map(snapshot: Path) -> dict[str, str]:
    index = snapshot / "model.safetensors.index.json"
    if not index.is_file():
        raise FileNotFoundError(index)
    weight_map = json.loads(index.read_text()).get("weight_map") or {}
    if not weight_map:
        raise ValueError(f"empty weight map in {index}")
    return weight_map


def build_source_map(weight_map: dict[str, str]) -> dict:
    fused, split = {}, {}
    for name, shard in weight_map.items():
        lower = name.lower()
        if "mtp." in lower or lower.startswith("mtp.") or ".visual." in lower or ".vision." in lower:
            continue
        match = FUSED_RE.match(name)
        if match:
            layer, projection = int(match.group(1)), match.group(2)
            if 0 <= layer < NUM_LAYERS:
                fused[(layer, projection)] = (name, shard)
            continue
        match = PER_EXPERT_RE.match(name)
        if match:
            layer, expert, projection = int(match.group(1)), int(match.group(2)), match.group(3)
            if 0 <= layer < NUM_LAYERS and 0 <= expert < NUM_EXPERTS:
                split[(layer, expert, projection)] = (name, shard)
    if len(fused) == NUM_LAYERS * 2:
        return {"layout": "fused", "map": fused}
    if len(split) == NUM_LAYERS * NUM_EXPERTS * 3:
        return {"layout": "per-expert", "map": split}
    raise ValueError(
        f"routed layout incomplete: fused={len(fused)}/{NUM_LAYERS * 2}, "
        f"split={len(split)}/{NUM_LAYERS * NUM_EXPERTS * 3}"
    )


def _slice_to_cuda(snapshot: Path, item, slices, expected_shape) -> torch.Tensor:
    name, shard = item
    path = snapshot / shard
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensor_slice = handle.get_slice(name)
        if tuple(tensor_slice.get_shape()) != tuple(expected_shape):
            raise ValueError(f"{name}: {tuple(tensor_slice.get_shape())} != {tuple(expected_shape)}")
        tensor = tensor_slice[slices]
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name}: expected bfloat16, got {tensor.dtype}")
    return tensor.to("cuda", non_blocking=False)


def load_matrix(snapshot: Path, source: dict, layer: int, expert: int, projection: str) -> torch.Tensor:
    if source["layout"] == "fused":
        fused_name = "gate_up_proj" if projection in ("gate_proj", "up_proj") else "down_proj"
        item = source["map"][(layer, fused_name)]
        if projection == "gate_proj":
            return _slice_to_cuda(snapshot, item, (expert, slice(0, INTER), slice(None)),
                                  (NUM_EXPERTS, W13_OUT, HIDDEN))
        if projection == "up_proj":
            return _slice_to_cuda(snapshot, item, (expert, slice(INTER, W13_OUT), slice(None)),
                                  (NUM_EXPERTS, W13_OUT, HIDDEN))
        return _slice_to_cuda(snapshot, item, (expert, slice(None), slice(None)),
                              (NUM_EXPERTS, HIDDEN, INTER))
    item = source["map"][(layer, expert, projection)]
    expected = (INTER, HIDDEN) if projection in ("gate_proj", "up_proj") else (HIDDEN, INTER)
    return _slice_to_cuda(snapshot, item, (slice(None), slice(None)), expected)
