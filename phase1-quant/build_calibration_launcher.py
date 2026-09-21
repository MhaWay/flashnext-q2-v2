#!/usr/bin/env python3
"""Derive a calibration-only launcher from the frozen v0.11.4 control.

The patch is intentionally tied to the exact control hash.  It instruments
only the route-direct prefill path and stores cumulative statistics under the
already persistent profile mount.  The original launcher is never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path


BASELINE_SHA256 = "006a19accf3067f51cd2ed8409af64b569494dbf1dd049f4d0f148f3c33eda58"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return text.replace(old, new, 1)


def replace_last(text: str, old: str, new: str, label: str) -> str:
    position = text.rfind(old)
    if position < 0:
        raise RuntimeError(f"{label}: anchor not found")
    return text[:position] + new + text[position + len(old):]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    raw = args.baseline.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != BASELINE_SHA256:
        raise SystemExit(f"baseline sha256 {actual} != frozen {BASELINE_SHA256}")
    text = raw.decode()
    stats_source = (Path(__file__).resolve().parent / "runtime/activation_stats.py").read_text()
    method_marker = '  cat > "$PLUGIN_DIR/flashnext_q2/method.py" <<\'PY\'\n'
    stats_heredoc = (
        '  cat > "$PLUGIN_DIR/flashnext_q2/activation_stats.py" <<\'PYPHASE1CAL\'\n'
        + stats_source + "\nPYPHASE1CAL\n\n" + method_marker
    )
    text = replace_once(text, method_marker, stats_heredoc, "plugin module")
    text = replace_once(
        text,
        "from .fast_init import in_profile_dummy\n",
        "from .fast_init import in_profile_dummy\n"
        "from .activation_stats import ExpertActivationStats\n",
        "method import",
    )
    text = replace_once(
        text,
        "        self._fastinit_bypass_reported = False\n",
        "        self._fastinit_bypass_reported = False\n"
        "        self._phase1_stats = None\n"
        "        self._phase1_calibration_calls = 0\n",
        "method init",
    )
    layer_anchor = (
        "        if not m:\n"
        "            raise RuntimeError(f\"cannot derive layer id from prefix {self.prefix!r}\")\n"
        "        return int(m.group(1))\n"
    )
    helpers = layer_anchor + '''

    def _phase1_get_stats(self, x):
        root = os.environ.get("FLASHNEXT_PHASE1_CALIBRATION_DIR", "").strip()
        if not root:
            return None
        if self._phase1_stats is None:
            self._phase1_stats = ExpertActivationStats(
                self._layer_id(), 512, 2560, 640, x.device)
            _p(f"Phase 1 calibration active at layer {self._layer_id():02d}: {root}")
        return self._phase1_stats

    def _phase1_add_w13(self, xrot, ids):
        stats = self._phase1_get_stats(xrot)
        if stats is not None:
            stats.add_w13(xrot, ids)
        return xrot

    def _phase1_add_w2(self, zrot, ids):
        stats = self._phase1_get_stats(zrot)
        if stats is not None:
            stats.add_w2(zrot.reshape(-1, zrot.shape[-1]), ids.reshape(-1))
            self._phase1_calibration_calls += 1
            every = max(1, int(os.environ.get(
                "FLASHNEXT_PHASE1_CALIBRATION_FLUSH_EVERY", "1")))
            if self._phase1_calibration_calls % every == 0:
                stats.flush(os.environ["FLASHNEXT_PHASE1_CALIBRATION_DIR"], {
                    "baseline_sha256": "006a19accf3067f51cd2ed8409af64b569494dbf1dd049f4d0f148f3c33eda58",
                    "capture_path": "route-direct-prefill-v0.7",
                })
        return zrot
'''
    text = replace_once(text, layer_anchor, helpers, "layer helper")
    route_direct = (
        "                xrot = fwht128(xc).to(xc.dtype).contiguous()\n"
        "                z = q2_prefill_w13_swiglu(xrot, layer.w13_weight, layer.w13_weight_scale_inv, ids)\n"
        "                zshape = z.shape\n"
        "                zrot = fwht128(z.reshape(-1, zshape[-1])).to(z.dtype).reshape(zshape).contiguous()\n"
    )
    instrumented = (
        "                xrot = self._phase1_add_w13(\n"
        "                    fwht128(xc).to(xc.dtype).contiguous(), ids)\n"
        "                z = q2_prefill_w13_swiglu(xrot, layer.w13_weight, layer.w13_weight_scale_inv, ids)\n"
        "                zshape = z.shape\n"
        "                zrot = self._phase1_add_w2(\n"
        "                    fwht128(z.reshape(-1, zshape[-1])).to(z.dtype).reshape(zshape).contiguous(), ids)\n"
    )
    text = replace_once(text, route_direct, instrumented, "route-direct prefill")
    docker_anchor = (
        '    -e FLASHNEXT_Q2_PREFILL_V8_ENABLE="$Q2_PREFILL_V8_ENABLE" \\\n'
        '    -e FLASHNEXT_Q2_PREFILL_V8_MIN_M="$Q2_PREFILL_V8_MIN_M" \\\n'
    )
    docker_env = docker_anchor + (
        '    -e FLASHNEXT_PHASE1_CALIBRATION_DIR=/profile-data/phase1-calibration \\\n'
        '    -e FLASHNEXT_PHASE1_CALIBRATION_FLUSH_EVERY="${FLASHNEXT_PHASE1_CALIBRATION_FLUSH_EVERY:-1}" \\\n'
    )
    text = replace_last(text, docker_anchor, docker_env, "serve calibration environment")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(f".{args.out.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.chmod(0o755)
    os.replace(temporary, args.out)
    print(f"wrote {args.out}")
    print(f"source_sha256={actual}")
    print("calibration_path=<current profile dir>/phase1-calibration")


if __name__ == "__main__":
    main()
