#!/usr/bin/env python3
"""Compute Prometheus metric deltas (before -> after) into structured JSON.

Handles plain counters, labeled counters, and Prometheus histograms
(bucket/ count / sum families). Never fails the sweep: on any problem it
writes empty deltas and returns 0.
"""
import json
import re
import sys

LINE = re.compile(r'^(?P<name>[a-zA-Z_:][\w:]*)(?P<labels>\{[^}]*\})?\s+(?P<value>[-+0-9.eE]+)(\s+\d+)?\s*$')


def parse(path):
    metrics = {}
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                if line.startswith("#") or not line.strip():
                    continue
                m = LINE.match(line.strip())
                if not m:
                    continue
                name = m.group("name")
                labels = m.group("labels") or ""
                try:
                    value = float(m.group("value"))
                except ValueError:
                    continue
                metrics.setdefault(name, {})[labels] = value
    except OSError:
        pass
    return metrics


def main():
    if len(sys.argv) != 4:
        sys.stderr.write("usage: metrics_delta.py before after out.json\n")
        return 0
    before = parse(sys.argv[1])
    after = parse(sys.argv[2])
    counters, series, hists = {}, {}, {}
    for name, entries in after.items():
        b = before.get(name, {})
        delta = {k: v - b.get(k, 0.0) for k, v in entries.items()}
        if name.endswith("_bucket"):
            base = name[: -len("_bucket")]
            by_le = {}
            for labels, dv in delta.items():
                le = "0"
                for part in labels.split(","):
                    if 'le="' in part:
                        le = part.split('le="', 1)[1].strip('"}')
                by_le[le] = by_le.get(le, 0.0) + dv
            hists[base] = by_le
        elif name.endswith(("_count", "_sum")):
            pass
        else:
            # vLLM counters normally carry engine/model labels. Always expose
            # the aggregate as well as the per-label series: callers asking
            # for the total must not silently get zero merely because a label
            # was added by a newer runtime.
            counters[name] = sum(delta.values())
            if len(entries) > 1 or any(delta):
                series[name] = {k[:500]: v for k, v in delta.items()}
    out = {"counters_delta": counters,
           "counter_series_deltas": series,
           "histogram_bucket_deltas": hists}
    try:
        with open(sys.argv[3], "w") as fh:
            json.dump(out, fh)
    except OSError as exc:
        sys.stderr.write(f"metrics_delta: cannot write output: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
