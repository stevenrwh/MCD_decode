#!/usr/bin/env python3
"""
Summarize the metadata chunks inside Vm component records so we can see
which glyphs/tails reuse the same patterns.

Usage:
    python summarize_vm_meta.py --components FONTS/components_Mcalf020 \
        --output vm_meta_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from font_components import iter_component_files, POINT_SCALE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Vm component metadata chunks.")
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("vm_meta_summary.json"))
    return parser.parse_args()


def _chunk_records(values):
    if len(values) <= 8:
        return
    body = values[4:-4]
    for idx in range(0, len(body), 12):
        chunk = body[idx : idx + 12]
        if len(chunk) < 12:
            break
        meta = tuple(chunk[:4])
        coords = tuple(chunk[4:])
        yield meta, coords


def _meta_to_points(meta: tuple[int, int, int, int]):
    return tuple(val * POINT_SCALE for val in meta)


def main() -> int:
    args = parse_args()
    summary = defaultdict(lambda: {"count": 0, "labels": set(), "tails": set(), "examples": []})
    for definition in iter_component_files(args.components):
        tail = tuple(definition.records[0].values[-4:]) if definition.records else None
        for record in definition.records:
            for meta, coords in _chunk_records(record.values):
                entry = summary[meta]
                entry["count"] += 1
                entry["labels"].add(definition.label)
                if tail:
                    entry["tails"].add(tail)
                if len(entry["examples"]) < 3:
                    entry["examples"].append(
                        {
                            "label": definition.label,
                            "tail": tail,
                            "coords": [val * POINT_SCALE for val in coords[:8]],
                        }
                    )

    payload = [
        {
            "meta": meta,
            "meta_points": _meta_to_points(meta),
            "count": data["count"],
            "tails": sorted({str(t) for t in data["tails"]}),
            "labels": sorted(data["labels"]),
            "examples": data["examples"],
        }
        for meta, data in summary.items()
    ]
    payload.sort(key=lambda row: row["count"], reverse=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[+] Meta summary written to {args.output} ({len(payload)} unique entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
