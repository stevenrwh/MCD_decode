#!/usr/bin/env python3
"""
Export Vm tail tuples into concrete stroke records so we can rebuild glyphs
without relying on the Monu-CAD DXF fallback.

This script mirrors map_tail_offsets.py but keeps the full geometry for every
record instead of summarizing offsets.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from extract_font_components import extract_components
from font_components import iter_component_files
from mcd_to_dxf import collect_candidate_records
from summarize_font_components import LINE_TYPE, ARC_TYPE, assign_components, _looks_valid

GeometryRecord = Tuple[int, int, int, float, float, float, float]
VALID_TYPES = {LINE_TYPE, ARC_TYPE}


def load_tail_map(components_dir: Path) -> dict[str, tuple[int, int, int, int]]:
    mapping: dict[str, tuple[int, int, int, int]] = {}
    for definition in iter_component_files(components_dir):
        if not definition.records:
            continue
        tail = tuple(definition.records[0].values[-4:])
        mapping[definition.label] = tail
    return mapping


def group_records_by_tail(
    tail_map: dict[str, tuple[int, int, int, int]],
    grouped: Dict[str, List[GeometryRecord]],
) -> dict[tuple[int, int, int, int], list[GeometryRecord]]:
    buckets: dict[tuple[int, int, int, int], list[GeometryRecord]] = defaultdict(list)
    for label, records in grouped.items():
        tail = tail_map.get(label)
        if not tail:
            continue
        buckets[tail].extend(records)
    for records in buckets.values():
        records.sort(key=lambda rec: rec[0])
    return buckets


def _compress_ranges(offsets: Sequence[int], *, gap_threshold: int = 64) -> list[dict]:
    if not offsets:
        return []
    ranges: list[dict] = []
    start = offsets[0]
    prev = offsets[0]
    count = 1
    for offset in offsets[1:]:
        gap = offset - prev
        if gap > gap_threshold:
            ranges.append({"start": start, "end": prev, "count": count})
            start = offset
            count = 1
        else:
            count += 1
        prev = offset
    ranges.append({"start": start, "end": prev, "count": count})
    return ranges


def record_to_dict(record: GeometryRecord) -> dict:
    offset, layer, etype, x1, y1, x2, y2 = record
    if etype == LINE_TYPE:
        kind = "line"
    elif etype == ARC_TYPE:
        kind = "arc"
    else:
        kind = f"etype_{etype}"
    return {
        "offset": offset,
        "layer": layer,
        "type": kind,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Vm tail tuples into concrete stroke lists.")
    parser.add_argument("--fnt", type=Path, required=True, help="Path to Mcalf020.fnt.decompressed")
    parser.add_argument("--components", type=Path, required=True, help="Directory with Vm component slices")
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON file for the stroke table")
    parser.add_argument("--pad", type=int, default=0, help="Padding bytes used when slicing components (default 0)")
    parser.add_argument(
        "--gap-threshold",
        type=int,
        default=64,
        help="Maximum byte gap that still counts as part of the same contiguous offset run (default 64)",
    )
    parser.add_argument(
        "--coord-format",
        choices=("double", "float"),
        default="float",
        help="How to decode geometry coordinates (fonts typically need 'float')",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tail_map = load_tail_map(args.components)

    blob = args.fnt.read_bytes()
    components = extract_components(blob, pad=args.pad)
    intervals = [(start, end, label) for label, start, end, _ in components]
    records: List[GeometryRecord] = [
        rec
        for rec in collect_candidate_records(blob, coord_format=args.coord_format)
        if rec[2] in VALID_TYPES and _looks_valid(rec)
    ]
    grouped = assign_components(records, intervals)
    buckets = group_records_by_tail(tail_map, grouped)

    payload = {
        "source": str(args.fnt),
        "components_dir": str(args.components),
        "tails": [],
    }

    for tail in sorted(buckets.keys()):
        recs = buckets[tail]
        offsets = [rec[0] for rec in recs]
        payload["tails"].append(
            {
                "tail": list(tail),
                "record_count": len(recs),
                "ranges": _compress_ranges(offsets, gap_threshold=args.gap_threshold),
                "records": [record_to_dict(rec) for rec in recs],
            }
        )

    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[+] Stroke table written to {args.output} ({len(payload['tails'])} tail family/families)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
