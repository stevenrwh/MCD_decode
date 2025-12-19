#!/usr/bin/env python3
"""
Correlate component tail tuples with the global geometry offsets they reference.

This script:
  1. Loads a .fnt.decompressed blob and all component slices in a directory.
  2. Uses summarize_font_components' assignment logic to map every geometry
     record (line or arc) to its owning label.
  3. Reads the trailing 4-int tuple from each component record.
  4. Buckets the geometry offsets by tail tuple so we can inspect which global
     ranges each tuple references.

Example:
    python map_tail_offsets.py --fnt FONTS/Mcalf020.fnt.decompressed \
        --components FONTS/components_Mcalf020
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
        mapping[definition.label] = tail  # assuming one record per component (true for Vm).
    return mapping


def bucket_offsets_by_tail(
    tail_map: dict[str, tuple[int, int, int, int]],
    grouped: Dict[str, List[GeometryRecord]],
) -> dict[tuple[int, int, int, int], list[int]]:
    buckets: dict[tuple[int, int, int, int], list[int]] = defaultdict(list)
    for label, records in grouped.items():
        tail = tail_map.get(label)
        if not tail:
            continue
        for rec in records:
            buckets[tail].append(rec[0])
    for offsets in buckets.values():
        offsets.sort()
    return buckets


def _compress_ranges(offsets: Sequence[int], *, gap_threshold: int) -> list[dict]:
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


def summarize_bucket(offsets: Sequence[int], *, gap_threshold: int) -> dict:
    if not offsets:
        return {"count": 0, "ranges": []}
    gaps = [b - a for a, b in zip(offsets, offsets[1:])]
    return {
        "count": len(offsets),
        "min_offset": min(offsets),
        "max_offset": max(offsets),
        "avg_gap": sum(gaps) / len(gaps) if gaps else None,
        "ranges": _compress_ranges(offsets, gap_threshold=gap_threshold),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Map component tail tuples to global geometry offsets.")
    parser.add_argument("--fnt", type=Path, required=True, help="Path to *.fnt.decompressed blob")
    parser.add_argument("--components", type=Path, required=True, help="Directory with component slices")
    parser.add_argument("--pad", type=int, default=0, help="Padding used when slicing components (default 0)")
    parser.add_argument("--json", type=Path, help="Optional JSON output path for machine-readable summary")
    parser.add_argument(
        "--gap-threshold",
        type=int,
        default=64,
        help="Maximum delta (bytes) between offsets to keep them in the same range (default: 64)",
    )
    parser.add_argument(
        "--coord-format",
        choices=("double", "float"),
        default="float",
        help="How to decode coordinate fields inside the geometry records (default: float for fonts)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tail_map = load_tail_map(args.components)

    blob = args.fnt.read_bytes()
    components = extract_components(blob, pad=args.pad)
    intervals = [(start, end, label) for label, start, end, _ in components]
    records = [
        rec
        for rec in collect_candidate_records(blob, coord_format=args.coord_format)
        if rec[2] in VALID_TYPES and _looks_valid(rec)
    ]
    grouped = assign_components(records, intervals)

    buckets = bucket_offsets_by_tail(tail_map, grouped)
    summary = {
        str(tail): summarize_bucket(offsets, gap_threshold=args.gap_threshold)
        for tail, offsets in buckets.items()
    }

    for tail, offsets in buckets.items():
        stats = summary[str(tail)]
        print(f"tail {tail}: count={stats['count']} min={stats['min_offset']} max={stats['max_offset']}")

    if args.json:
        args.json.write_text(json.dumps({"tail_stats": summary}, indent=2), encoding="utf-8")
        print(f"[+] JSON summary written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
