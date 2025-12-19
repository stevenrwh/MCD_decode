#!/usr/bin/env python3
"""
Group the raw MAIN font geometry (lines/circles) by glyph/component label.

Usage:
    python summarize_font_components.py FONTS/Mcalf092.fnt.decompressed \
        --json FONTS/MAIN_component_summary.json

This script uses the label offsets discovered earlier to carve the payload into
per-glyph ranges, then assigns every candidate record (etype=2 line) to the
range that contains it. The result is a JSON summary with per-letter line
counts, bounding boxes, and layer ids, plus a short console table.
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from extract_font_components import extract_components
from mcd_to_dxf import collect_candidate_records

COORD_LIMIT = 1.0
LINE_TYPE = 2
ARC_TYPE = 3
VALID_TYPES = {LINE_TYPE, ARC_TYPE}


GeometryRecord = Tuple[int, int, int, float, float, float, float]


def _looks_valid(rec: GeometryRecord) -> bool:
    _, _, _, x1, y1, x2, y2 = rec
    return all(abs(val) <= COORD_LIMIT for val in (x1, y1, x2, y2))


def assign_components(
    records: Sequence[GeometryRecord],
    intervals: Sequence[Tuple[int, int, str]],
) -> Dict[str, List[GeometryRecord]]:
    starts = [start for start, _, _ in intervals]
    grouped: Dict[str, List[GeometryRecord]] = defaultdict(list)
    for rec in records:
        offset = rec[0]
        idx = bisect.bisect_right(starts, offset) - 1
        if idx < 0:
            continue
        start, end, label = intervals[idx]
        if offset >= end:
            continue
        grouped[label].append(rec)
    return grouped


def summarize_label(records: Sequence[GeometryRecord]) -> dict:
    if not records:
        return {"line_count": 0, "arc_count": 0}
    min_x = min_y = float("inf")
    max_x = max_y = float("-inf")
    layers = set()
    for _, layer, _, x1, y1, x2, y2 in records:
        layers.add(layer)
        min_x = min(min_x, x1, x2)
        min_y = min(min_y, y1, y2)
        max_x = max(max_x, x1, x2)
        max_y = max(max_y, y1, y2)
    line_count = sum(1 for rec in records if rec[2] == LINE_TYPE)
    arc_count = sum(1 for rec in records if rec[2] == ARC_TYPE)
    return {
        "line_count": line_count,
        "arc_count": arc_count,
        "layers": sorted(layers),
        "bbox": [min_x, min_y, max_x, max_y],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize MAIN font components by label.")
    parser.add_argument("input", type=Path, help="Path to Mcalf092.fnt.decompressed")
    parser.add_argument("--pad", type=int, default=32, help="Padding bytes included before each label chunk")
    parser.add_argument(
        "--min-lines",
        type=int,
        default=1,
        help="Only include labels with at least this many lines in the console output",
    )
    parser.add_argument("--json", type=Path, help="Optional JSON dump path for all labels")
    parser.add_argument(
        "--coord-format",
        choices=("double", "float"),
        default="float",
        help="How to decode the geometry payload (fonts typically use float)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blob = args.input.read_bytes()
    components = extract_components(blob, pad=args.pad)
    intervals = [(start, end, label) for label, start, end, _ in components]
    records = [
        rec
        for rec in collect_candidate_records(blob, coord_format=args.coord_format)
        if rec[2] in VALID_TYPES and _looks_valid(rec)
    ]
    grouped = assign_components(records, intervals)

    rows: List[dict] = []
    for label, start, end, _ in components:
        summary = summarize_label(grouped.get(label, []))
        rows.append(
            {
                "label": label,
                "offset_start": start,
                "offset_end": end,
                **summary,
            }
        )

    rows_sorted = sorted(rows, key=lambda r: r["label"])

    # Console preview
    print(f"Total labels detected: {len(rows_sorted)}")
    preview = [row for row in rows_sorted if row["line_count"] >= args.min_lines]
    for row in preview:
        bbox = row.get("bbox")
        bbox_str = " x ".join(f"{v:.4f}" for v in bbox) if bbox else "n/a"
        print(
            f"{row['label']:>10}: {row['line_count']:3d} lines / {row.get('arc_count', 0):3d} arcs, "
            f"layers={row.get('layers', [])}, bbox={bbox_str}"
        )

    if args.json:
        args.json.write_text(json.dumps(rows_sorted, indent=2), encoding="utf-8")
        print(f"\nJSON summary written to {args.json}")

    unmatched = len(records) - sum(len(vals) for vals in grouped.values())
    if unmatched:
        print(f"\n[warn] {unmatched} line records did not map to any component chunk")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
