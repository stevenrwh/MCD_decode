#!/usr/bin/env python3
"""
Instrument Vm chunk metadata against the decoded tail stroke records.

This helper links every chunk stored in `vm_chunk_dump.json` with the actual
line/arc records we pulled out of `Mcalf020.fnt.decompressed`.  The goal is not
to solve the mapping outright but to capture enough per-chunk statistics
(segment counts, bounding boxes, closest record offsets, distance scores) so
we can spot the real metadata -> tail patterns.

Typical usage:

    python match_vm_chunk_tail_candidates.py \
        --chunk-dump vm_chunk_dump.json \
        --fnt FONTS/Mcalf020.fnt.decompressed \
        --components FONTS/components_Mcalf020 \
        --transform-stats FONTS/Vm_tail_transform_stats.csv \
        --output vm_chunk_match_candidates.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Sequence, Tuple

from extract_font_components import extract_components
from font_components import iter_component_files
from mcd_to_dxf import collect_candidate_records
from summarize_font_components import ARC_TYPE, LINE_TYPE, VALID_TYPES, _looks_valid, assign_components

ChunkEntry = dict
Point = Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Correlate Vm chunk metadata with tail stroke candidates.")
    parser.add_argument("--chunk-dump", type=Path, required=True, help="Path to vm_chunk_dump.json")
    parser.add_argument("--fnt", type=Path, required=True, help="Path to Mcalf020.fnt.decompressed")
    parser.add_argument(
        "--components",
        type=Path,
        required=True,
        help="Directory containing components_Mcalf020/*.bin slices",
    )
    parser.add_argument(
        "--transform-stats",
        type=Path,
        help="Optional CSV from analyze_vm_tail_transforms.py (used when transformed_points are missing)",
    )
    parser.add_argument("--output", type=Path, required=True, help="Destination JSON path for the candidate dump")
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Max number of closest stroke candidates to keep per chunk segment (default: 5)",
    )
    parser.add_argument(
        "--coord-format",
        choices=("float", "double"),
        default="float",
        help="Coordinate encoding used by the font payload (fonts typically use float)",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=0,
        help="Padding bytes preserved around each component slice (must match extract_font_components usage)",
    )
    return parser.parse_args()


def _load_chunk_dump(path: Path) -> dict[str, List[ChunkEntry]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_transform_map(path: Path | None) -> dict[str, Tuple[float, float, float, float]]:
    if not path or not path.exists():
        return {}
    mapping: dict[str, Tuple[float, float, float, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                mapping[row["label"]] = (
                    float(row["scale_x"]),
                    float(row["scale_y"]),
                    float(row["translate_x"]),
                    float(row["translate_y"]),
                )
            except (KeyError, ValueError):
                continue
    return mapping


def _load_tail_map(components_dir: Path) -> dict[str, Tuple[int, int, int, int]]:
    mapping: dict[str, Tuple[int, int, int, int]] = {}
    for definition in iter_component_files(components_dir):
        if not definition.records:
            continue
        tail = tuple(int(val) for val in definition.records[0].values[-4:])
        mapping[definition.label] = tail
    return mapping


def _load_label_records(
    fnt_path: Path,
    *,
    pad: int = 0,
    coord_format: str = "float",
) -> dict[str, List[dict]]:
    blob = fnt_path.read_bytes()
    components = extract_components(blob, pad=pad)
    intervals = [(start, end, label) for label, start, end, _ in components]
    records = [
        rec
        for rec in collect_candidate_records(blob, coord_format=coord_format)
        if rec[2] in VALID_TYPES and _looks_valid(rec)
    ]
    grouped = assign_components(records, intervals)
    label_records: dict[str, List[dict]] = {}
    for label, recs in grouped.items():
        converted: List[dict] = []
        for offset, layer, etype, x1, y1, x2, y2 in recs:
            converted.append(
                {
                    "offset": int(offset),
                    "layer": int(layer),
                    "type": "line" if etype == LINE_TYPE else "arc",
                    "start": (float(x1), float(y1)),
                    "end": (float(x2), float(y2)),
                }
            )
        if converted:
            label_records[label] = converted
    return label_records


def _bbox(points: Sequence[Point]) -> list[float] | None:
    if not points:
        return None
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def _segment_distance(seg: Tuple[Point, Point], record: dict) -> float:
    """Return the squared distance between the chunk segment and a stroke record."""

    def _point_diff(a: Point, b: Point) -> float:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    start, end = seg
    rec_start = record["start"]
    rec_end = record["end"]
    forward = _point_diff(start, rec_start) + _point_diff(end, rec_end)
    reverse = _point_diff(start, rec_end) + _point_diff(end, rec_start)
    return min(forward, reverse)


def _segmentify(points: Sequence[Sequence[float]]) -> List[Tuple[Point, Point]]:
    segments: List[Tuple[Point, Point]] = []
    usable = len(points) // 2 * 2
    for idx in range(0, usable, 2):
        start = points[idx]
        end = points[idx + 1]
        segments.append(((float(start[0]), float(start[1])), (float(end[0]), float(end[1]))))
    return segments


def _apply_transform(points: Sequence[Sequence[float]], transform: Tuple[float, float, float, float]) -> List[Point]:
    sx, sy, tx, ty = transform
    return [(float(x) * sx + tx, float(y) * sy + ty) for x, y in points]


def _ensure_transformed_points(
    chunk: ChunkEntry,
    *,
    label: str,
    transform_map: dict[str, Tuple[float, float, float, float]],
) -> List[Point]:
    transformed = chunk.get("transformed_points")
    if transformed:
        return [(float(x), float(y)) for x, y in transformed]
    raw_points = chunk.get("raw_points") or []
    transform = transform_map.get(label)
    if transform:
        return _apply_transform(raw_points, transform)
    return [(float(x), float(y)) for x, y in raw_points]


def _decode_raw_points(chunk: ChunkEntry) -> List[Point]:
    return [(float(x), float(y)) for x, y in chunk.get("raw_points", [])]


def compute_matches(
    chunk_data: dict[str, List[ChunkEntry]],
    *,
    tail_map: dict[str, Tuple[int, int, int, int]],
    label_records: dict[str, List[dict]],
    transform_map: dict[str, Tuple[float, float, float, float]],
    top_n: int,
) -> dict[str, List[dict]]:
    payload: dict[str, List[dict]] = defaultdict(list)
    for label, chunks in chunk_data.items():
        records = label_records.get(label)
        if not records:
            continue
        tail = tail_map.get(label)
        for idx, chunk in enumerate(chunks):
            raw_points = _decode_raw_points(chunk)
            transformed_points = _ensure_transformed_points(chunk, label=label, transform_map=transform_map)
            segments = _segmentify(transformed_points if transformed_points else raw_points)
            best_matches: List[dict] = []
            for seg_idx, segment in enumerate(segments):
                scored = [
                    (float(_segment_distance(segment, record)), record)
                    for record in records
                ]
                scored.sort(key=lambda item: item[0])
                top_scores = scored[: max(1, top_n)]
                for score, record in top_scores:
                    best_matches.append(
                        {
                            "segment_index": seg_idx,
                            "offset": record["offset"],
                            "type": record["type"],
                            "layer": record["layer"],
                            "distance_sq": score,
                        }
                    )
            span = None
            if best_matches:
                offsets = [entry["offset"] for entry in best_matches]
                span = [min(offsets), max(offsets)]
            payload[label].append(
                {
                    "chunk_index": idx,
                    "meta": chunk.get("meta"),
                    "tail": list(tail) if tail else None,
                    "segment_count": len(segments),
                    "raw_point_count": len(raw_points),
                    "raw_bbox": _bbox(raw_points),
                    "transformed_bbox": _bbox(transformed_points),
                    "best_matches": best_matches,
                    "candidate_span": span,
                }
            )
    return payload


def main() -> int:
    args = parse_args()
    chunk_data = _load_chunk_dump(args.chunk_dump)
    transform_map = _load_transform_map(args.transform_stats)
    tail_map = _load_tail_map(args.components)
    label_records = _load_label_records(
        args.fnt,
        pad=args.pad,
        coord_format=args.coord_format,
    )
    matches = compute_matches(
        chunk_data,
        tail_map=tail_map,
        label_records=label_records,
        transform_map=transform_map,
        top_n=args.top_n,
    )
    output = {
        "chunk_dump": str(args.chunk_dump),
        "fnt_source": str(args.fnt),
        "components_dir": str(args.components),
        "transform_stats": str(args.transform_stats) if args.transform_stats else None,
        "top_n": args.top_n,
        "labels": matches,
    }
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[+] Candidate mapping written to {args.output} ({sum(len(v) for v in matches.values())} chunks)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
