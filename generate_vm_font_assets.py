#!/usr/bin/env python3
"""
Generate reusable glyph data for the Vm / VERMARCO font by decoding the
component slices under `FONTS/components_Mcalf020/`.

Each component record stores signed 16-bit coordinate pairs (scaled by
1/32768). We interpret those pairs as raw line segments, optionally apply the
affine transform recovered by `analyze_vm_tail_transforms.py`, and emit both:

1. `FONTS/VERMARCO_glyphs.json` – consumed by FontManager inside mcd_to_dxf.py.
2. `FONTS/VERMARCO_glyphs.dxf` – preview grid for quick inspection.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from font_components import POINT_SCALE, iter_component_files
from mcd_to_dxf import (
    Glyph,
    LineEntity,
    _build_vm_mapping,
    _parse_unexploded_blocks,
    write_dxf,
)


def _decode_pairs(values: Sequence[int]) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    usable = len(values) // 4 * 4
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for idx in range(0, usable, 4):
        chunk = values[idx : idx + 4]
        if len(chunk) < 4:
            break
        if all(val == 0 for val in chunk):
            continue
        x1, y1, x2, y2 = (val * POINT_SCALE for val in chunk)
        segments.append(((x1, y1), (x2, y2)))
    return segments


def _segments_from_definition(definition) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for record in definition.records:
        values = record.values
        if len(values) <= 8:
            continue
        body = values[4:-4]
        if len(body) < 12:
            continue
        for idx in range(0, len(body), 12):
            chunk = body[idx : idx + 12]
            if len(chunk) < 12:
                break
            coords = chunk[4:]
            if len(coords) < 8:
                continue
            segments.extend(_decode_pairs(coords))
    return segments


def _load_vm_segments(components_dir: Path) -> Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    mapping: Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]] = {}
    for definition in iter_component_files(components_dir):
        segs = _segments_from_definition(definition)
        if segs:
            mapping[definition.label] = segs
    return mapping


def _apply_transform(
    segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
    transform: Tuple[float, float, float, float] | None,
) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    if not transform:
        return [((float(sx), float(sy)), (float(ex), float(ey))) for (sx, sy), (ex, ey) in segments]
    scale_x, scale_y, trans_x, trans_y = transform
    transformed: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for start, end in segments:
        sx = start[0] * scale_x + trans_x
        sy = start[1] * scale_y + trans_y
        ex = end[0] * scale_x + trans_x
        ey = end[1] * scale_y + trans_y
        transformed.append(((sx, sy), (ex, ey)))
    return transformed


def _segments_to_bounds(
    segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for start, end in segments:
        xs.extend((start[0], end[0]))
        ys.extend((start[1], end[1]))
    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _segments_to_dicts(
    segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> List[dict]:
    return [
        {
            "layer": 0,
            "start": [float(start[0]), float(start[1])],
            "end": [float(end[0]), float(end[1])],
        }
        for start, end in segments
    ]


def _build_glyph_entries(
    glyph_segments: Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]],
    transform_map: dict[str, Tuple[float, float, float, float]] | None = None,
) -> Dict[str, dict]:
    entries: Dict[str, dict] = {}
    for label, segments in glyph_segments.items():
        transformed = _apply_transform(segments, transform_map.get(label) if transform_map else None)
        if not transformed:
            continue
        bbox = _segments_to_bounds(transformed)
        min_x, min_y, max_x, max_y = bbox
        advance = max(max_x - min_x, 0.0)
        entries[label] = {
            "label": label,
            "bbox": list(bbox),
            "baseline": min_y,
            "advance": advance if advance > 0 else max_x,
            "line_count": len(transformed),
            "lines": _segments_to_dicts(transformed),
        }
    return entries


def _glyphs_to_entries(glyphs: Dict[str, Glyph]) -> Dict[str, dict]:
    entries: Dict[str, dict] = {}
    for label, glyph in glyphs.items():
        lines = [
            {
                "layer": 0,
                "start": [float(start[0]), float(start[1])],
                "end": [float(end[0]), float(end[1])],
            }
            for start, end in glyph.segments
        ]
        min_x, min_y, max_x, max_y = glyph.bounds
        advance = max(glyph.advance, max_x - min_x, 0.0)
        entries[label] = {
            "label": label,
            "bbox": [min_x, min_y, max_x, max_y],
            "baseline": min_y,
            "advance": advance,
            "line_count": len(lines),
            "lines": lines,
        }
    return entries


def _build_json_payload(glyph_entries: Dict[str, dict]) -> dict:
    mapping = _build_vm_mapping()
    glyphs: List[dict] = []
    for char, label in mapping:
        entry = glyph_entries.get(label)
        if not entry:
            continue
        glyphs.append({"char": char, **entry})
    return {
        "font": "VERMARCO",
        "source": "components_Mcalf020",
        "glyphs": glyphs,
    }


PREVIEW_ROWS = [
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "0123456789",
    "&-.,:;'\"",
]


def _build_preview_entities(payload: dict) -> List[LineEntity]:
    lookup = {entry["char"]: entry for entry in payload["glyphs"]}
    widths: List[float] = []
    heights: List[float] = []
    for entry in lookup.values():
        bbox = entry.get("bbox")
        if bbox:
            widths.append(bbox[2] - bbox[0])
            heights.append(bbox[3] - bbox[1])
    cell_w = (max(widths) if widths else 1.0) + 0.1
    cell_h = (max(heights) if heights else 1.0) + 0.1
    entities: List[LineEntity] = []
    for row_idx, row in enumerate(PREVIEW_ROWS):
        for col_idx, char in enumerate(row):
            entry = lookup.get(char)
            if not entry:
                continue
            bbox = entry.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            min_x, min_y = bbox[0], bbox[1]
            tx = col_idx * cell_w - min_x
            ty = -row_idx * cell_h - min_y
            for line in entry.get("lines", []):
                sx, sy = line["start"]
                ex, ey = line["end"]
                entities.append(
                    LineEntity(
                        layer=0,
                        start=(sx + tx, sy + ty),
                        end=(ex + tx, ey + ty),
                    )
                )
    return entities


def _load_transform_stats(path: Path | None) -> dict[str, Tuple[float, float, float, float]]:
    if not path or not path.exists():
        return {}
    mapping: dict[str, Tuple[float, float, float, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                scale_x = float(row["scale_x"])
                scale_y = float(row["scale_y"])
                translate_x = float(row["translate_x"])
                translate_y = float(row["translate_y"])
            except (KeyError, ValueError):
                continue
            mapping[row["label"]] = (scale_x, scale_y, translate_x, translate_y)
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Vm/VERMARCO font glyph JSON + DXF preview.")
    parser.add_argument("--json", type=Path, required=True, help="Destination JSON path")
    parser.add_argument("--dxf", type=Path, required=True, help="Destination DXF preview path")
    parser.add_argument(
        "--components",
        type=Path,
        help="Directory containing components_Mcalf020/*.bin slices (optional if reference DXF is provided)",
    )
    parser.add_argument(
        "--reference-dxf",
        type=Path,
        help="Path to Mcalf020_exported_from_monucad.dxf (preferred source when available)",
    )
    parser.add_argument(
        "--transform-stats",
        type=Path,
        help="Optional CSV from analyze_vm_tail_transforms.py with per-label transforms",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    glyph_entries: Dict[str, dict] = {}

    if args.reference_dxf and args.reference_dxf.exists():
        reference_glyphs = _parse_unexploded_blocks(args.reference_dxf)
        glyph_entries = _glyphs_to_entries(reference_glyphs)

    if not glyph_entries:
        if not args.components:
            raise SystemExit("No reference DXF found and --components was not provided.")
        glyph_segments = _load_vm_segments(args.components)
        transform_map = _load_transform_stats(args.transform_stats)
        glyph_entries = _build_glyph_entries(glyph_segments, transform_map=transform_map)

    payload = _build_json_payload(glyph_entries)
    if not payload["glyphs"]:
        raise SystemExit("No glyph geometry was reconstructed; aborting.")
    args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[+] JSON glyph data written to {args.json}")
    preview_entities = _build_preview_entities(payload)
    if preview_entities:
        write_dxf(preview_entities, [], [], args.dxf)
        print(f"[+] DXF preview written to {args.dxf}")
    else:
        print("[warn] Preview DXF skipped (no glyph segments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
