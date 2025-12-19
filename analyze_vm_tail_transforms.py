#!/usr/bin/env python3
"""
Correlate Vm component metadata with the actual glyph geometry captured from
Monu-CAD's DXF export.

For every component that references a tail tuple, we compute:
  * the tail tuple (last four ints in the record)
  * the header slots (converted to floats)
  * the bounding box reported by the reference DXF
  * the bounding box of the global tail strokes
  * the scale/translation that would map the tail bbox onto the reference bbox

The output CSV (`FONTS/Vm_tail_transform_stats.csv`) helps us reverse the
affine parameters stored in the component headers.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from font_components import parse_component_file
from mcd_to_dxf import _parse_unexploded_blocks


def _load_tail_bboxes(path: Path) -> dict[tuple[int, int, int, int], tuple[float, float, float, float]]:
    data = json.loads(path.read_text())
    boxes: dict[tuple[int, int, int, int], tuple[float, float, float, float]] = {}
    for entry in data.get("tails", []):
        tail = tuple(entry.get("tail", []))
        xs: List[float] = []
        ys: List[float] = []
        for record in entry.get("records", []):
            xs.extend([record.get("x1", 0.0), record.get("x2", 0.0)])
            ys.extend([record.get("y1", 0.0), record.get("y2", 0.0)])
        if not xs or not ys:
            continue
        boxes[tail] = (min(xs), min(ys), max(xs), max(ys))
    return boxes


def _scale_translate(
    source_bbox: tuple[float, float, float, float] | None,
    target_bbox: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    if not source_bbox or not target_bbox:
        return 0.0, 0.0, 0.0, 0.0
    sx0, sy0, sx1, sy1 = source_bbox
    tx0, ty0, tx1, ty1 = target_bbox
    sw = sx1 - sx0
    sh = sy1 - sy0
    tw = tx1 - tx0
    th = ty1 - ty0
    if abs(sw) < 1e-6 or abs(sh) < 1e-6:
        return 0.0, 0.0, 0.0, 0.0
    scale_x = tw / sw
    scale_y = th / sh
    translate_x = tx0 - sx0 * scale_x
    translate_y = ty0 - sy0 * scale_y
    return scale_x, scale_y, translate_x, translate_y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Vm component tail transforms.")
    parser.add_argument("--components", type=Path, required=True, help="Directory containing components_Mcalf020/*.bin")
    parser.add_argument("--tail-records", type=Path, required=True, help="Path to VM_tail_records.json")
    parser.add_argument("--reference-dxf", type=Path, required=True, help="Path to Mcalf020_exported_from_monucad.dxf")
    parser.add_argument("--output", type=Path, default=Path("FONTS/Vm_tail_transform_stats.csv"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tail_boxes = _load_tail_bboxes(args.tail_records)
    reference_glyphs = _parse_unexploded_blocks(args.reference_dxf)

    rows: List[dict] = []
    for component_path in sorted(args.components.glob("*.bin")):
        definition = parse_component_file(component_path)
        if not definition.records:
            continue
        record = definition.records[0]
        tail = tuple(record.values[-4:])
        ref_glyph = reference_glyphs.get(definition.label)
        ref_bbox = ref_glyph.bounds if ref_glyph else None
        tail_bbox = tail_boxes.get(tail)
        scale_x, scale_y, trans_x, trans_y = _scale_translate(tail_bbox, ref_bbox)
        row = {
            "label": definition.label,
            "tail": tail,
                "tail_bbox_min_x": tail_bbox[0] if tail_bbox else "",
                "tail_bbox_min_y": tail_bbox[1] if tail_bbox else "",
                "tail_bbox_max_x": tail_bbox[2] if tail_bbox else "",
                "tail_bbox_max_y": tail_bbox[3] if tail_bbox else "",
                "ref_bbox_min_x": ref_bbox[0] if ref_bbox else "",
                "ref_bbox_min_y": ref_bbox[1] if ref_bbox else "",
                "ref_bbox_max_x": ref_bbox[2] if ref_bbox else "",
                "ref_bbox_max_y": ref_bbox[3] if ref_bbox else "",
                "scale_x": scale_x,
                "scale_y": scale_y,
                "translate_x": trans_x,
                "translate_y": trans_y,
        }
        for idx in range(len(definition.header)):
            row[f"header_raw_{idx}"] = definition.header[idx]
            row[f"header_f{idx}"] = definition.header[idx] / 32768.0
        rows.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()) if rows else [],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[+] Wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
