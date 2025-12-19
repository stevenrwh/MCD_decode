#!/usr/bin/env python3
"""
Generate reusable artifacts for the MAIN font (Mcalf092):

1. Extract each glyph's raw line segments into a JSON file that maps
   character -> label (M92A, M92BL, etc.) along with geometry.
2. Arrange all glyphs into a preview grid and emit a DXF so we can visually
   confirm the decoding without launching Monu-CAD.

Example:
    python generate_main_font_assets.py FONTS/Mcalf092.fnt.decompressed \
        --json FONTS/MAIN_glyphs.json \
        --dxf FONTS/MAIN_glyphs.dxf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from glyph_tlv_parser import GlyphComponent, parse_glyph_components_from_blob
from mcd_to_dxf import LineEntity, write_dxf

# Human-friendly mapping between characters and MAIN glyph labels.
GLYPH_MAP = [
    *((chr(ord("A") + i), f"M92{chr(ord('A') + i)}") for i in range(26)),
    *((str(i), f"M92{i}") for i in range(10)),
    *((chr(ord("a") + i), f"M92{chr(ord('A') + i)}L") for i in range(26)),
    ("-", "M92-"),
    (".", "M92PERID"),
    (",", "M92COMMA"),
    (":", "M92COLN"),
    (";", "M92SEMI"),
    ("\"", "M92QUOTE"),
    ("'", "M92APOST"),
]

# Layout order for the DXF preview (rows of characters).
LAYOUT_ROWS = [
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "0123456789",
    "abcdefghijklmnopqrstuvwxyz",
    "-.,;:\"'",
]

def _line_dicts(component: GlyphComponent) -> List[dict]:
    return [
        {
            "layer": 0,
            "start": [segment[0][0], segment[0][1]],
            "end": [segment[1][0], segment[1][1]],
        }
        for segment in component.segments
    ]


def build_json_payload(
    component_lookup: Dict[str, GlyphComponent],
    label_lookup: Dict[str, str],
) -> dict:
    glyphs = []
    for char, label in label_lookup.items():
        component = component_lookup.get(label)
        if not component:
            continue
        lines = _line_dicts(component)
        glyphs.append(
            {
                "char": char,
                "label": label,
                "baseline": component.baseline,
                "advance": component.advance,
                "bbox": list(component.bbox),
                "line_count": len(lines),
                "lines": lines,
                "header_ints": list(component.header_ints),
            }
        )
    return {
        "font": "MAIN",
        "source": "Mcalf092.fnt",
        "glyphs": glyphs,
    }


def build_preview_lines(glyph_data: dict, margin: float = 0.2) -> List[LineEntity]:
    lookup = {entry["char"]: entry for entry in glyph_data["glyphs"]}
    widths = []
    heights = []
    for entry in lookup.values():
        bbox = entry.get("bbox")
        if not bbox:
            continue
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    cell_w = (max(widths) if widths else 1.0) + margin
    cell_h = (max(heights) if heights else 1.0) + margin

    entities: List[LineEntity] = []
    for row_idx, row in enumerate(LAYOUT_ROWS):
        for col_idx, char in enumerate(row):
            entry = lookup.get(char)
            if not entry or not entry.get("lines"):
                continue
            bbox = entry.get("bbox")
            min_x = bbox[0] if bbox else 0.0
            min_y = bbox[1] if bbox else 0.0
            tx = col_idx * cell_w - min_x
            ty = -row_idx * cell_h - min_y
            for line in entry["lines"]:
                sx, sy = line["start"]
                ex, ey = line["end"]
                entities.append(
                    LineEntity(
                        layer=line["layer"],
                        start=(sx + tx, sy + ty),
                        end=(ex + tx, ey + ty),
                    )
                )
    return entities


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MAIN font glyph JSON + DXF preview.")
    parser.add_argument("input", type=Path, help="Path to Mcalf092.fnt.decompressed")
    parser.add_argument("--json", type=Path, required=True, help="Destination JSON path")
    parser.add_argument("--dxf", type=Path, required=True, help="Destination DXF preview path")
    parser.add_argument("--pad", type=int, default=32, help="Padding bytes to keep before each label (default 32)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blob = args.input.read_bytes()
    glyph_components = parse_glyph_components_from_blob(blob, pad=args.pad)
    component_lookup = {component.label: component for component in glyph_components}
    char_to_label = {char: label for char, label in GLYPH_MAP}

    glyph_json = build_json_payload(component_lookup, char_to_label)
    args.json.write_text(json.dumps(glyph_json, indent=2), encoding="utf-8")
    print(f"[+] JSON glyph data written to {args.json}")

    preview_lines = build_preview_lines(glyph_json)
    if not preview_lines:
        raise SystemExit("No glyph geometry was captured; aborting DXF generation.")
    write_dxf(preview_lines, [], [], args.dxf)
    print(f"[+] DXF preview written to {args.dxf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
