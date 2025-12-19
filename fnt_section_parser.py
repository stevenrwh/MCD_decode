#!/usr/bin/env python3
"""
Summarize Monu-CAD .fnt/.dta archives (after stripping the fake gzip wrapper).

Outputs glyph counts, baselines/advances/bboxes, and optionally emits a JSON
blob so we can diff font payloads without reopening MCPro9.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from glyph_tlv_parser import GlyphComponent, parse_glyph_components_from_blob
from mcd_to_dxf import brute_force_deflate


def _load_font_payload(path: Path) -> tuple[int, bytes]:
    data = path.read_bytes()
    if path.suffix.lower().endswith(".decompressed"):
        return 0, data
    return brute_force_deflate(data)


def _print_preview(glyphs: Sequence[GlyphComponent], limit: int) -> None:
    preview = glyphs[:limit]
    print(f"[+] Parsed {len(glyphs)} glyph component(s)")
    if not preview:
        return
    print("[preview]")
    for glyph in preview:
        bbox = glyph.bbox
        print(
            f"  {glyph.label:<12} segments={len(glyph.segments):3d} "
            f"baseline={glyph.baseline:+.4f} advance={glyph.advance:.4f} "
            f"bbox=({bbox[0]:+.4f},{bbox[1]:+.4f})-({bbox[2]:+.4f},{bbox[3]:+.4f})"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Monu-CAD .fnt/.dta payloads.")
    parser.add_argument("input", type=Path, help="Path to *.fnt / *.dta (or *.decompressed)")
    parser.add_argument("--json", type=Path, help="Optional destination for glyph metadata JSON")
    parser.add_argument("--limit", type=int, default=10, help="Preview line cap (default: 10)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    offset, payload = _load_font_payload(args.input)
    print(f"[+] Payload size {len(payload)} bytes (deflate offset 0x{offset:X})")
    glyphs = parse_glyph_components_from_blob(payload)
    glyphs_sorted = sorted(glyphs, key=lambda g: g.label)
    _print_preview(glyphs_sorted, args.limit)
    if args.json:
        bundle = {
            "source": str(args.input),
            "deflate_offset": offset,
            "glyphs": [glyph.to_dict() for glyph in glyphs_sorted],
        }
        args.json.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        print(f"[+] JSON summary written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
