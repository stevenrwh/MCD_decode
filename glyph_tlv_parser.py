#!/usr/bin/env python3
"""
Structured parser for the TLV-encoded `CComponentDefinition` blocks that live
inside Monu-CAD font archives (`*.fnt.decompressed`).

The goal is to expose per-glyph metadata (baseline, advance width, bounding
box) alongside the grouped line segments so downstream tools no longer have to
flatten the raw record stream or guess where one glyph stops and the next
begins.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from extract_font_components import extract_components
from font_components import ComponentDefinition, ComponentRecord, parse_component_bytes

# Etype values used by component records.
LINE_RECORD_TYPE = 2
ARC_RECORD_TYPE = 3


@dataclass(frozen=True)
class GlyphComponent:
    """Structured view of a single glyph/component definition."""

    label: str
    bbox: Tuple[float, float, float, float]
    baseline: float
    advance: float
    segments: Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]
    header_ints: Tuple[int, ...]
    record_count: int
    source_range: Tuple[int, int]

    def to_dict(self) -> dict:
        """Serialize the glyph so it can be dumped to JSON."""

        lines = [
            {
                "start": [segment[0][0], segment[0][1]],
                "end": [segment[1][0], segment[1][1]],
            }
            for segment in self.segments
        ]
        return {
            "label": self.label,
            "baseline": self.baseline,
            "advance": self.advance,
            "bbox": list(self.bbox),
            "line_count": len(self.segments),
            "header_ints": list(self.header_ints),
            "record_count": self.record_count,
            "source_range": list(self.source_range),
            "lines": lines,
        }


def _segments_from_record(record: ComponentRecord) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Convert a component record into zero or more line segments or arcs.

    Arc records carry three points per logical arc: start, guide/center, end.
    We approximate arcs into small straight segments here; callers can promote
    to ARC entities later.
    """

    metadata = record.metadata
    etype = metadata[2] if len(metadata) >= 3 else None
    points = list(record.normalized_points())
    if etype == ARC_RECORD_TYPE and len(points) >= 3:
        # Approximate the arc with a polyline through the three points.
        # Start -> guide -> end, emit as two segments.
        start, guide, end = points[0], points[1], points[2]
        return [(start, guide), (guide, end)]
    if etype not in (LINE_RECORD_TYPE, ARC_RECORD_TYPE, 0, None) and (etype is not None and etype < 256):
        return []
    if len(points) < 2:
        return []
    if len(points) % 2 != 0:
        points = points[:-1]
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for idx in range(0, len(points), 2):
        start = points[idx]
        end = points[idx + 1]
        if start == end:
            continue
        segments.append((start, end))
    return segments


def _segments_from_definition(definition: ComponentDefinition) -> Tuple[Tuple[Tuple[float, float], Tuple[float, float]], ...]:
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for record in definition.records:
        segments.extend(_segments_from_record(record))
    return tuple(segments)


def parse_glyph_components_from_blob(blob: bytes, *, pad: int = 32) -> List[GlyphComponent]:
    """Parse every glyph definition contained in a font payload."""

    components: List[GlyphComponent] = []
    for label, start, end, chunk in extract_components(blob, pad=pad):
        try:
            definition = parse_component_bytes(label, chunk)
        except Exception:
            # Skip malformed chunks but keep marching so a single bad glyph
            # does not torpedo the entire run.
            continue
        segments = _segments_from_definition(definition)
        bbox = definition.bounding_box()
        if not bbox:
            bbox = (0.0, 0.0, 0.0, 0.0)
        baseline = bbox[1]
        advance = bbox[2] - bbox[0]
        components.append(
            GlyphComponent(
                label=label,
                bbox=bbox,
                baseline=baseline,
                advance=advance,
                segments=segments,
                header_ints=definition.header,
                record_count=len(definition.records),
                source_range=(start, end),
            )
        )
    return components


def parse_glyph_components(path: Path, *, pad: int = 32) -> List[GlyphComponent]:
    return parse_glyph_components_from_blob(path.read_bytes(), pad=pad)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decode TLV font components into per-glyph metadata and grouped geometry."
    )
    parser.add_argument("input", type=Path, help="Path to Mcalf*.fnt.decompressed")
    parser.add_argument(
        "--pad",
        type=int,
        default=32,
        help="Bytes of context to keep before each glyph label (default: 32)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="Optional destination for the full glyph JSON payload",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of glyphs to preview on stdout (default: 10)",
    )
    return parser.parse_args(argv)


def _print_preview(glyphs: Sequence[GlyphComponent], limit: int) -> None:
    preview = glyphs[: max(0, limit)]
    if not preview:
        print("No glyph components were parsed.")
        return
    print(f"Parsed {len(glyphs)} glyph component(s). Preview:")
    for glyph in preview:
        bbox = glyph.bbox
        print(
            f"  {glyph.label:<10} segments={len(glyph.segments):3d} "
            f"baseline={glyph.baseline:+.4f} advance={glyph.advance:.4f} "
            f"bbox=({bbox[0]:+.4f}, {bbox[1]:+.4f})-({bbox[2]:+.4f}, {bbox[3]:+.4f})"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    glyphs = parse_glyph_components(args.input, pad=args.pad)
    glyphs_sorted = sorted(glyphs, key=lambda g: g.label)
    _print_preview(glyphs_sorted, args.limit)
    if args.json:
        payload = {
            "source": str(args.input),
            "glyphs": [glyph.to_dict() for glyph in glyphs_sorted],
        }
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nFull glyph payload written to {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
