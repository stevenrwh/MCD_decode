#!/usr/bin/env python3
"""
Probe a Monu-CAD font archive (.fnt/.dta/.decompressed):
  - deflate offsets/sizes (if input is wrapped)
  - glyph count and label list
  - bbox/advance stats
  - kerning pair count (when a kerning JSON is found alongside)

Usage:
    python diagnostics/fnt_probe.py FONTS/Mcalf020.dta
    python diagnostics/fnt_probe.py FONTS/Mcalf020_exported_from_monucad.dxf --from-dxf
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from monucad.deflate_io import collect_deflate_streams, brute_force_deflate
from monucad.fonts import FontDefinition, Glyph, load_glyphs_from_reference, parse_kerning_file
from font_components import iter_component_files
from extract_font_components import extract_components


def _load_payload(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix == ".decompressed":
        return data
    try:
        _, payload = brute_force_deflate(data)
        return payload
    except Exception:
        return data


def _glyph_stats(glyphs: dict[str, Glyph]) -> dict:
    bboxes = []
    advances = []
    for glyph in glyphs.values():
        min_x, min_y, max_x, max_y = glyph.bounds
        bboxes.append((min_x, min_y, max_x, max_y))
        advances.append(glyph.advance)
    def _summary(values):
        return {
            "min": min(values) if values else 0.0,
            "max": max(values) if values else 0.0,
            "mean": sum(values) / len(values) if values else 0.0,
        }
    return {
        "count": len(glyphs),
        "adv": _summary(advances),
        "bbox_x_span": _summary([max_x - min_x for min_x, _, max_x, _ in bboxes]) if bboxes else {},
        "bbox_y_span": _summary([max_y - min_y for _, min_y, _, max_y in bboxes]) if bboxes else {},
        "labels": sorted(glyphs.keys())[:50],
    }


def _load_from_components(path: Path) -> dict[str, Glyph]:
    glyphs: dict[str, Glyph] = {}
    for definition in iter_component_files(path):
        segments = []
        bounds = [float("inf"), float("-inf"), float("inf"), float("-inf")]
        scale = 1.0 / 32768.0
        for record in definition.records:
            vals = record.values
            if len(vals) < 12:
                continue
            coords = vals[4:12]
            for idx in range(0, len(coords), 4):
                x1, y1, x2, y2 = coords[idx : idx + 4]
                if x1 == y1 == x2 == y2 == 0:
                    continue
                p1 = (x1 * scale, y1 * scale)
                p2 = (x2 * scale, y2 * scale)
                segments.append((p1, p2))
                bounds[0] = min(bounds[0], p1[0], p2[0])
                bounds[1] = max(bounds[1], p1[0], p2[0])
                bounds[2] = min(bounds[2], p1[1], p2[1])
                bounds[3] = max(bounds[3], p1[1], p2[1])
        if not segments:
            continue
        min_x, max_x, min_y, max_y = bounds[0], bounds[1], bounds[2], bounds[3]
        advance = max(max_x - min_x, 0.0)
        glyphs[definition.label] = Glyph(
            label=definition.label,
            segments=segments,
            bounds=(min_x, min_y, max_x, max_y),
            advance=advance,
            baseline=min_y,
        )
    return glyphs


def probe(path: Path, *, from_dxf: bool = False) -> dict:
    summary = {"source": str(path)}
    if from_dxf:
        glyphs = load_glyphs_from_reference(path.stem)
        summary["source_type"] = "dxf"
        summary["glyphs"] = _glyph_stats(glyphs)
        summary["deflate_streams"] = []
    else:
        data = path.read_bytes()
        streams = collect_deflate_streams(data, min_payload=64)
        summary["deflate_streams"] = [{"offset": off, "size": len(payload)} for off, payload in streams]
        payload = streams[0][1] if streams else data
        glyphs = {}
        try:
            components = list(extract_components(payload))
            if components:
                for label, _, _, chunk in components:
                    # reuse glyph_tlv_parser output
                    glyphs.setdefault(label, Glyph(label=label, segments=[], bounds=(0, 0, 0, 0), advance=0, baseline=0))
        except Exception:
            glyphs = {}
        if not glyphs:
            glyphs = _load_from_components(path)
        summary["source_type"] = "payload"
        summary["glyphs"] = _glyph_stats(glyphs)

    # Kerning sidecar (.json) if present
    kerning = {}
    json_candidate = path.with_suffix(".json")
    if json_candidate.exists():
        kerning = parse_kerning_file(json_candidate)
    summary["kerning_pairs"] = len(kerning)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe Monu-CAD font archives (.fnt/.dta/.decompressed).")
    parser.add_argument("input", type=Path, help="Source .fnt/.dta/.decompressed (or DXF with --from-dxf)")
    parser.add_argument("--from-dxf", action="store_true", help="Treat input as an exported DXF font (BLOCKS)")
    parser.add_argument("--json", type=Path, help="Optional path to write JSON summary")
    args = parser.parse_args(argv)

    summary = probe(args.input, from_dxf=args.from_dxf)
    print(f"{args.input}:")
    print(f"  source_type: {summary['source_type']}")
    streams = summary.get("deflate_streams", [])
    if streams:
        stream_str = ", ".join(f"0x{s['offset']:X}/{s['size']}B" for s in streams)
        print(f"  deflate streams: {stream_str}")
    print(f"  glyph count: {summary['glyphs'].get('count', 0)}")
    if summary['glyphs'].get("labels"):
        print(f"  labels (up to 50): {summary['glyphs']['labels']}")
    print(f"  kerning pairs: {summary['kerning_pairs']}")

    if args.json:
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  JSON summary written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
