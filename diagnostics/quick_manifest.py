#!/usr/bin/env python3
"""
Generate a quick manifest for a .mcd/.mcc/.decompressed payload:
  - deflate offsets/sizes
  - presence of CComponentDefinition marker
  - component definition count and placement trailer count
  - catalog entry count (best-effort)
  - geometry counts (lines/arcs/circles/inserts) via parse_entities (best-effort)

Usage:
    python diagnostics/quick_manifest.py path/to/file.mcd [--json manifest.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component_parser import COMPONENT_MARKER, iter_component_definitions
from placement_parser import iter_placement_trailers
from monucad.deflate_io import collect_deflate_streams
from mcd_to_dxf import parse_entities


def summarize(path: Path) -> Dict[str, Any]:
    blob = path.read_bytes()
    streams = collect_deflate_streams(blob, min_payload=32)
    streams_sorted = sorted(streams, key=lambda item: len(item[1]), reverse=True)
    best_offset, best_payload = streams_sorted[0] if streams_sorted else (None, b"")
    marker_offset = blob.find(COMPONENT_MARKER)
    payload = best_payload

    defs = list(iter_component_definitions(payload)) if payload else []
    placements = list(iter_placement_trailers(payload)) if payload else []

    try:
        from diagnostics.dump_catalog_records import iter_catalog_records

        catalog_entries = list(iter_catalog_records(payload)) if payload else []
    except Exception:
        catalog_entries = []

    geom_counts = {}
    try:
        lines, arcs, circles, inserts = parse_entities(payload)
        geom_counts = {
            "lines": len(lines),
            "arcs": len(arcs),
            "circles": len(circles),
            "inserts": len(inserts),
        }
    except Exception:
        geom_counts = {"error": "parse_entities failed"}

    return {
        "source": str(path),
        "size_bytes": len(blob),
        "deflate_streams": [{"offset": off, "size": len(data)} for off, data in streams_sorted],
        "primary_offset": best_offset,
        "primary_payload_size": len(payload),
        "has_component_marker": marker_offset != -1,
        "component_defs": len(defs),
        "placement_trailers": len(placements),
        "catalog_entries": len(catalog_entries),
        "geometry": geom_counts,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quick manifest for .mcd/.mcc payloads.")
    parser.add_argument("input", type=Path, help="Path to .mcd/.mcc/.decompressed file")
    parser.add_argument("--json", type=Path, help="Optional path to write JSON summary")
    args = parser.parse_args(argv)

    summary = summarize(args.input)
    print(f"{args.input}:")
    print(f"  size: {summary['size_bytes']} bytes")
    streams = summary["deflate_streams"]
    if streams:
        stream_desc = ", ".join(f"0x{s['offset']:X}/{s['size']}B" for s in streams)
        print(f"  deflate streams: {stream_desc}")
    else:
        print("  deflate streams: none found")
    print(f"  has CComponentDefinition marker: {'yes' if summary['has_component_marker'] else 'no'}")
    print(f"  component defs: {summary['component_defs']}")
    print(f"  placement trailers: {summary['placement_trailers']}")
    print(f"  catalog entries (best-effort): {summary['catalog_entries']}")
    geom = summary["geometry"]
    if "error" in geom:
        print(f"  geometry: {geom['error']}")
    else:
        print(
            f"  geometry: lines={geom.get('lines', 0)}, arcs={geom.get('arcs', 0)}, "
            f"circles={geom.get('circles', 0)}, inserts={geom.get('inserts', 0)}"
        )

    if args.json:
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  JSON summary written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
