#!/usr/bin/env python3
"""
Quick manifest focused on .mcc component packs (also works on .mcd/.decompressed):
  - deflate offsets/sizes
  - presence of CComponentDefinition marker
  - component definition list (id, bbox, sub-block counts)
  - placement trailers list (name, component_id, instance_id)
  - optional JSON dump

Usage:
    python diagnostics/quick_manifest_mcc.py path/to/file.mcc [--json manifest.json]
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


def summarize(path: Path) -> Dict[str, Any]:
    blob = path.read_bytes()
    streams = collect_deflate_streams(blob, min_payload=32)
    streams_sorted = sorted(streams, key=lambda item: len(item[1]), reverse=True)
    best_payload = streams_sorted[0][1] if streams_sorted else b""
    marker_offset = blob.find(COMPONENT_MARKER)

    defs = list(iter_component_definitions(best_payload)) if best_payload else []
    placements = list(iter_placement_trailers(best_payload)) if best_payload else []

    def_summaries = []
    for definition in defs:
        def_summaries.append(
            {
                "offset": definition.offset,
                "length": definition.length,
                "component_id": definition.component_id,
                "bbox": tuple(definition.bbox) if definition.bbox else None,
                "sub_blocks": [
                    {"tag": hex(blk.tag), "dtype": blk.dtype, "count": blk.count, "size": len(blk.payload)}
                    for blk in definition.sub_blocks
                ],
            }
        )

    placement_summaries = [
        {"name": p.name, "component_id": p.component_id, "instance_id": p.instance_id} for p in placements
    ]

    return {
        "source": str(path),
        "size_bytes": len(blob),
        "deflate_streams": [{"offset": off, "size": len(data)} for off, data in streams_sorted],
        "has_component_marker": marker_offset != -1,
        "component_defs": def_summaries,
        "placement_trailers": placement_summaries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Quick manifest for .mcc component packs (and .mcd blobs).")
    parser.add_argument("input", type=Path, help="Path to .mcc/.mcd/.decompressed file")
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
    defs = summary["component_defs"]
    print(f"  component defs: {len(defs)}")
    for d in defs[:10]:
        print(
            f"    id=0x{d['component_id']:08X} bbox={d['bbox']} "
            f"sub_blocks={len(d['sub_blocks'])} off=0x{d['offset']:X} len={d['length']}"
        )
    placements = summary["placement_trailers"]
    print(f"  placement trailers: {len(placements)}")
    for p in placements[:10]:
        print(f"    name={p['name']} component_id=0x{p['component_id']:08X} inst_id={p['instance_id']}")

    if args.json:
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"  JSON summary written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
