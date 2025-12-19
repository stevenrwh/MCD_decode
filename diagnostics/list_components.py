#!/usr/bin/env python3
"""
List every component definition discovered in an .mcd by scanning all deflate
streams and raw markers. Read-only; no converter changes.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Iterable

from component_parser import COMPONENT_MARKER, iter_component_definitions
from mcd_to_dxf import collect_deflate_streams


def iter_marker_definitions(blob: bytes):
    """Scan raw bytes for COMPONENT_MARKER and parse definitions."""
    offsets = []
    start = 0
    while True:
        pos = blob.find(COMPONENT_MARKER, start)
        if pos == -1:
            break
        offsets.append(pos)
        start = pos + 1
    for pos in offsets:
        tail = blob[pos:]
        try:
            yield from iter_component_definitions(tail)
        except Exception:
            continue


def list_components(path: Path) -> None:
    blob = path.read_bytes()
    streams = collect_deflate_streams(blob, min_payload=64)
    seen_ids = set()
    defs = []
    # From deflate streams
    for _, payload in streams:
        for d in iter_component_definitions(payload):
            if d.component_id in seen_ids:
                continue
            seen_ids.add(d.component_id)
            defs.append(d)
    # Raw marker scan
    for d in iter_marker_definitions(blob):
        if d.component_id in seen_ids:
            continue
        seen_ids.add(d.component_id)
        defs.append(d)
    print(f"{path.name}: found {len(defs)} component definitions")
    for d in defs:
        print(f"  cid={d.component_id} bbox={d.bbox} sub_blocks={len(d.sub_blocks)}")


def main() -> None:
    targets: Iterable[Path] = [
        Path("new_style_mcd_files/TW84.mcd"),
        Path("new_style_mcd_files/JF6050-THOMPSON.mcd"),
    ]
    for t in targets:
        if t.exists():
            list_components(t)
        else:
            print(f"{t} missing")


if __name__ == "__main__":
    main()
