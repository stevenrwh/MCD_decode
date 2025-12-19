#!/usr/bin/env python3
"""
Detect whether an .mcd looks like old-style or new-style based on markers and deflate streams.
Read-only; prints a short report.
"""

from __future__ import annotations

from pathlib import Path
from mcd_to_dxf import collect_deflate_streams
from component_parser import COMPONENT_MARKER


def detect(path: Path) -> None:
    blob = path.read_bytes()
    markers = blob.find(COMPONENT_MARKER)
    streams = collect_deflate_streams(blob, min_payload=32)
    print(f"{path.name}: size={len(blob)} bytes")
    print(f"  deflate streams: {[(off, len(data)) for off, data in streams]}")
    print(f"  has CComponentDefinition marker: {'yes' if markers != -1 else 'no'}")
    if markers == -1 and len(streams) == 1:
        print("  heuristic: old-style (single deflate, no markers)")
    elif markers != -1:
        print("  heuristic: new-style (marker present)")
    else:
        print("  heuristic: ambiguous")


def main() -> None:
    targets = [
        Path("old_style_mcd_files/TW84.mcd"),
        Path("new_style_mcd_files/TW84.mcd"),
    ]
    for t in targets:
        if t.exists():
            detect(t)
        else:
            print(f"{t} missing")


if __name__ == "__main__":
    main()
