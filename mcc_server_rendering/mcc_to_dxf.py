#!/usr/bin/env python3
"""
Server-friendly wrapper that converts a MONU-CAD .mcc/.mcd file into a DXF.

Usage:
    python mcc_to_dxf.py INPUT_FILE OUTPUT_DXF

Exit codes:
    0 -> success
    1 -> failure
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import mcd_to_dxf


def convert_to_dxf(source: Path, destination: Path) -> None:
    blob = source.read_bytes()
    _, payload = mcd_to_dxf.brute_force_deflate(blob)
    lines, arcs, circles, inserts = mcd_to_dxf.parse_entities(payload)
    if not (lines or arcs or circles or inserts):
        raise RuntimeError("No supported entities were discovered inside the payload.")
    mcd_to_dxf.write_dxf(lines, arcs, circles, destination, inserts=inserts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a .mcc/.mcd file into DXF format.")
    parser.add_argument("input", type=Path, help="Source .mcc/.mcd/.decompressed payload")
    parser.add_argument("output", type=Path, help="Destination DXF path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        convert_to_dxf(args.input, args.output)
        return 0
    except Exception as exc:  # pragma: no cover - server wrapper
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
