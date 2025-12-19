#!/usr/bin/env python3
"""
Wrapper script for the server pipeline: render a MONU-CAD .mcc/.mcd file
to a PNG preview using the existing geometry parser.

Usage:
    python mcc_to_png_preview.py INPUT_FILE OUTPUT_PNG [--size 400]

Exit codes:
    0 -> success
    1 -> bad arguments / runtime error
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

import render_component_png


def render_preview(input_path: Path, output_path: Path, size: int) -> None:
    lines, arcs, circles = render_component_png.load_geometry(input_path)
    render_component_png.render_png(
        lines,
        arcs,
        circles,
        destination=output_path,
        size_px=size,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a .mcc/.mcd file into a PNG preview.")
    parser.add_argument("input", type=Path, help="Source .mcc/.mcd/.decompressed payload")
    parser.add_argument("output", type=Path, help="Destination PNG path")
    parser.add_argument(
        "--size",
        type=int,
        default=400,
        help="Square output size in pixels (default: 400)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        render_preview(args.input, args.output, args.size)
        return 0
    except Exception as exc:  # pragma: no cover - server wrapper
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
