#!/usr/bin/env python3
"""
Dump helper for M92.dta (MAIN font spacing/kerning table).

The file contains 9,025 uint16 entries which fits 95x95 (printable ASCII). We
assume index [row,col] corresponds to character pairs (row=first glyph, col=next
glyph).  Values appear to be thousandths of an em (1000 nominal), with zeros for
unused combos.
"""

from __future__ import annotations

import argparse
import json
import string
from pathlib import Path


PRINTABLE = [chr(i) for i in range(32, 127)]


def load_matrix(path: Path) -> list[list[int]]:
    data = path.read_bytes()
    if len(data) != 9025 * 2:
        raise SystemExit(f"Unexpected M92.dta size: {len(data)} bytes")
    vals = [int.from_bytes(data[i : i + 2], "little") for i in range(0, len(data), 2)]
    size = 95
    return [vals[i * size : (i + 1) * size] for i in range(size)]


def find_non_default(matrix: list[list[int]], *, default: int = 1000) -> list[tuple[int, int, int]]:
    special: list[tuple[int, int, int]] = []
    for r, row in enumerate(matrix):
        for c, value in enumerate(row):
            if value not in (default, 0):
                special.append((r, c, value))
    return special


def describe_pair(r: int, c: int) -> tuple[str, str]:
    first = PRINTABLE[r] if 0 <= r < len(PRINTABLE) else f"?{r}"
    second = PRINTABLE[c] if 0 <= c < len(PRINTABLE) else f"?{c}"
    return first, second


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MAIN font kerning values from M92.dta")
    parser.add_argument("input", type=Path, help="Path to M92.dta")
    parser.add_argument("--top", type=int, default=40, help="How many non-default pairs to print (default 40)")
    parser.add_argument("--json", type=Path, help="Optional JSON dump path for all non-default entries")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    matrix = load_matrix(args.input)
    specials = find_non_default(matrix)
    print(f"Matrix: {len(matrix)}x{len(matrix[0])} entries")
    print(f"Non-default kerning pairs: {len(specials)}")
    for pair in specials[: args.top]:
        r, c, value = pair
        first, second = describe_pair(r, c)
        print(f"row={r:02d} ({first!r}) col={c:02d} ({second!r}) -> {value}")
    if args.json:
        payload = []
        for r, c, value in specials:
            first, second = describe_pair(r, c)
            payload.append({"row": r, "col": c, "first": first, "second": second, "value": value})
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"JSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
