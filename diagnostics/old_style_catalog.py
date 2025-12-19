#!/usr/bin/env python3
"""
Extract label/catalog-like records from older-style decompressed .mcd payloads.
Useful for mapping label strings (FSK, MAIN, M38, VM, etc.) back to offsets so
we can align them with geometry blocks when no formal catalog table exists.
"""

from __future__ import annotations

import argparse
import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from monucad.deflate_io import brute_force_deflate


@dataclass
class LabelRecord:
    label: str
    offset: int
    shorts: tuple[int, ...]


def extract_catalog(payload: bytes, patterns: Iterable[str], *, window_shorts: int = 16) -> List[LabelRecord]:
    text = payload.decode("latin-1", errors="ignore")
    records: List[LabelRecord] = []
    for pat in patterns:
        for m in re.finditer(re.escape(pat), text):
            pos = m.start()
            start = max(0, pos - window_shorts * 2)
            end = min(len(payload), pos + len(pat) + window_shorts * 2)
            blob = payload[start:end]
            even = len(blob) // 2 * 2
            shorts = struct.unpack("<{}h".format(even // 2), blob[:even])
            records.append(LabelRecord(label=pat, offset=pos, shorts=shorts))
    return records


def run(path: Path, patterns: Sequence[str], *, window_shorts: int = 16) -> List[LabelRecord]:
    raw = path.read_bytes()
    _, payload = brute_force_deflate(raw)
    return extract_catalog(payload, patterns, window_shorts=window_shorts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract label/catalog records from older-style .mcd payloads (heuristic)."
    )
    parser.add_argument("input", type=Path, help="Path to .mcd file")
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["FSK", "MAIN", "M38", "VM"],
        help="Label strings to search for",
    )
    parser.add_argument(
        "--window-shorts",
        type=int,
        default=16,
        help="How many int16 values to include before/after each label hit",
    )
    args = parser.parse_args(argv)
    records = run(args.input, args.patterns, window_shorts=args.window_shorts)
    print(f"{args.input.name}: {len(records)} label hits")
    for rec in records:
        print(f"offset={rec.offset} label={rec.label} shorts={rec.shorts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
