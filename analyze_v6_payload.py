#!/usr/bin/env python3
"""
Quick-and-dirty instrumentation for the Monu-CAD "v6" payloads.

Usage:
    python analyze_v6_payload.py

It scans every *.decompressed sample (e.g. line_0,0_to_*.decompressed),
identifies which byte positions actually change across the corpus, and
emits a short report so we can focus on the regions that matter.
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_samples(pattern: str = "line_0,0_to_*.decompressed") -> dict[str, bytes]:
    samples = {}
    for item in sorted(Path(".").glob(pattern)):
        samples[item.name] = item.read_bytes()
    if not samples:
        raise SystemExit("No *.decompressed samples were found.")
    lengths = {len(data) for data in samples.values()}
    if len(lengths) != 1:
        raise SystemExit(f"Samples have differing sizes: {sorted(lengths)}")
    return samples


def summarize_variability(samples: dict[str, bytes]) -> None:
    names = list(samples.keys())
    data = np.frombuffer(b"".join(samples[name] for name in names), dtype=np.uint8)
    rows = len(names)
    cols = len(samples[names[0]])
    matrix = data.reshape(rows, cols)
    varying = np.where(matrix.max(axis=0) != matrix.min(axis=0))[0]

    print(f"{rows} samples, {cols} bytes each")
    print(f"{len(varying)} byte positions differ across the set.")
    if len(varying) > 0:
        print("First 32 variable offsets:", varying[:32])
        print("Last 32 variable offsets:", varying[-32:])


def dump_region(name: str, data: bytes, start: int, length: int = 128) -> None:
    print(f"\n{name}: bytes {start:#04x}-{start+length:#04x}")
    for offset in range(start, start + length, 16):
        chunk = data[offset : offset + 16]
        hexpart = chunk.hex()
        print(f"{offset:04x}: {hexpart}")


def main() -> None:
    samples = load_samples()
    summarize_variability(samples)

    # Heuristic: the first 0x100 bytes + a small tail are the only regions
    # that ever change across the line_0,0_to_* corpus. Dump them for eyeballing.
    for name, data in list(samples.items())[:3]:  # only print a few to keep output short
        dump_region(name, data, 0x0000, 0x100)
        dump_region(name, data, 0x01b0, 0x40)
        dump_region(name, data, 0x01c0, 0x40)


if __name__ == "__main__":
    main()
