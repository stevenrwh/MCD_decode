#!/usr/bin/env python3
"""
Dump context around ASCII label hits inside the decompressed payload of an .mcd.
This helps reveal the surrounding record structure so we can write a parser.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from mcd_to_dxf import brute_force_deflate


def dump_context(path: Path, labels: Iterable[str], window: int = 64, max_hits: int = 3) -> None:
    raw = path.read_bytes()
    _, payload = brute_force_deflate(raw)
    print(f"{path.name}: payload {len(payload)} bytes")
    for label in labels:
        needle = label.encode("ascii", errors="ignore")
        print(f"\nLabel '{label}':")
        idx = 0
        hits = 0
        while hits < max_hits:
            pos = payload.find(needle, idx)
            if pos == -1:
                break
            start = max(0, pos - window)
            end = min(len(payload), pos + len(needle) + window)
            blob = payload[start:end]
            print(f"  hit at {pos} (slice {start}:{end})")
            # show hex
            hex_bytes = blob.hex(" ")
            print(f"    hex: {hex_bytes}")
            # show as little-endian int16
            even = len(blob) // 2 * 2
            shorts = []
            if even >= 2:
                shorts = [int.from_bytes(blob[i:i+2], "little", signed=True) for i in range(0, even, 2)]
            print(f"    int16: {shorts}")
            idx = pos + len(needle)
            hits += 1


def main() -> None:
    targets = [Path("new_style_mcd_files/TW84.mcd")]
    labels = ["FSK", "MAIN", "M38", "VM"]
    for t in targets:
        if t.exists():
            dump_context(t, labels)
        else:
            print(f"{t} missing")


if __name__ == "__main__":
    main()
