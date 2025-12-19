#!/usr/bin/env python3
"""
Dump representative 0x4803 component sub-block payloads so we can observe the
actual structure (no decoding guesses). This is read-only and writes a small
report to stdout.
"""

from __future__ import annotations

import struct
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from component_parser import iter_component_definitions
from mcd_to_dxf import brute_force_deflate


def _shorts(payload: bytes, limit: int = 40) -> str:
    even = len(payload) // 2 * 2
    shorts = struct.unpack("<{}h".format(even // 2), payload[:even])
    return ", ".join(str(s) for s in shorts[:limit])


def _uints(payload: bytes, limit: int = 10) -> str:
    pad = (4 - len(payload) % 4) % 4
    data = payload + b"\x00" * pad
    uints = struct.unpack("<{}I".format(len(data) // 4), data)
    return ", ".join(str(u) for u in uints[:limit])


def _floats(payload: bytes, limit: int = 10) -> str:
    pad = (4 - len(payload) % 4) % 4
    data = payload + b"\x00" * pad
    floats = struct.unpack("<{}f".format(len(data) // 4), data)
    return ", ".join(f"{f:.6g}" for f in floats[:limit])


def dump_file(path: Path, max_lengths: int = 5) -> None:
    blob = path.read_bytes()
    _, payload = brute_force_deflate(blob)
    defs = list(iter_component_definitions(payload))
    if not defs:
        print(f"{path}: no component definitions found")
        return
    comp = defs[0]
    blocks = [blk for blk in comp.sub_blocks if blk.tag == 0x4803]
    lengths: defaultdict[int, list] = defaultdict(list)
    for blk in blocks:
        lengths[len(blk.payload)].append(blk)
    print(f"{path.name}: defs={len(defs)} blocks={len(blocks)} unique_lengths={len(lengths)}")
    for length, blks in sorted(lengths.items())[:max_lengths]:
        blk = blks[0]
        print(f"  len={length} count={len(blks)} dtype={blk.dtype} count_field={blk.count} off={blk.offset}")
        print(f"    shorts: { _shorts(blk.payload) }")
        print(f"    uints:  { _uints(blk.payload) }")
        print(f"    floats: { _floats(blk.payload) }")


def main() -> None:
    files: Iterable[Path] = (
        Path("new_style_mcd_files/TW84.mcd"),
        Path("new_style_mcd_files/JF6050-THOMPSON.mcd"),
    )
    for f in files:
        if f.exists():
            dump_file(f)
        else:
            print(f"{f} missing")


if __name__ == "__main__":
    main()
