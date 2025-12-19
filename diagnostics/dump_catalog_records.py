#!/usr/bin/env python3
"""
Dump the structured catalog records we discovered in the newer TW84 payloads.

Record layout (relative to the record start offset):
    0x00: u32 big-endian = 1   (constant in observed samples)
    0x04: u32 big-endian = 0   (constant)
    0x08: u32 big-endian = 0   (constant)
    0x0C: u32 big-endian = name length (N)
    0x10: N bytes of ASCII name (no padding)
    0x10+N: u16 big-endian = 0x0180 (flag/version)
    0x12+N: four little-endian float32 values
    0x22+N: u32 little-endian sentinel 0x075BCD15
    0x26+N: u32 little-endian index (likely geometry lookup)
    0x2A+N: u16 big-endian (usually 0)
    0x2C+N: u32 big-endian (usually 2)

We scan for the fixed header + sentinel pattern and print the decoded fields.
This is meant to be a lightweight inspection helper so we can correlate names
to geometry indices without modifying the converter.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

# Ensure project root is on sys.path so sibling modules resolve.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcd_to_dxf import brute_force_deflate


@dataclass(frozen=True)
class CatalogRecord:
    offset: int
    name: str
    index: int
    floats: tuple[float, float, float, float]
    tail2: int
    tail3: int


def iter_catalog_records(payload: bytes) -> Iterable[CatalogRecord]:
    """Yield catalog records that match the pattern described above."""

    sig = b"\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00"
    sentinel = b"\x15\xcd\x5b\x07"  # 123456789 little-endian
    idx = 0
    limit = len(payload)
    while True:
        pos = payload.find(sig, idx)
        if pos == -1:
            break
        if pos + 16 > limit:
            break
        name_len = struct.unpack(">I", payload[pos + 12 : pos + 16])[0]
        name_start = pos + 16
        name_end = name_start + name_len
        if name_end > limit:
            idx = pos + 1
            continue
        name_bytes = payload[name_start:name_end]
        if not name_bytes or any(b > 0x7F for b in name_bytes):
            idx = pos + 1
            continue
        flag_pos = name_end
        floats_pos = flag_pos + 2
        sentinel_pos = floats_pos + 16
        if sentinel_pos + 8 > limit:
            idx = pos + 1
            continue
        if payload[sentinel_pos:sentinel_pos + 4] != sentinel:
            idx = pos + 1
            continue
        try:
            floats = struct.unpack("<4f", payload[floats_pos: floats_pos + 16])
        except struct.error:
            idx = pos + 1
            continue
        index = struct.unpack("<I", payload[sentinel_pos + 4: sentinel_pos + 8])[0]
        tail2 = struct.unpack(">H", payload[sentinel_pos + 8: sentinel_pos + 10])[0]
        tail3 = struct.unpack(">I", payload[sentinel_pos + 10: sentinel_pos + 14])[0]
        yield CatalogRecord(
            offset=pos,
            name=name_bytes.decode("ascii"),
            index=index,
            floats=floats,
            tail2=tail2,
            tail3=tail3,
        )
        idx = pos + 1


def dump_catalog(path: Path) -> List[CatalogRecord]:
    raw = path.read_bytes()
    _, payload = brute_force_deflate(raw)
    recs = list(iter_catalog_records(payload))
    print(f"{path.name}: {len(recs)} catalog records")
    for rec in recs:
        f1, f2, f3, f4 = rec.floats
        print(
            f"  @{rec.offset:06d} name={rec.name:10s} "
            f"index={rec.index:3d} floats=({f1:.4f},{f2:.4f},{f3:.4f},{f4:.4f}) "
            f"tail2={rec.tail2} tail3={rec.tail3}"
        )
    return recs


def main() -> None:
    targets = [
        Path("new_style_mcd_files/TW84.mcd"),
        Path("old_style_mcd_files/TW84.mcd"),
    ]
    for path in targets:
        if path.exists():
            dump_catalog(path)
        else:
            print(f"{path} missing")


if __name__ == "__main__":
    main()
