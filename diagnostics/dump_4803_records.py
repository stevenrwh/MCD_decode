#!/usr/bin/env python3
"""
Diagnostic: dump structured views of 0x4803 component sub-blocks.

We group payloads into fixed-size short records and report min/max to spot which
fields look like coordinates (likely scaled by 1/32768) versus flags/ids.
No converter changes; read-only.
"""

from __future__ import annotations

import math
import struct
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from component_parser import ComponentSubBlock, iter_component_definitions
from mcd_to_dxf import brute_force_deflate


def _chunk_shorts(payload: bytes, record_shorts: int) -> list[tuple[int, ...]]:
    even = len(payload) // 2 * 2
    shorts = struct.unpack("<{}h".format(even // 2), payload[:even])
    records = []
    for i in range(0, len(shorts), record_shorts):
        chunk = shorts[i : i + record_shorts]
        if len(chunk) == record_shorts:
            records.append(chunk)
    return records


def _coord_stats(values: Iterable[int], scale: float) -> tuple[float, float]:
    scaled = [v * scale for v in values]
    if not scaled:
        return 0.0, 0.0
    return min(scaled), max(scaled)


def _analyze_block(blk: ComponentSubBlock, record_shorts: int) -> str:
    records = _chunk_shorts(blk.payload, record_shorts)
    if not records:
        return "  (no records)"
    cols = list(zip(*records))
    lines = []
    lines.append(f"  records={len(records)} len={len(blk.payload)}")
    for idx, col in enumerate(cols):
        mn, mx = min(col), max(col)
        mn_s, mx_s = _coord_stats(col, 1.0 / 32768.0)
        lines.append(
            f"    col{idx}: min={mn} max={mx} scaled=[{mn_s:.4f},{mx_s:.4f}] sample={col[:5]}"
        )
    return "\n".join(lines)


def dump_file(path: Path, record_shorts: int = 4, max_lengths: int = 5) -> None:
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
        print(f" len={length} count={len(blks)} dtype={blk.dtype} count_field={blk.count} off={blk.offset}")
        print(_analyze_block(blk, record_shorts))


def main() -> None:
    targets = [
        Path("new_style_mcd_files/TW84.mcd"),
        Path("new_style_mcd_files/JF6050-THOMPSON.mcd"),
    ]
    for t in targets:
        if t.exists():
            dump_file(t, record_shorts=4)
        else:
            print(f"{t} missing")


if __name__ == "__main__":
    main()
