#!/usr/bin/env python3
"""
Summarize placement-like records inside each component sub-block. This helps
isolate which 0x4803 blocks actually contain glyph placement tables versus the
fixed 0x3805 label table.
"""

from __future__ import annotations

import argparse
import math
import re
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from component_parser import ComponentSubBlock, iter_component_definitions
from mcd_to_dxf import brute_force_deflate
from placement_parser import GlyphPlacementRecord, VALUES_PADDING_BYTES, extract_glyph_records_with_offsets


def _looks_like_stride_table(payload: bytes) -> bool:
    """
    Heuristic cribbed from placement_parser.extract_glyph_records(): skip the
    fixed-stride label tables (e.g., the large 0x3805 block) that lack the
    placement sentinel but are packed with ASCII labels at ~71 byte intervals.
    """

    if b"\x15\xcd\x5b\x07" in payload:
        return False
    sample = payload[: min(len(payload), 2048)]
    hits = [m.start() for m in re.finditer(rb"[A-Z0-9_]{3,8}", sample)]
    if len(hits) < 5:
        return False
    deltas = [hits[i + 1] - hits[i] for i in range(len(hits) - 1)]
    if not deltas:
        return False
    common = max(set(deltas), key=deltas.count)
    return common in (70, 71, 72)


def _scan_block_raw(
    payload: bytes,
    *,
    allow_spaces: bool = True,
    max_label_len: int = 80,
) -> List[Tuple[int, GlyphPlacementRecord]]:
    """
    Lightweight replica of extract_glyph_records() that runs directly on a block
    payload and reports record offsets even when the stride heuristic would have
    short-circuited the structured parser.
    """

    pad = b"\x01" + (b"\x00" * (VALUES_PADDING_BYTES - 1))
    records: list[tuple[int, GlyphPlacementRecord]] = []
    seen: set[int] = set()
    search = 0
    blob_len = len(payload)

    while True:
        pos = payload.find(pad, search)
        if pos == -1:
            break
        record_start = pos - 40
        if record_start < 0 or record_start in seen:
            search = pos + 1
            continue
        length_pos = pos + VALUES_PADDING_BYTES
        if length_pos >= blob_len:
            break
        label_len = payload[length_pos]
        if not (1 <= label_len <= max_label_len):
            search = pos + 1
            continue
        label_start = length_pos + 1
        label_end = label_start + label_len
        if label_end > blob_len:
            search = pos + 1
            continue
        label_bytes = payload[label_start:label_end]
        if not label_bytes.strip():
            search = pos + 1
            continue
        if any(byte < 32 or byte > 126 for byte in label_bytes):
            search = pos + 1
            continue
        if not allow_spaces and any(byte == 32 for byte in label_bytes):
            search = pos + 1
            continue
        try:
            values = struct.unpack_from("<5d", payload, record_start)
        except struct.error:
            search = pos + 1
            continue
        tx, ty, rot_deg, sx, sy = values
        if not all(math.isfinite(val) for val in values):
            search = pos + 1
            continue
        if abs(tx) > 1e7 or abs(ty) > 1e7:
            search = pos + 1
            continue
        if abs(rot_deg) > 1e5:
            search = pos + 1
            continue
        if abs(sx) > 1e6 or abs(sy) > 1e6:
            search = pos + 1
            continue
        if abs(sx) < 1e-12 and abs(sy) < 1e-12:
            search = pos + 1
            continue
        footer_offset = label_end
        if footer_offset + 20 <= blob_len:
            footer = struct.unpack_from("<IIIII", payload, footer_offset)
        else:
            footer = (0, 0, 0, 0, 0)
        try:
            label = label_bytes.decode("ascii")
        except UnicodeDecodeError:
            search = pos + 1
            continue
        record = GlyphPlacementRecord(
            label=label,
            values=values,
            component_id=footer[0],
            field1=footer[1],
            field2=footer[2],
            field3=footer[3],
            data_offset=footer[4],
        )
        records.append((record_start, record))
        seen.add(record_start)
        search = pos + 1

    return records


def _format_range(values: Iterable[float]) -> str:
    vals = list(values)
    return f"[{min(vals):.2f}, {max(vals):.2f}]"


def summarize(path: Path, *, max_label_len: int = 80, raw: bool = False) -> None:
    raw = path.read_bytes()
    _, payload = brute_force_deflate(raw)
    for def_idx, definition in enumerate(iter_component_definitions(payload)):
        block_base = definition.offset + len(b"CComponentDefinition") + 44
        for blk_idx, block in enumerate(definition.sub_blocks):
            if block.tag not in (0x3805, 0x4803):
                continue
            records = extract_glyph_records_with_offsets(
                block.payload, allow_spaces=True, max_label_len=max_label_len
            )
            used_raw = False
            if not records:
                if _looks_like_stride_table(block.payload):
                    print(
                        f"[def {def_idx}] blk {blk_idx:04d} tag=0x{block.tag:04X} "
                        f"size={len(block.payload)} -> skipped (looks like fixed-stride label table)"
                    )
                # A raw scan can still be useful when the table heuristics reject
                # a block; keep it behind an opt-in flag to avoid noisy matches.
                if not raw:
                    continue
                records = _scan_block_raw(block.payload, allow_spaces=True, max_label_len=max_label_len)
                used_raw = True
            if not records:
                continue
            labels = [rec.label for _, rec in records]
            tx = [rec.values[0] for _, rec in records]
            ty = [rec.values[1] for _, rec in records]
            rot = [rec.values[2] for _, rec in records]
            sx = [rec.values[3] for _, rec in records]
            sy = [rec.values[4] for _, rec in records]
            comp_ids = {rec.component_id for _, rec in records}
            field2 = {rec.field2 for _, rec in records}
            offsets = [block_base + block.offset + start for start, _ in records]
            source_note = " raw" if used_raw else ""
            summary = (
                f"[def {def_idx}] blk {blk_idx:04d} tag=0x{block.tag:04X} "
                f"dtype={block.dtype} count={block.count} size={len(block.payload)} "
                f"abs=({min(offsets)}-{max(offsets)}) "
                f"recs={len(records)} labels={len(set(labels))}{source_note}"
            )
            print(summary)
            print(
                f"  tx={_format_range(tx)} ty={_format_range(ty)} "
                f"rot={_format_range(rot)} sx={_format_range(sx)} sy={_format_range(sy)} "
                f"comp_ids={sorted(comp_ids)} field2={sorted(field2)}"
            )
            top_labels = ", ".join(lbl for lbl, _ in Counter(labels).most_common(5))
            print(f"  top labels: {top_labels}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default="new_style_mcd_files/TW84.mcd", help="Path to .mcd file")
    parser.add_argument("--max-label-len", type=int, default=80, help="Max label length to accept when scanning")
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Also run a raw padding-sentinel scan when the structured parser rejects a block",
    )
    args = parser.parse_args()
    summarize(Path(args.input), max_label_len=args.max_label_len, raw=args.raw)


if __name__ == "__main__":
    main()
