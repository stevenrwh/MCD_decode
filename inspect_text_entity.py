#!/usr/bin/env python3
"""
Helper to dissect the editable-text block captured in TEST_WITH_FONT.mcd.

The structure we currently observe (starting near offset 0x21F in the deflated
payload) is:

    uint8 text_len
    text bytes (ASCII)
    9 floats (likely placement/scaling metadata)
    uint8 font_len + font string
    repeated uint8/string annotations (version/kern/style notes)
    repeated layer/style descriptors (len + ASCII)
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def _try_prefixed_string(blob: bytes, offset: int) -> tuple[str | None, int]:
    if offset >= len(blob):
        return None, offset
    length = blob[offset]
    start = offset + 1
    end = start + length
    if end > len(blob):
        return None, offset
    chunk = blob[start:end]
    try:
        text = chunk.decode("ascii")
    except UnicodeDecodeError:
        return None, offset
    return text, end


def _read_c_string(blob: bytes, offset: int) -> tuple[str, int]:
    cursor = offset
    while cursor < len(blob) and blob[cursor] != 0:
        cursor += 1
    text = blob[offset:cursor].decode("ascii")
    return text, min(cursor + 1, len(blob))


def read_lp_string(blob: bytes, offset: int) -> tuple[str, int]:
    """
    Legacy helper that prefers length-prefixed strings but falls back to C-style
    null-terminated ASCII when the prefix is actually the first character (Vm samples).
    """

    text, new_offset = _try_prefixed_string(blob, offset)
    if text is not None:
        return text, new_offset
    return _read_c_string(blob, offset)


def scan_ascii_runs(blob: bytes, start: int, *, limit: int, min_len: int = 4) -> list[str]:
    strings: list[str] = []
    current: list[int] = []
    offset = start
    while offset < len(blob) and len(strings) < limit:
        byte = blob[offset]
        if 32 <= byte < 127:
            current.append(byte)
        else:
            if len(current) >= min_len:
                strings.append(bytes(current).decode("ascii"))
                if len(strings) >= limit:
                    break
            current = []
        offset += 1
    if len(strings) < limit and len(current) >= min_len:
        strings.append(bytes(current).decode("ascii"))
    return strings


def parse_text_block(blob: bytes, start: int, *, float_count: int = 9) -> dict:
    cursor = start
    text, cursor = read_lp_string(blob, cursor)
    while cursor < len(blob) and blob[cursor] == 0:
        cursor += 1
    cursor = (cursor + 3) & ~3
    floats = list(struct.unpack_from(f"<{float_count}f", blob, cursor))
    cursor += float_count * 4

    while floats and cursor < len(blob) and blob[cursor] >= 32:
        cursor -= 4
        floats.pop()
    font, cursor = read_lp_string(blob, cursor)
    notes: list[str] = []
    for _ in range(3):
        note, cursor = read_lp_string(blob, cursor)
        notes.append(note)
    tail_strings = scan_ascii_runs(blob, cursor, limit=12)
    return {
        "text": text,
        "metrics": floats,
        "font": font,
        "notes": notes,
        "tail_strings": tail_strings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect the editable text block inside a .mcd payload.")
    parser.add_argument("input", type=Path, help="Path to TEST_WITH_FONT.decompressed (deflated payload)")
    parser.add_argument("--offset", type=lambda x: int(x, 0), default=0x21F, help="Start offset of the text chunk")
    parser.add_argument("--float-count", type=int, default=9, help="Number of floats to read before the font field")
    args = parser.parse_args()
    blob = args.input.read_bytes()
    info = parse_text_block(blob, args.offset, float_count=args.float_count)
    print(f"Text string: {info['text']!r}")
    print(f"Metrics ({len(info['metrics'])} floats):")
    for idx, value in enumerate(info["metrics"]):
        print(f"  f{idx}: {value}")
    print(f"Font: {info['font']}")
    print("Notes:")
    for note in info["notes"]:
        print(f"  - {note}")
    print("Tail ASCII strings (best-effort scan):")
    for item in info["tail_strings"]:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
