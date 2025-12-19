#!/usr/bin/env python3
"""
Decode the tiny secondary deflate payload embedded inside Mcalf092.fnt.

The 156-byte stream at offset 0x97D inflates to a highly repetitive blob that
looks like `0x94 0x54` noise.  Each 32-bit little-endian word is actually the
base pattern `0x54945494` plus an obfuscated value that has been left-shifted by
6 bits.  Undoing that shift (and subtracting the base) recovers 39 signed
integers that appear to be pairs of 16-bit quantities (-1, 255, etc.).

This tool prints the decoded table so we can start mapping those values back to
stroke indices / ranges.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

BASE_PATTERN = 0x54945494
SHIFT_BITS = 6
ENTRY_COUNT = 39  # 156 bytes / 4


def decode_words(blob: bytes) -> list[dict]:
    if len(blob) != ENTRY_COUNT * 4:
        raise ValueError(f"Expected {ENTRY_COUNT*4} bytes, found {len(blob)}")
    words = struct.unpack("<{}I".format(ENTRY_COUNT), blob)
    entries: list[dict] = []
    for index, raw in enumerate(words):
        diff = (raw - BASE_PATTERN) & 0xFFFFFFFF
        value = ((raw - BASE_PATTERN) >> SHIFT_BITS)
        remainder = diff & ((1 << SHIFT_BITS) - 1)
        unsigned = value & 0xFFFFFFFF
        hi = ((unsigned >> 16) + 0x8000) % 0x10000 - 0x8000
        lo = ((unsigned & 0xFFFF) + 0x8000) % 0x10000 - 0x8000
        entries.append(
            {
                "index": index,
                "raw": raw,
                "value": value,
                "hi": hi,
                "lo": lo,
                "remainder": remainder,
            }
        )
    return entries


def load_blob(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) < ENTRY_COUNT * 4:
        raise ValueError(f"{path} only contains {len(data)} bytes")
    return data[: ENTRY_COUNT * 4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decode the secondary stroke-index payload.")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("FONTS/Mcalf092_second.decompressed"),
        help="Path to the 156-byte decompressed payload (default: FONTS/Mcalf092_second.decompressed)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blob = load_blob(args.input)
    entries = decode_words(blob)
    print(f"Decoded {len(entries)} entries from {args.input}:")
    for entry in entries:
        print(
            f"[{entry['index']:02d}] raw=0x{entry['raw']:08X} "
            f"value=0x{entry['value'] & 0xFFFFFFFF:08X} "
            f"hi={entry['hi']:6d} lo={entry['lo']:6d} "
            f"remainder=0x{entry['remainder']:X}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
