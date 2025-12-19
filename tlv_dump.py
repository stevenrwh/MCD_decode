#!/usr/bin/env python3
"""
Minimal TLV dumper for Monu-CAD resources (.fnt archives, text blocks, etc.).

The records we have seen so far follow this layout (little endian):

    uint16 tag_id
    uint16 data_type    # 0xFFFF → ASCII string, others TBD
    uint16 item_count   # Often 1
    uint16 byte_count   # Length of the payload right after the header
    <payload bytes>

This tool iterates those records and prints a compact summary so we can eyeball
which tags correspond to glyph names, numeric arrays, style tables, etc.
"""

from __future__ import annotations

import argparse
import itertools
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

# Known/suspected data types. We only special-case strings for now.
TYPE_ASCII = 0xFFFF


@dataclass(frozen=True)
class TLVField:
    offset: int
    tag: int
    dtype: int
    count: int
    size: int
    payload: bytes

    def as_ascii(self) -> str | None:
        if self.dtype != TYPE_ASCII or self.size == 0:
            return None
        try:
            return self.payload.decode("ascii")
        except UnicodeDecodeError:
            return None


def iter_tlv(
    blob: bytes,
    *,
    start: int = 0,
    stop: int | None = None,
    require_ascii: bool = False,
) -> Iterator[TLVField]:
    """Yield TLV fields from the byte blob.

    The stream occasionally contains non-TLV padding or binary blobs, so we
    resync by sliding forward two bytes whenever a candidate header would run
    past EOF.
    """

    limit = len(blob) if stop is None else min(stop, len(blob))
    offset = max(0, start)
    while offset + 8 <= limit:
        tag, dtype, count, size = struct.unpack_from("<HHHH", blob, offset)
        payload_offset = offset + 8
        payload_end = payload_offset + size
        if size == 0 and dtype == 0 and tag == 0:
            offset += 2
            continue
        if payload_end > limit or size > limit:
            offset += 2  # Not a valid TLV header; slide forward and retry.
            continue
        payload = blob[payload_offset:payload_end]
        field = TLVField(offset=offset, tag=tag, dtype=dtype, count=count, size=size, payload=payload)
        if require_ascii and field.as_ascii() is None:
            offset = payload_end
            continue
        yield field
        offset = payload_end


def describe_field(field: TLVField, *, show_bytes: bool = False) -> str:
    parts = [
        f"off=0x{field.offset:04X}",
        f"tag=0x{field.tag:04X}",
        f"type=0x{field.dtype:04X}",
        f"count={field.count}",
        f"size={field.size}",
    ]
    ascii_text = field.as_ascii()
    if ascii_text is not None:
        parts.append(f"ascii={ascii_text!r}")
    elif show_bytes and field.size:
        sample = " ".join(f"{b:02X}" for b in field.payload[: min(16, field.size)])
        if field.size > 16:
            sample += " …"
        parts.append(f"bytes={sample}")
    return " | ".join(parts)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump TLV records from a Monu-CAD blob.")
    parser.add_argument("input", type=Path, help="Path to the binary blob (.decompressed, .fnt, etc.)")
    parser.add_argument("--start", type=lambda x: int(x, 0), default=0, help="Byte offset to start parsing (default 0)")
    parser.add_argument("--stop", type=lambda x: int(x, 0), help="Optional exclusive stop offset")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of TLV records to print (default: no limit)",
    )
    parser.add_argument(
        "--bytes",
        action="store_true",
        help="Include a short hex dump for non-string payloads",
    )
    parser.add_argument(
        "--strings-only",
        action="store_true",
        help="Emit only fields whose dtype looks like ASCII (0xFFFF)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    blob = args.input.read_bytes()
    fields = iter_tlv(blob, start=args.start, stop=args.stop, require_ascii=args.strings_only)
    if args.limit is not None:
        fields = itertools.islice(fields, args.limit)
    count = 0
    for field in fields:
        print(describe_field(field, show_bytes=args.bytes))
        count += 1
    if count == 0:
        print("No TLV records discovered in the requested window.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
