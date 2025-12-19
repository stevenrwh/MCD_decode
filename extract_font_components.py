#!/usr/bin/env python3
r"""
Slice the inflated MAIN font payload into per-glyph blobs.

For each label that matches /(M92[A-Z0-9\-]+|MAIN[a-z0-9]*)/ we capture the
binary window that runs from a few bytes before the label up to the byte just
before the next label.  The chunks go into FONTS/components/<index>_<label>.bin
so they can be inspected independently.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

from catalog_font_labels import scan_labels


def extract_components(payload: bytes, *, pad: int = 32) -> list[tuple[str, int, int, bytes]]:
    matches = scan_labels(payload)
    prefix = _select_prefix(matches)
    if prefix:
        filtered: list[tuple[int, bytes]] = []
        for offset, label_bytes in matches:
            try:
                label_text = label_bytes.decode("ascii")
            except UnicodeDecodeError:
                continue
            if label_text.startswith(prefix):
                filtered.append((offset, label_bytes))
        if filtered:
            matches = filtered
    pieces: list[tuple[str, int, int, bytes]] = []
    for idx, (offset, label_bytes) in enumerate(matches):
        label = label_bytes.decode("ascii")
        start = max(0, offset - pad)
        end = matches[idx + 1][0] if idx + 1 < len(matches) else len(payload)
        chunk = payload[start:end]
        pieces.append((label, start, end, chunk))
    return pieces


def _select_prefix(matches: list[tuple[int, bytes]]) -> str | None:
    counts: collections.Counter[str] = collections.Counter()
    for _, label_bytes in matches:
        try:
            text = label_bytes.decode("ascii")
        except UnicodeDecodeError:
            continue
        idx = 0
        while idx < len(text) and text[idx].isalnum():
            idx += 1
        if idx >= 2:
            counts.update([text[:idx]])
    total = len(matches)
    if not counts or total == 0:
        return None
    prefix, count = counts.most_common(1)[0]
    if count / total < 0.5:
        return None
    return prefix


def main() -> int:
    parser = argparse.ArgumentParser(description="Split Mcalf*.fnt.decompressed into per-glyph chunks.")
    parser.add_argument("input", type=Path, help="Path to Mcalf092.fnt.decompressed")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("FONTS/components"),
        help="Destination directory for the chunks",
    )
    parser.add_argument(
        "--pad",
        type=int,
        default=32,
        help="Number of bytes to include before each label (default 32)",
    )
    args = parser.parse_args()
    blob = args.input.read_bytes()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pieces = extract_components(blob, pad=args.pad)
    for idx, (label, start, end, chunk) in enumerate(pieces, start=1):
        name = f"{idx:03d}_{label}.bin"
        path = args.output_dir / name
        path.write_bytes(chunk)
        print(f"{name}: offset=0x{start:05X}-0x{end:05X} ({len(chunk)} bytes)")
    print(f"\nExtracted {len(pieces)} component blobs into {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
