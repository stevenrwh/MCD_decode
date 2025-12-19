#!/usr/bin/env python3
"""
Quick-and-dirty scanner that lists glyph/component labels embedded inside a
Monu-CAD .fnt payload.  We focus on identifiers that start with "M92" (MAIN
font) or "MAIN" followed by a letter/digit.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def _load_prefixes() -> list[str]:
    prefixes: set[str] = {"M92", "MAIN", "VM"}
    fonts_dir = Path(__file__).resolve().parent / "FONTS"
    config_path = fonts_dir / "mcfonts.lst"
    if not config_path.exists():
        return sorted(prefixes, key=len, reverse=True)
    with config_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                continue
            if "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", 1)]
            if key.lower() != "dtafile":
                continue
            value = value.split(";", 1)[0].strip()
            token = re.sub(r"[^0-9A-Za-z]", "", value).upper()
            if not token:
                continue
            prefixes.add(token)
            prefixes.add(f"F{token}")
    filtered = [prefix for prefix in prefixes if 2 <= len(prefix) <= 8]
    return sorted(filtered, key=lambda item: (-len(item), item))


_PREFIXES = _load_prefixes()
if _PREFIXES:
    _pattern_bytes = b"|".join(prefix.encode("ascii") + rb"[A-Z0-9&\-]*" for prefix in _PREFIXES)
    PATTERN = re.compile(rb"(" + _pattern_bytes + rb")")
else:
    PATTERN = re.compile(rb"(M92[A-Z0-9\-]+|MAIN[a-z0-9]*|VM[A-Z0-9&\-]+)")


def scan_labels(blob: bytes) -> list[tuple[int, bytes]]:
    matches: list[tuple[int, bytes]] = []
    for match in PATTERN.finditer(blob):
        label = match.group(1)
        matches.append((match.start(), label))
    return matches


def dump_context(blob: bytes, offset: int, width: int = 16) -> str:
    start = max(0, offset - width)
    end = min(len(blob), offset + width)
    window = blob[start:end]
    hex_bytes = " ".join(f"{b:02X}" for b in window)
    return f"{start:#06x}-{end:#06x}: {hex_bytes}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List MAIN font glyph labels inside a .fnt payload.")
    parser.add_argument("input", type=Path, help="Path to Mcalf*.fnt.decompressed (or raw deflated payload)")
    parser.add_argument(
        "--context",
        type=int,
        default=32,
        help="Number of bytes of hex context to print before/after each label (default 32)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of matches to print",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blob = args.input.read_bytes()
    matches = scan_labels(blob)
    total = len(matches)
    if args.limit is not None:
        matches = matches[: args.limit]
    for idx, (offset, label) in enumerate(matches, start=1):
        ctx = dump_context(blob, offset, width=args.context)
        print(f"[{idx:04d}/{total}] off=0x{offset:05X} label={label.decode('ascii')} | {ctx}")
    print(f"\nTotal matches: {total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
