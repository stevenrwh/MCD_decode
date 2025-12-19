#!/usr/bin/env python3
"""
Scan decompressed payload for length-prefixed ASCII labels and print surrounding
numeric fields to infer the catalog structure.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path
from typing import Iterable

from mcd_to_dxf import brute_force_deflate


def scan_labels(payload: bytes, patterns: Iterable[str], window: int = 16, max_hits: int = 5) -> None:
    text = payload.decode("latin-1", errors="ignore")
    for pat in patterns:
        regex = re.compile(re.escape(pat))
        hits = [m.start() for m in regex.finditer(text)]
        print(f"Pattern '{pat}': {len(hits)} hits")
        for pos in hits[:max_hits]:
            start = max(0, pos - window)
            end = min(len(payload), pos + len(pat) + window)
            blob = payload[start:end]
            even = len(blob) // 2 * 2
            shorts = struct.unpack("<{}h".format(even // 2), blob[:even])
            print(f"  hit at {pos}, slice {start}:{end}, int16={shorts}")


def run(path: Path) -> None:
    raw = path.read_bytes()
    _, payload = brute_force_deflate(raw)
    print(f"{path.name}: payload {len(payload)} bytes")
    scan_labels(payload, ["FSK", "MAIN", "M38", "VM"])


def main() -> None:
    for p in [Path("new_style_mcd_files/TW84.mcd"), Path("old_style_mcd_files/TW84.mcd")]:
        if p.exists():
            run(p)
        else:
            print(f"{p} missing")


if __name__ == "__main__":
    main()
