#!/usr/bin/env python3
"""
Extract structured label records from the decompressed payload and list their fields.
Intended for old/new style TW84 where labels (FSK, MAIN, M38, VM, etc.) appear
in a consistent int16 pattern.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from mcd_to_dxf import brute_force_deflate


@dataclass
class LabelRecord:
    label: str
    offset: int
    fields: tuple[int, ...]


def extract_catalog(payload: bytes, patterns: Iterable[str], window_shorts: int = 16) -> List[LabelRecord]:
    text = payload.decode("latin-1", errors="ignore")
    records: List[LabelRecord] = []
    for pat in patterns:
        for m in re.finditer(re.escape(pat), text):
            pos = m.start()
            start = max(0, pos - window_shorts * 2)
            end = min(len(payload), pos + len(pat) + window_shorts * 2)
            blob = payload[start:end]
            even = len(blob) // 2 * 2
            shorts = struct.unpack("<{}h".format(even // 2), blob[:even])
            records.append(LabelRecord(label=pat, offset=pos, fields=shorts))
    return records


def dump_catalog(path: Path, patterns: Iterable[str]) -> None:
    raw = path.read_bytes()
    _, payload = brute_force_deflate(raw)
    recs = extract_catalog(payload, patterns)
    print(f"{path.name}: {len(recs)} catalog entries")
    for rec in recs[:10]:
        print(f"  label={rec.label} offset={rec.offset} fields={rec.fields}")


def main() -> None:
    patterns = ["FSK", "MAIN", "M38", "VM"]
    for p in [
        Path("new_style_mcd_files/TW84.mcd"),
        Path("old_style_mcd_files/TW84.mcd"),
    ]:
        if p.exists():
            dump_catalog(p, patterns)
        else:
            print(f"{p} missing")


if __name__ == "__main__":
    main()
