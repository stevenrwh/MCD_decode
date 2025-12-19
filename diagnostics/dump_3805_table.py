#!/usr/bin/env python3
"""
Dump the structured entries inside the large 0x3805 block of new-style TW84.
This appears to be a fixed-stride table (71 bytes per entry) containing
label + numeric fields that likely map labels to geometry indices/blocks.
"""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mcd_to_dxf import brute_force_deflate
from component_parser import iter_component_definitions


def dump_table(path: Path, *, stride: int = 71, limit: int = 50) -> None:
    raw = path.read_bytes()
    _, payload = brute_force_deflate(raw)
    definition = next(iter(iter_component_definitions(payload)))
    blocks_3805 = [blk for blk in definition.sub_blocks if getattr(blk, "tag", None) == 0x3805]
    if len(blocks_3805) < 2:
        print("No large 0x3805 block found")
        return
    data = blocks_3805[1].payload  # large block
    labels = []
    text = data.decode("latin-1", errors="ignore")
    for m in re.finditer(r"[A-Z0-9_]{3,}", text):
        labels.append((m.start(), m.group()))
    # Filter to expected label prefixes
    labels = [(pos, lbl) for pos, lbl in labels if lbl[:3] in ("FSK", "MAI", "M38", "VM", "SKC")]
    positions = [pos for pos, _ in labels]
    print(f"{path.name}: entries={len(labels)}, common stride assumption={stride}")
    for i, (pos, lbl) in enumerate(labels[:limit]):
        base = pos - 16
        chunk = data[base : base + stride]
        even = len(chunk) // 2 * 2
        shorts = struct.unpack("<{}h".format(even // 2), chunk[:even])
        print(f"#{i:03d} pos={pos:06d} label={lbl:8s} shorts={shorts}")


def main() -> None:
    target = Path("new_style_mcd_files/TW84.mcd")
    if target.exists():
        dump_table(target)
    else:
        print(f"{target} missing")


if __name__ == "__main__":
    main()
