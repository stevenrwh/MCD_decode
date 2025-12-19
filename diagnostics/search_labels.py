#!/usr/bin/env python3
"""
Scan an .mcd file for ASCII component labels (e.g., FSK*, MAIN*, M38*) and
report their offsets. Read-only; intended to locate where definitions may be
stored outside the parsed deflate payload.
"""

from __future__ import annotations

import re
from pathlib import Path


def search_labels(path: Path, patterns: list[str]) -> None:
    data = path.read_bytes()
    text = data.decode("latin-1", errors="ignore")
    print(f"{path.name}: scanning for patterns {patterns}")
    for pat in patterns:
        regex = re.compile(re.escape(pat))
        hits = [m.start() for m in regex.finditer(text)]
        print(f"  {pat}: {len(hits)} hits; first 5 offsets: {hits[:5]}")


def main() -> None:
    patterns = ["FSK", "MAIN", "M38", "VM"]
    for p in [Path("new_style_mcd_files/TW84.mcd")]:
        if p.exists():
            search_labels(p, patterns)
        else:
            print(f"{p} missing")


if __name__ == "__main__":
    main()
