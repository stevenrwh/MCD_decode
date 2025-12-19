#!/usr/bin/env python3
"""
Search raw and decompressed payloads for labels in ASCII and UTF-16 (LE/BE).
Helps locate component names stored in non-ASCII form.
"""

from __future__ import annotations

import re
from pathlib import Path
from mcd_to_dxf import brute_force_deflate


def encode_patterns(pat: str) -> list[tuple[str, bytes]]:
    return [
        ("ascii", pat.encode("ascii", errors="ignore")),
        ("utf16le", pat.encode("utf-16le")),
        ("utf16be", pat.encode("utf-16be")),
    ]


def search_blob(name: str, blob: bytes, patterns: list[str]) -> None:
    text = blob.decode("latin-1", errors="ignore")
    print(f"[{name}] size={len(blob)}")
    for pat in patterns:
        for enc_name, needle in encode_patterns(pat):
            hits = [m.start() for m in re.finditer(re.escape(needle.decode('latin-1', errors='ignore')), text)]
            print(f"  {pat} ({enc_name}): {len(hits)} hits; first 5: {hits[:5]}")


def main() -> None:
    pats = ["FSK", "MAIN", "M38", "VM"]
    path = Path("new_style_mcd_files/TW84.mcd")
    if not path.exists():
        print(f"{path} missing")
        return
    raw = path.read_bytes()
    _, payload = brute_force_deflate(raw)
    search_blob("raw", raw, pats)
    search_blob("payload", payload, pats)


if __name__ == "__main__":
    main()
