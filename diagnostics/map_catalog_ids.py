#!/usr/bin/env python3
"""
Summarize label catalog entries and search for their id1/id2 pairs in the payload.
Helps link labels to geometry blocks without guessing.
"""

from __future__ import annotations

import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from mcd_to_dxf import brute_force_deflate


def load_catalog(payload: bytes, window_shorts: int = 16) -> list[tuple[str, tuple[int, ...]]]:
    entries: list[tuple[str, tuple[int, ...]]] = []
    idx = 0
    limit = len(payload)
    while idx + 1 < limit:
        length = payload[idx]
        if not (1 <= length <= 16):
            idx += 1
            continue
        end = idx + 1 + length
        if end > limit:
            break
        label_bytes = payload[idx + 1 : end]
        try:
            label = label_bytes.decode("ascii")
        except UnicodeDecodeError:
            idx += 1
            continue
        if not label.isprintable():
            idx += 1
            continue
        start = max(0, idx - window_shorts * 2)
        stop = min(limit, end + window_shorts * 2)
        blob = payload[start:stop]
        even = len(blob) // 2 * 2
        shorts = struct.unpack("<{}h".format(even // 2), blob[:even])
        entries.append((label, shorts))
        idx = end
    return entries


def map_ids(payload: bytes, entries: list[tuple[str, tuple[int, ...]]], id_pos: tuple[int, int] = (8, 9)) -> None:
    ids_to_labels: defaultdict[tuple[int, int], list[str]] = defaultdict(list)
    for label, shorts in entries:
        if len(shorts) <= max(id_pos):
            continue
        key = (shorts[id_pos[0]], shorts[id_pos[1]])
        ids_to_labels[key].append(label)
    print(f"Total id pairs: {len(ids_to_labels)}")
    common = Counter({k: len(v) for k, v in ids_to_labels.items()})
    for key, cnt in common.most_common(10):
        print(f"  id pair {key} -> {cnt} labels (sample {ids_to_labels[key][:5]})")
    # search payload for id pair bytes
    for key in list(common.keys())[:5]:
        b = struct.pack("<hh", key[0], key[1])
        hits = []
        start = 0
        while True:
            pos = payload.find(b, start)
            if pos == -1:
                break
            hits.append(pos)
            start = pos + 1
        print(f"  id pair {key} occurs {len(hits)} times; first hits {hits[:5]}")


def run(path: Path) -> None:
    raw = path.read_bytes()
    _, payload = brute_force_deflate(raw)
    entries = load_catalog(payload)
    print(f"{path.name}: catalog entries {len(entries)}")
    map_ids(payload, entries)


def main() -> None:
    for p in [
        Path("new_style_mcd_files/TW84.mcd"),
        Path("old_style_mcd_files/TW84.mcd"),
    ]:
        if p.exists():
            run(p)
        else:
            print(f"{p} missing")


if __name__ == "__main__":
    main()
