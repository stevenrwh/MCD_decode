#!/usr/bin/env python3
"""
Instrumentation helper to inspect the per-chunk metadata inside Vm component
records and see how they line up with the reference DXF geometry.

Usage:
    python analyze_vm_component_chunks.py \
        --components FONTS/components_Mcalf020 \
        --transform-stats FONTS/Vm_tail_transform_stats.csv \
        --reference-dxf FONTS/Mcalf020_exported_from_monucad.dxf \
        --output vm_chunk_dump.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from font_components import POINT_SCALE, iter_component_files
from mcd_to_dxf import _parse_unexploded_blocks


def _load_transform_stats(path: Path) -> dict[str, Tuple[float, float, float, float]]:
    mapping: dict[str, Tuple[float, float, float, float]] = {}
    import csv

    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                mapping[row["label"]] = (
                    float(row["scale_x"]),
                    float(row["scale_y"]),
                    float(row["translate_x"]),
                    float(row["translate_y"]),
                )
            except (KeyError, ValueError):
                continue
    return mapping


def _apply_transform(
    coords: Sequence[Tuple[float, float]],
    transform: Tuple[float, float, float, float] | None,
) -> List[Tuple[float, float]]:
    if not transform:
        return [(float(x), float(y)) for x, y in coords]
    sx, sy, tx, ty = transform
    return [(x * sx + tx, y * sy + ty) for x, y in coords]


def _chunk_records(values: Sequence[int]) -> Iterable[Tuple[List[int], List[int]]]:
    if len(values) <= 8:
        return []
    body = values[4:-4]
    for idx in range(0, len(body), 12):
        chunk = body[idx : idx + 12]
        if len(chunk) < 12:
            break
        meta = chunk[:4]
        coords = chunk[4:]
        yield meta, coords


def _decode_coords(raw: Sequence[int]) -> List[Tuple[float, float]]:
    usable = len(raw) // 2 * 2
    coords: List[Tuple[float, float]] = []
    for idx in range(0, usable, 2):
        coords.append((raw[idx] * POINT_SCALE, raw[idx + 1] * POINT_SCALE))
    return coords


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump Vm component chunk metadata vs reference geometry.")
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--reference-dxf", type=Path, required=True)
    parser.add_argument("--transform-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("vm_chunk_dump.json"))
    return parser.parse_args()


def _line_key(start: Tuple[float, float], end: Tuple[float, float], tol: float = 1e-3) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    def _round(pt: Tuple[float, float]) -> Tuple[float, float]:
        return (round(pt[0] / tol) * tol, round(pt[1] / tol) * tol)

    s = _round(start)
    e = _round(end)
    return (s, e) if s <= e else (e, s)


def main() -> int:
    args = parse_args()
    transform_map = _load_transform_stats(args.transform_stats)
    reference = _parse_unexploded_blocks(args.reference_dxf)

    payload: dict[str, List[dict]] = {}
    for definition in iter_component_files(args.components):
        transform = transform_map.get(definition.label)
        ref_segments = reference.get(definition.label)
        ref_lines = ref_segments.segments if ref_segments else []
        chunk_rows: List[dict] = []
        ref_keys = {_line_key(seg[0], seg[1]) for seg in ref_lines}
        for record in definition.records:
            for meta, coords in _chunk_records(record.values):
                points = _decode_coords(coords)
                transformed = _apply_transform(points, transform)
                matches = 0
                for idx in range(0, len(transformed), 2):
                    if idx + 1 >= len(transformed):
                        break
                    key = _line_key(transformed[idx], transformed[idx + 1])
                    if key in ref_keys:
                        matches += 1
                chunk_rows.append(
                    {
                        "meta": meta,
                        "raw_points": [(float(x), float(y)) for x, y in points],
                        "transformed_points": transformed,
                        "matched_segments": matches,
                    }
                )
        payload[definition.label] = chunk_rows

    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[+] Chunk dump written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
