#!/usr/bin/env python3
"""
Helpers for parsing the sliced CComponentDefinition blobs such as
FONTS/components/001_M92A.bin.

Each blob consists of:
    * 28 little-endian int16 header slots
    * N body records delimited by the sentinel 0x8003 (-32765 as int16)
      where each record stores 16 int16 values (four metadata ints plus
      eight coordinate ints).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import struct

SENTINEL = -0x7FFD  # 0x8003
HEADER_LEN = 28
POINT_SCALE = 1.0 / 32768.0


@dataclass(frozen=True)
class ComponentRecord:
    values: tuple[int, ...]

    @property
    def metadata(self) -> tuple[int, int, int, int]:
        return self.values[:4]

    @property
    def point_pairs(self) -> tuple[tuple[int, int], ...]:
        if len(self.values) < 6:
            return tuple()
        coords = []
        upper = len(self.values) - 1
        for idx in range(4, upper, 2):
            if idx + 1 >= len(self.values):
                break
            coords.append((self.values[idx], self.values[idx + 1]))
        return tuple(coords)

    def normalized_points(self) -> tuple[tuple[float, float], ...]:
        return tuple((x * POINT_SCALE, y * POINT_SCALE) for x, y in self.point_pairs)


@dataclass(frozen=True)
class ComponentDefinition:
    label: str
    header: tuple[int, ...]
    records: tuple[ComponentRecord, ...]

    def bounding_box(self) -> tuple[float, float, float, float] | None:
        points: list[tuple[float, float]] = []
        for record in self.records:
            points.extend(record.normalized_points())
        if not points:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def header_floats(self) -> tuple[float, ...]:
        return tuple(value * POINT_SCALE for value in self.header)

    @property
    def record_count_header(self) -> int:
        return self.header[-1] if self.header else 0


def _split_records(values: Sequence[int]) -> Iterator[ComponentRecord]:
    current: list[int] | None = None
    for value in values:
        if value == SENTINEL:
            if current is not None:
                yield ComponentRecord(tuple(current))
            current = []
        else:
            if current is None:
                current = []
            current.append(value)
    if current:
        yield ComponentRecord(tuple(current))


def parse_component_bytes(label: str, blob: bytes) -> ComponentDefinition:
    data = blob[: len(blob) // 2 * 2]
    shorts = struct.unpack("<{}h".format(len(data) // 2), data)
    header = tuple(shorts[:HEADER_LEN])
    body = shorts[HEADER_LEN:]
    records = tuple(_split_records(body))
    return ComponentDefinition(label=label, header=header, records=records)


def parse_component_file(path: Path) -> ComponentDefinition:
    label = path.stem.split("_", 1)[1]
    return parse_component_bytes(label, path.read_bytes())


def iter_component_files(directory: Path) -> Iterable[ComponentDefinition]:
    for path in sorted(directory.glob("*.bin")):
        yield parse_component_file(path)
