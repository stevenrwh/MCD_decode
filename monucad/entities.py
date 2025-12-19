from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class LineEntity:
    layer: int
    start: Tuple[float, float]
    end: Tuple[float, float]


@dataclass(frozen=True)
class ArcEntity:
    layer: int
    center: Tuple[float, float]
    start: Tuple[float, float]
    end: Tuple[float, float]


@dataclass(frozen=True)
class CircleEntity:
    layer: int
    center: Tuple[float, float]
    radius: float


@dataclass(frozen=True)
class DuplicateRecord:
    offset: int
    original_offset: int
    layer: int
    etype: int
    start: Tuple[float, float]
    end: Tuple[float, float]


@dataclass(frozen=True)
class InsertEntity:
    name: str
    position: Tuple[float, float]
    rotation: float
    scale: Tuple[float, float]
    layer: int = 0
