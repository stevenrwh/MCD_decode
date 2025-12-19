"""
Core MonuCAD parsing utilities split into modules for reuse.
"""

from .deflate_io import DEFAULT_MIN_PAYLOAD, brute_force_deflate, collect_deflate_streams
from .entities import ArcEntity, CircleEntity, DuplicateRecord, InsertEntity, LineEntity

__all__ = [
    "DEFAULT_MIN_PAYLOAD",
    "brute_force_deflate",
    "collect_deflate_streams",
    "LineEntity",
    "ArcEntity",
    "CircleEntity",
    "DuplicateRecord",
    "InsertEntity",
]
