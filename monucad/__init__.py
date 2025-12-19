"""
Core MonuCAD parsing utilities split into modules for reuse.
"""

from .deflate_io import DEFAULT_MIN_PAYLOAD, brute_force_deflate, collect_deflate_streams
from .entities import ArcEntity, CircleEntity, DuplicateRecord, InsertEntity, LineEntity
from .logging import ArcHelperLogger, log_duplicate_records
from .fonts import FontDefinition, FontManager, Glyph, TextEntity, GLYPH_COORD_SCALE, PRINTABLE_ASCII
from .geometry import (
    DUP_FINGERPRINT_PLACES,
    HELPER_AXIS_TOL,
    MAX_COORD_MAGNITUDE,
    fuzzy_eq,
    is_alignment_helper,
    point_in_bbox,
    points_match,
    prune_lines_against_arcs,
    record_fingerprint,
    round_coord,
)

__all__ = [
    "DEFAULT_MIN_PAYLOAD",
    "brute_force_deflate",
    "collect_deflate_streams",
    "LineEntity",
    "ArcEntity",
    "CircleEntity",
    "DuplicateRecord",
    "InsertEntity",
    "ArcHelperLogger",
    "log_duplicate_records",
    "FontDefinition",
    "FontManager",
    "Glyph",
    "TextEntity",
    "GLYPH_COORD_SCALE",
    "PRINTABLE_ASCII",
    "MAX_COORD_MAGNITUDE",
    "HELPER_AXIS_TOL",
    "DUP_FINGERPRINT_PLACES",
    "fuzzy_eq",
    "points_match",
    "point_in_bbox",
    "is_alignment_helper",
    "prune_lines_against_arcs",
    "record_fingerprint",
    "round_coord",
]
