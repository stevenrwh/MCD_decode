from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from .entities import ArcEntity, LineEntity

DUP_FINGERPRINT_PLACES = 6
MAX_COORD_MAGNITUDE = 1e5
HELPER_AXIS_TOL = 1e-3


def fuzzy_eq(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


def points_match(p1: Tuple[float, float], p2: Tuple[float, float], tol: float = 1e-6) -> bool:
    return fuzzy_eq(p1[0], p2[0], tol) and fuzzy_eq(p1[1], p2[1], tol)


def point_in_bbox(pt: Tuple[float, float], bbox: Tuple[float, float, float, float], *, tol: float = 0.0) -> bool:
    x, y = pt
    min_x, min_y, max_x, max_y = bbox
    return (min_x - tol) <= x <= (max_x + tol) and (min_y - tol) <= y <= (max_y + tol)


def round_coord(value: float, places: int = DUP_FINGERPRINT_PLACES) -> float:
    return round(value, places)


def record_fingerprint(
    layer: int,
    etype: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Tuple[int, int, float, float, float, float]:
    return (
        layer,
        etype,
        round_coord(x1),
        round_coord(y1),
        round_coord(x2),
        round_coord(y2),
    )


def is_alignment_helper(line: LineEntity) -> bool:
    if line.layer != 3:
        return False
    x_vals = (line.start[0], line.end[0])
    y_vals = (line.start[1], line.end[1])
    if max(abs(x) for x in x_vals) > HELPER_AXIS_TOL:
        return False
    if abs(x_vals[0] - x_vals[1]) > HELPER_AXIS_TOL:
        return False
    if min(abs(y) for y in y_vals) > HELPER_AXIS_TOL:
        return False
    return True


def prune_lines_against_arcs(lines: Sequence[LineEntity], arcs: Sequence[ArcEntity], *, tol: float = 1e-4) -> list[LineEntity]:
    """Drop lines that coincide with arcs to reduce duplicates."""

    if not arcs or not lines:
        return list(lines)

    def point_on_arc(pt: Tuple[float, float], arc: ArcEntity) -> bool:
        cx, cy = arc.center
        r = math.hypot(arc.start[0] - cx, arc.start[1] - cy)
        if r <= 0:
            return False
        tol_r = max(tol, r * 1e-4)
        d = abs(math.hypot(pt[0] - cx, pt[1] - cy) - r)
        if d > tol_r:
            return False
        start_ang = math.atan2(arc.start[1] - cy, arc.start[0] - cx) % math.tau
        end_ang = math.atan2(arc.end[1] - cy, arc.end[0] - cx) % math.tau
        ang = math.atan2(pt[1] - cy, pt[0] - cx) % math.tau
        sweep = (end_ang - start_ang) % math.tau
        rel = (ang - start_ang) % math.tau
        tol_ang = max(1e-3, tol_r / r)
        return rel <= sweep + tol_ang

    # Fast index by layer
    arcs_by_layer: dict[int, list[ArcEntity]] = {}
    for a in arcs:
        arcs_by_layer.setdefault(a.layer, []).append(a)

    pruned: list[LineEntity] = []
    for ln in lines:
        layer_arcs = arcs_by_layer.get(ln.layer, [])
        removed = False
        for arc in layer_arcs:
            if point_on_arc(ln.start, arc) and point_on_arc(ln.end, arc):
                removed = True
                break
        if not removed:
            pruned.append(ln)
    return pruned
