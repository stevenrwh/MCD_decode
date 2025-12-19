from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import List, Sequence, Tuple

from component_parser import (
    CirclePrimitive,
    extract_circle_primitives,
    iter_component_definitions,
    iter_component_placements,
    iter_label_chunks,
)
from font_components import POINT_SCALE as COMPONENT_POINT_SCALE, SENTINEL as COMPONENT_SENTINEL
from font_components import parse_component_bytes
from extract_font_components import extract_components
from placement_parser import extract_glyph_records, iter_placement_trailers
from .entities import ArcEntity, CircleEntity, InsertEntity, LineEntity
from .fonts import Glyph
from .geometry import MAX_COORD_MAGNITUDE, circle_from_points as _circle_from_points, point_in_bbox as _point_in_bbox, points_match as _points_match, prune_lines_against_arcs as _prune_lines_against_arcs


def write_dxf(
    lines: Sequence[LineEntity],
    arcs: Sequence[ArcEntity],
    circles: Sequence[CircleEntity],
    destination: Path,
    inserts: Sequence[InsertEntity] | None = None,
) -> None:
    """
    Emit a bare-bones DXF file that places every entity on its original layer
    (named MCD_Layer_<id>) and keeps Z at 0.
    """

    def emit(code: str, value: str) -> str:
        return f"{code}\n{value}\n"

    chunks: list[str] = []

    # Optional placeholder BLOCK definitions for any inserts we plan to emit.
    if inserts:
        block_names = {ins.name for ins in inserts}
        chunks.append(emit("0", "SECTION"))
        chunks.append(emit("2", "BLOCKS"))
        for name in sorted(block_names):
            chunks.append(emit("0", "BLOCK"))
            chunks.append(emit("8", "0"))
            chunks.append(emit("2", name))
            chunks.append(emit("70", "0"))
            chunks.append(emit("10", "0.0"))
            chunks.append(emit("20", "0.0"))
            chunks.append(emit("30", "0.0"))
            chunks.append(emit("3", name))
            chunks.append(emit("1", name))
            chunks.append(emit("0", "ENDBLK"))
        chunks.append(emit("0", "ENDSEC"))

    chunks.append(emit("0", "SECTION"))
    chunks.append(emit("2", "ENTITIES"))

    for line in lines:
        layer_name = f"MCD_Layer_{line.layer}"
        chunks.append(emit("0", "LINE"))
        chunks.append(emit("8", layer_name))
        chunks.append(emit("10", f"{line.start[0]:.6f}"))
        chunks.append(emit("20", f"{line.start[1]:.6f}"))
        chunks.append(emit("30", "0.0"))
        chunks.append(emit("11", f"{line.end[0]:.6f}"))
        chunks.append(emit("21", f"{line.end[1]:.6f}"))
        chunks.append(emit("31", "0.0"))

    for arc in arcs:
        layer_name = f"MCD_Layer_{arc.layer}"
        radius = math.hypot(arc.start[0] - arc.center[0], arc.start[1] - arc.center[1])
        if radius < 1e-9:
            continue

        def _angle(pt: Tuple[float, float]) -> float:
            return math.degrees(math.atan2(pt[1] - arc.center[1], pt[0] - arc.center[0])) % 360.0

        start_ang = _angle(arc.start)
        end_ang = _angle(arc.end)

        chunks.append(emit("0", "ARC"))
        chunks.append(emit("8", layer_name))
        chunks.append(emit("10", f"{arc.center[0]:.6f}"))
        chunks.append(emit("20", f"{arc.center[1]:.6f}"))
        chunks.append(emit("40", f"{radius:.6f}"))
        chunks.append(emit("50", f"{start_ang:.6f}"))
        chunks.append(emit("51", f"{end_ang:.6f}"))

    for circle in circles:
        layer_name = f"MCD_Layer_{circle.layer}"
        if circle.radius < 1e-9:
            continue
        chunks.append(emit("0", "CIRCLE"))
        chunks.append(emit("8", layer_name))
        chunks.append(emit("10", f"{circle.center[0]:.6f}"))
        chunks.append(emit("20", f"{circle.center[1]:.6f}"))
        chunks.append(emit("40", f"{circle.radius:.6f}"))

    if inserts:
        for ins in inserts:
            layer_name = f"MCD_Layer_{ins.layer}"
            chunks.append(emit("0", "INSERT"))
            chunks.append(emit("8", layer_name))
            chunks.append(emit("2", ins.name))
            chunks.append(emit("10", f"{ins.position[0]:.6f}"))
            chunks.append(emit("20", f"{ins.position[1]:.6f}"))
            chunks.append(emit("30", "0.0"))
            chunks.append(emit("41", f"{ins.scale[0]:.6f}"))
            chunks.append(emit("42", f"{ins.scale[1]:.6f}"))
            chunks.append(emit("43", "1.0"))
            if abs(ins.rotation) > 1e-9:
                chunks.append(emit("50", f"{ins.rotation:.6f}"))

    chunks.append(emit("0", "ENDSEC"))
    chunks.append(emit("0", "EOF"))
    destination.write_text("".join(chunks), encoding="ascii")


def _collect_component_circles(
    payload: bytes,
) -> Tuple[List[CircleEntity], List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    definitions = {definition.component_id: definition for definition in iter_component_definitions(payload)}
    if not definitions:
        return [], []

    circle_cache: dict[int, List[CirclePrimitive]] = {}
    helper_cache: dict[int, List[Tuple[Tuple[float, float], Tuple[float, float]]]] = {}

    for component_id, definition in definitions.items():
        circles = extract_circle_primitives(definition)
        if circles:
            circle_cache[component_id] = circles
            helper_cache[component_id] = [(circle.center, circle.rim) for circle in circles]

    if not circle_cache:
        return [], []

    placements = list(iter_component_placements(payload))
    entities: List[CircleEntity] = []
    helper_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []

    def _merge(component_id: int) -> None:
        primitives = circle_cache.get(component_id)
        if not primitives:
            return
        for primitive in primitives:
            entities.append(
                CircleEntity(
                    layer=0,
                    center=primitive.center,
                    radius=primitive.radius,
                )
            )
        helper_segments.extend(helper_cache.get(component_id, []))

    if placements:
        for placement in placements:
            _merge(placement.component_id)

    if not entities:
        for component_id in circle_cache.keys():
            _merge(component_id)

    return entities, helper_segments


def _segments_from_inline_floats(
    chunk: bytes,
    *,
    stride: int = 16,
    limit: float = 2.0,
    min_segments: int = 10,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Greedy float decoder for inline blobs that pack 4-float stroke tuples."""

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    seen: set[tuple[float, float, float, float]] = set()
    for offset in range(0, len(chunk) - 16 + 1, stride):
        try:
            x1, y1, x2, y2 = struct.unpack_from("<4f", chunk, offset)
        except struct.error:
            break
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        if any(abs(v) > limit for v in (x1, y1, x2, y2)):
            continue
        if x1 == x2 and y1 == y2:
            continue
        key = (round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6))
        if key in seen:
            continue
        seen.add(key)
        segments.append(((x1, y1), (x2, y2)))
    if len(segments) < min_segments:
        return []
    return segments


def _parse_inline_float_records(
    chunk: bytes,
    *,
    coord_limit: float = 10.0,
) -> tuple[
    tuple[float, float, float, float] | None,
    list[LineEntity],
    list[tuple[int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]],
]:
    """
    Parse legacy inline component blobs that store fixed-size float32 records.

    Observed layout (Monu-CAD v9 era, exploded font components):
      - 2 bytes: 0x01 0x80
      - 4*float32 bbox: min_x, min_y, max_x, max_y
      - 4 bytes: unknown
      - records start at offset 22

    Record formats:
      - Line (etype=2): 34 bytes (layer,u32)(etype,u32)(x1,y1,x2,y2 floats)(u1,u2 u32)(tail u16)
      - Arc  (etype=3): 42 bytes (layer,u32)(etype,u32)(start,mid,end as 6 floats)(u1,u2 u32)(tail u16)
    """

    if len(chunk) < 22 or chunk[:2] != b"\x01\x80":
        return None, [], []

    try:
        min_x, min_y, max_x, max_y = struct.unpack_from("<4f", chunk, 2)
    except struct.error:
        return None, [], []
    if not all(math.isfinite(v) for v in (min_x, min_y, max_x, max_y)):
        return None, [], []
    if any(abs(v) > coord_limit for v in (min_x, min_y, max_x, max_y)):
        return None, [], []
    if max_x < min_x or max_y < min_y:
        return None, [], []
    bbox = (float(min_x), float(min_y), float(max_x), float(max_y))
    width = max_x - min_x
    height = max_y - min_y
    tol = max(1e-4, float(max(width, height)) * 0.02)

    offset = 22
    lines: list[LineEntity] = []
    arcs: list[tuple[int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]] = []

    while offset + 8 <= len(chunk):
        try:
            layer, etype = struct.unpack_from("<II", chunk, offset)
        except struct.error:
            break
        if layer > 2048:
            break
        if etype not in (2, 3):
            break

        if etype == 2:
            coord_off = offset + 8
            if coord_off + 16 > len(chunk):
                break
            try:
                x1, y1, x2, y2 = struct.unpack_from("<4f", chunk, coord_off)
            except struct.error:
                break
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                offset += 34
                continue
            if any(abs(v) > coord_limit for v in (x1, y1, x2, y2)):
                offset += 34
                continue
            p1 = (float(x1), float(y1))
            p2 = (float(x2), float(y2))
            if p1 != p2 and _point_in_bbox(p1, bbox, tol=tol) and _point_in_bbox(p2, bbox, tol=tol):
                lines.append(LineEntity(layer=int(layer), start=p1, end=p2))
            # Advance by full stride when possible; tolerate a truncated final record.
            if offset + 34 <= len(chunk):
                offset += 34
            else:
                break
            continue

        # etype == 3 (arc)
        coord_off = offset + 8
        if coord_off + 24 > len(chunk):
            break
        try:
            x0, y0, x1, y1, x2, y2 = struct.unpack_from("<6f", chunk, coord_off)
        except struct.error:
            break
        if not all(math.isfinite(v) for v in (x0, y0, x1, y1, x2, y2)):
            offset += 42
            continue
        if any(abs(v) > coord_limit for v in (x0, y0, x1, y1, x2, y2)):
            offset += 42
            continue
        start = (float(x0), float(y0))
        mid = (float(x1), float(y1))
        end = (float(x2), float(y2))
        if (
            _point_in_bbox(start, bbox, tol=tol)
            and _point_in_bbox(mid, bbox, tol=tol)
            and _point_in_bbox(end, bbox, tol=tol)
        ):
            arcs.append((int(layer), (start, mid, end)))
        if offset + 42 <= len(chunk):
            offset += 42
        else:
            break

    return bbox, lines, arcs


def _extract_bbox_from_inline_chunk(
    chunk: bytes,
    *,
    coord_limit: float = 10.0,
) -> tuple[float, float, float, float] | None:
    """Best-effort bbox extractor for inline glyph blobs."""

    if len(chunk) >= 22 and chunk[:2] == b"\x01\x80":
        try:
            min_x, min_y, max_x, max_y = struct.unpack_from("<4f", chunk, 2)
        except struct.error:
            return None
        if not all(math.isfinite(v) for v in (min_x, min_y, max_x, max_y)):
            return None
        if any(abs(v) > coord_limit for v in (min_x, min_y, max_x, max_y)):
            return None
        if max_x < min_x or max_y < min_y:
            return None
        return (float(min_x), float(min_y), float(max_x), float(max_y))

    marker = b"CComponentDefinition"
    marker_pos = chunk.find(marker)
    if marker_pos != -1:
        bbox_off = marker_pos + len(marker)
        if bbox_off + 16 <= len(chunk):
            try:
                min_x, min_y, max_x, max_y = struct.unpack_from("<4f", chunk, bbox_off)
            except struct.error:
                min_x = min_y = max_x = max_y = 0.0
            if (
                all(math.isfinite(v) for v in (min_x, min_y, max_x, max_y))
                and max_x >= min_x
                and max_y >= min_y
                and max(abs(v) for v in (min_x, min_y, max_x, max_y)) <= coord_limit * 2
            ):
                return (float(min_x), float(min_y), float(max_x), float(max_y))

    return None


def _scan_inline_double_records(
    chunk: bytes,
    bbox: tuple[float, float, float, float],
    *,
    coord_limit: float = 10.0,
) -> tuple[
    list[LineEntity],
    list[tuple[int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]],
]:
    """
    Inline glyph blobs sometimes mirror the float32 layout but store float64 coords and
    lack reliable record boundaries. This scans candidate (layer, etype) headers and
    validates coordinates against the supplied component bbox.
    """

    min_x, min_y, max_x, max_y = bbox
    width = max_x - min_x
    height = max_y - min_y
    tol = max(1e-4, float(max(width, height)) * 0.02)

    def scan(
        *,
        start_offset: int,
        step: int,
        lines: list[LineEntity],
        arcs: list[tuple[int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]],
        seen_lines: set[tuple[int, float, float, float, float]],
        seen_arcs: set[tuple[int, float, float, float, float, float, float]],
    ) -> None:
        offset = start_offset
        limit = len(chunk)
        while offset + 8 <= limit:
            try:
                layer, etype = struct.unpack_from("<II", chunk, offset)
            except struct.error:
                break
            if layer <= 2048 and etype in (2, 3):
                if etype == 2 and offset + 8 + 32 <= limit:
                    try:
                        x1, y1, x2, y2 = struct.unpack_from("<4d", chunk, offset + 8)
                    except struct.error:
                        offset += step
                        continue
                    if (
                        all(math.isfinite(v) for v in (x1, y1, x2, y2))
                        and max(abs(v) for v in (x1, y1, x2, y2)) <= coord_limit
                    ):
                        p1 = (float(x1), float(y1))
                        p2 = (float(x2), float(y2))
                        if (
                            p1 != p2
                            and _point_in_bbox(p1, bbox, tol=tol)
                            and _point_in_bbox(p2, bbox, tol=tol)
                        ):
                            key = (
                                int(layer),
                                round(p1[0], 6),
                                round(p1[1], 6),
                                round(p2[0], 6),
                                round(p2[1], 6),
                            )
                            if key not in seen_lines:
                                seen_lines.add(key)
                                lines.append(LineEntity(layer=int(layer), start=p1, end=p2))
                elif etype == 3 and offset + 8 + 48 <= limit:
                    try:
                        x0, y0, x1, y1, x2, y2 = struct.unpack_from("<6d", chunk, offset + 8)
                    except struct.error:
                        offset += step
                        continue
                    if (
                        all(math.isfinite(v) for v in (x0, y0, x1, y1, x2, y2))
                        and max(abs(v) for v in (x0, y0, x1, y1, x2, y2)) <= coord_limit
                    ):
                        p0 = (float(x0), float(y0))
                        p1 = (float(x1), float(y1))
                        p2 = (float(x2), float(y2))
                        if (
                            _point_in_bbox(p0, bbox, tol=tol)
                            and _point_in_bbox(p1, bbox, tol=tol)
                            and _point_in_bbox(p2, bbox, tol=tol)
                        ):
                            key = (
                                int(layer),
                                round(p0[0], 6),
                                round(p0[1], 6),
                                round(p1[0], 6),
                                round(p1[1], 6),
                                round(p2[0], 6),
                                round(p2[1], 6),
                            )
                            if key not in seen_arcs:
                                seen_arcs.add(key)
                                arcs.append((int(layer), (p0, p1, p2)))
            offset += step

    lines: list[LineEntity] = []
    arcs: list[tuple[int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]] = []
    seen_lines: set[tuple[int, float, float, float, float]] = set()
    seen_arcs: set[tuple[int, float, float, float, float, float, float]] = set()

    # Records can start on either byte parity (observed in MCPro9-resaved files),
    # so scan both even and odd offsets.
    scan(
        start_offset=0,
        step=2,
        lines=lines,
        arcs=arcs,
        seen_lines=seen_lines,
        seen_arcs=seen_arcs,
    )
    scan(
        start_offset=1,
        step=2,
        lines=lines,
        arcs=arcs,
        seen_lines=seen_lines,
        seen_arcs=seen_arcs,
    )
    if not lines and not arcs:
        scan(
            start_offset=0,
            step=1,
            lines=lines,
            arcs=arcs,
            seen_lines=seen_lines,
            seen_arcs=seen_arcs,
        )

    return lines, arcs


def _parse_inline_labeled_float_records(
    chunk: bytes,
    *,
    coord_limit: float = 5.0,
    min_prims: int = 12,
) -> tuple[
    tuple[float, float, float, float] | None,
    list[LineEntity],
    list[tuple[int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]],
]:
    """
    Some legacy inline components store float-record geometry inside TLV-ish string
    labels like 'CLine' and 'CArc' without the leading 0x01 0x80 + bbox wrapper.
    """

    bbox: tuple[float, float, float, float] | None = None
    marker = b"CComponentDefinition"
    marker_pos = chunk.find(marker)
    if marker_pos != -1:
        bbox_off = marker_pos + len(marker)
        if bbox_off + 16 <= len(chunk):
            try:
                min_x, min_y, max_x, max_y = struct.unpack_from("<4f", chunk, bbox_off)
            except struct.error:
                min_x = min_y = max_x = max_y = 0.0
            if (
                all(math.isfinite(v) for v in (min_x, min_y, max_x, max_y))
                and max_x >= min_x
                and max_y >= min_y
                and max(abs(v) for v in (min_x, min_y, max_x, max_y)) <= coord_limit * 2
            ):
                bbox = (float(min_x), float(min_y), float(max_x), float(max_y))

    labels = [b"CLine", b"CArc"]
    hits: list[tuple[int, bytes, int]] = []
    for label in labels:
        pos = chunk.find(label)
        if pos == -1:
            continue
        header_off = pos - 8
        if header_off < 0:
            continue
        try:
            _, dtype, count, size = struct.unpack_from("<HHHH", chunk, header_off)
        except struct.error:
            continue
        if dtype != 0xFFFF or size != len(label) or count not in (0, 1):
            continue
        hits.append((pos, label, header_off))

    if not hits:
        return None, [], []

    hits.sort(key=lambda item: item[0])
    bounds: list[tuple[int, int, bytes]] = []
    for idx, (pos, label, header_off) in enumerate(hits):
        start = pos + len(label)
        end = len(chunk)
        if idx + 1 < len(hits):
            next_header = hits[idx + 1][2]
            end = max(start, min(end, next_header))
        bounds.append((start, end, label))

    lines: list[LineEntity] = []
    arcs: list[tuple[int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]] = []

    tol = 0.0
    if bbox is not None:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        tol = max(1e-4, max(w, h) * 0.02)

    def _accept_pt(pt: tuple[float, float]) -> bool:
        if not all(math.isfinite(v) for v in pt):
            return False
        if max(abs(pt[0]), abs(pt[1])) > coord_limit:
            return False
        if bbox is not None and not _point_in_bbox(pt, bbox, tol=tol):
            return False
        return True

    allowed_tails = {0x8008, 0x8003, 0x0008, 0x0000}
    seen_lines: set[tuple[int, float, float, float, float]] = set()
    seen_arcs: set[tuple[int, float, float, float, float, float, float]] = set()

    def _add_line(layer: int, p1: tuple[float, float], p2: tuple[float, float]) -> None:
        if p1 == p2:
            return
        if not (_accept_pt(p1) and _accept_pt(p2)):
            return
        key = (
            int(layer),
            round(p1[0], 6),
            round(p1[1], 6),
            round(p2[0], 6),
            round(p2[1], 6),
        )
        if key in seen_lines:
            return
        seen_lines.add(key)
        lines.append(LineEntity(layer=int(layer), start=p1, end=p2))

    def _add_arc(layer: int, p0: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]) -> None:
        if not (_accept_pt(p0) and _accept_pt(p1) and _accept_pt(p2)):
            return
        key = (
            int(layer),
            round(p0[0], 6),
            round(p0[1], 6),
            round(p1[0], 6),
            round(p1[1], 6),
            round(p2[0], 6),
            round(p2[1], 6),
        )
        if key in seen_arcs:
            return
        seen_arcs.add(key)
        arcs.append((int(layer), (p0, p1, p2)))

    for start, end, label in bounds:
        offset = start
        if label == b"CLine":
            stride = 34
            while offset + 8 + 16 <= end:
                try:
                    layer, etype = struct.unpack_from("<II", chunk, offset)
                    x1, y1, x2, y2 = struct.unpack_from("<4f", chunk, offset + 8)
                except struct.error:
                    break
                if etype != 2 or layer > 2048:
                    break
                _add_line(int(layer), (float(x1), float(y1)), (float(x2), float(y2)))
                if offset + stride <= end:
                    offset += stride
                else:
                    break
        elif label == b"CArc":
            stride = 42
            while offset + 8 + 24 <= end:
                try:
                    layer, etype = struct.unpack_from("<II", chunk, offset)
                    x0, y0, x1, y1, x2, y2 = struct.unpack_from("<6f", chunk, offset + 8)
                except struct.error:
                    break
                if etype != 3 or layer > 2048:
                    break
                _add_arc(
                    int(layer),
                    (float(x0), float(y0)),
                    (float(x1), float(y1)),
                    (float(x2), float(y2)),
                )
                if offset + stride <= end:
                    offset += stride
                else:
                    break

        # Some payloads interleave line+arc records and padding inside the same
        # labeled blob. Scan the full range for any remaining records.
        scan_offset = start
        while scan_offset + 8 <= end:
            try:
                layer, etype = struct.unpack_from("<II", chunk, scan_offset)
            except struct.error:
                break
            if layer > 2048:
                scan_offset += 2
                continue
            if etype == 2 and scan_offset + 8 + 16 <= end:
                try:
                    x1, y1, x2, y2 = struct.unpack_from("<4f", chunk, scan_offset + 8)
                except struct.error:
                    scan_offset += 2
                    continue
                if any(abs(v) > coord_limit for v in (x1, y1, x2, y2)):
                    scan_offset += 2
                    continue
                if scan_offset + 34 <= end:
                    tail = struct.unpack_from("<H", chunk, scan_offset + 32)[0]
                    if tail not in allowed_tails:
                        scan_offset += 2
                        continue
                _add_line(int(layer), (float(x1), float(y1)), (float(x2), float(y2)))
            elif etype == 3 and scan_offset + 8 + 24 <= end:
                try:
                    x0, y0, x1, y1, x2, y2 = struct.unpack_from("<6f", chunk, scan_offset + 8)
                except struct.error:
                    scan_offset += 2
                    continue
                if any(abs(v) > coord_limit for v in (x0, y0, x1, y1, x2, y2)):
                    scan_offset += 2
                    continue
                if scan_offset + 42 <= end:
                    tail = struct.unpack_from("<H", chunk, scan_offset + 40)[0]
                    if tail not in allowed_tails:
                        scan_offset += 2
                        continue
                _add_arc(
                    int(layer),
                    (float(x0), float(y0)),
                    (float(x1), float(y1)),
                    (float(x2), float(y2)),
                )
            scan_offset += 2

    if len(lines) + len(arcs) < min_prims:
        return bbox, [], []
    return bbox, lines, arcs


def _decode_glyph_chunk(
    label: str, chunk: bytes
) -> tuple[
    list[tuple[tuple[float, float], tuple[float, float]]],
    list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]],
    tuple[float, float, float, float],
]:
    try:
        definition = parse_component_bytes(label, chunk)
    except Exception:
        return [], [], (0.0, 0.0, 0.0, 0.0)
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    arcs: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
    for record in definition.records:
        etype = record.metadata[2] if len(record.metadata) >= 3 else None
        pts = list(record.normalized_points())
        if etype == 3 and len(pts) >= 3:
            start, guide, end = pts[0], pts[1], pts[2]
            arcs.append((start, guide, end))
            continue
        if len(pts) < 2:
            continue
        if len(pts) % 2 != 0:
            pts = pts[:-1]
        for idx in range(0, len(pts), 2):
            p1, p2 = pts[idx], pts[idx + 1]
            if p1 == p2:
                continue
            segments.append((p1, p2))

    # Inline / embedded component blobs frequently include marker segments like
    # (0,0)->(1/32768,0) and long rays to the origin used as stroke delimiters.
    # Those poison DXF output (starburst artifacts) so we strip them here.
    if segments:
        min_len = 1e-4  # glyph-space (~1e-5 after scaling) marker floor
        origin_tol = 1e-6
        origin_count = sum(
            1
            for p1, p2 in segments
            if (abs(p1[0]) <= origin_tol and abs(p1[1]) <= origin_tol)
            or (abs(p2[0]) <= origin_tol and abs(p2[1]) <= origin_tol)
        )
        drop_origin = origin_count >= max(8, int(len(segments) * 0.08))
        cleaned: list[tuple[tuple[float, float], tuple[float, float]]] = []
        for p1, p2 in segments:
            if math.hypot(p1[0] - p2[0], p1[1] - p2[1]) < min_len:
                continue
            if drop_origin and (
                (abs(p1[0]) <= origin_tol and abs(p1[1]) <= origin_tol)
                or (abs(p2[0]) <= origin_tol and abs(p2[1]) <= origin_tol)
            ):
                continue
            cleaned.append((p1, p2))
        segments = cleaned

    xs = [p for seg in segments for p in (seg[0][0], seg[1][0])]
    ys = [p for seg in segments for p in (seg[0][1], seg[1][1])]
    for start, guide, end in arcs:
        xs.extend([start[0], guide[0], end[0]])
        ys.extend([start[1], guide[1], end[1]])
    bbox = (0.0, 0.0, 0.0, 0.0) if not xs or not ys else (min(xs), min(ys), max(xs), max(ys))
    return segments, arcs, bbox


def _load_payload_glyph_components(
    payload: bytes,
) -> tuple[dict[str, Glyph], dict[str, list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]]]:
    """Load glyph components directly from the payload (TLV font slices)."""

    glyphs: dict[str, Glyph] = {}
    arc_map: dict[str, list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]] = {}
    try:
        components = list(extract_components(payload, pad=32))
    except Exception:
        return glyphs, arc_map

    # Some payloads contain multiple occurrences of the same label (e.g. label
    # strings repeated inside placement metadata). Parse every candidate and
    # keep the most plausible one (largest bbox area, then most primitives).
    best: dict[str, tuple[float, int, list[tuple[tuple[float, float], tuple[float, float]]], list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]], tuple[float, float, float, float]]] = {}
    for label, _, _, chunk in components:
        segments, arcs, bbox = _decode_glyph_chunk(label, chunk)
        if not segments and not arcs:
            continue
        min_x, min_y, max_x, max_y = bbox
        width = max(0.0, max_x - min_x)
        height = max(0.0, max_y - min_y)
        area = width * height
        prim_count = len(segments) + len(arcs)
        prev = best.get(label)
        if prev is None or (area, prim_count) > (prev[0], prev[1]):
            best[label] = (area, prim_count, segments, arcs, bbox)

    for label, (_, _, segments, arcs, bbox) in best.items():
        min_x, min_y, max_x, max_y = bbox
        advance = max(max_x - min_x, 0.0)
        glyphs[label] = Glyph(
            label=label,
            segments=segments,
            bounds=bbox,
            advance=advance,
            baseline=min_y,
        )
        if arcs:
            arc_map[label] = arcs
    return glyphs, arc_map


def _segments_from_short_chunk(chunk: bytes) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    try:
        definition = parse_component_bytes("TLV", chunk)
    except Exception:
        return []
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for record in definition.records:
        points = record.normalized_points()
        if len(points) < 2:
            continue
        if len(points) % 2 != 0:
            points = points[:-1]
        for idx in range(0, len(points), 2):
            p1 = points[idx]
            p2 = points[idx + 1]
            segments.append((p1, p2))
    return segments


def _extract_short_component_lines(payload: bytes) -> List[LineEntity]:
    short_lines: List[LineEntity] = []
    for definition in iter_component_definitions(payload):
        for label, chunk in iter_label_chunks(payload, definition):
            if label not in {"CLine", "CArc", "CCircle"}:
                continue
            segments = _segments_from_short_chunk(chunk)
            for start, end in segments:
                if not _points_match(start, end):
                    short_lines.append(LineEntity(layer=0, start=start, end=end))
    return short_lines


def _matches_component_helper(
    line: LineEntity,
    helpers: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
) -> bool:
    for center, rim in helpers:
        if _points_match(line.start, center) and _points_match(line.end, rim):
            return True
        if _points_match(line.start, rim) and _points_match(line.end, center):
            return True
    return False


def _looks_like_coordinate(value: float) -> bool:
    return math.isfinite(value) and abs(value) <= MAX_COORD_MAGNITUDE


def _parse_component_place_entries(payload: bytes) -> tuple[float | None, list[tuple[str, tuple[float, float, float, float, float]]], int]:
    marker = b"CComponentPlace"
    idx = payload.find(marker)
    if idx == -1:
        return None, [], -1
    ptr = idx + len(marker)
    try:
        struct.unpack_from("<II", payload, ptr)
        ptr += 8
        tx0, ty0, rot0, sx0, sy0 = struct.unpack_from("<5f", payload, ptr)
        ptr += 20
    except struct.error:
        return None, [], -1

    entries: list[tuple[str, tuple[float, float, float, float, float]]] = []
    current = (tx0, ty0, rot0, sx0, sy0)
    while ptr + 1 < len(payload):
        name_len = payload[ptr]
        if not (1 <= name_len <= 8):
            break
        name_bytes = payload[ptr + 1 : ptr + 1 + name_len]
        try:
            name = name_bytes.decode("ascii")
        except UnicodeDecodeError:
            break
        ptr += 1 + name_len
        if payload[ptr:ptr + 1] == b"_":
            ptr += 1
        if ptr + 9 + 20 > len(payload):
            entries.append((name, current))
            break
        ptr += 9
        try:
            next_transform = struct.unpack_from("<5f", payload, ptr)
        except struct.error:
            entries.append((name, current))
            break
        ptr += 20
        entries.append((name, current))
        current = tuple(float(v) for v in next_transform)

    return None, entries, idx


def _component_segments(definition) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for record in definition.records:
        points = record.normalized_points()
        if len(points) < 2:
            continue
        if len(points) % 2 != 0:
            points = points[:-1]
        for idx in range(0, len(points), 2):
            segments.append((points[idx], points[idx + 1]))
    return segments


def _segments_from_inline_chunk(chunk: bytes) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    if len(chunk) < 16:
        return []
    shorts_len = len(chunk) // 2
    shorts = struct.unpack("<{}h".format(shorts_len), chunk[: shorts_len * 2])
    sentinel = -0x7FFD
    records: list[tuple[int, ...]] = []
    current: list[int] = []
    for value in shorts:
        if value == sentinel:
            if current:
                records.append(tuple(current))
                current = []
            continue
        current.append(value)
    if current:
        records.append(tuple(current))
    if records:
        meta = records[0][:4]
        if any(abs(v) > 1024 for v in meta) or (len(meta) >= 3 and meta[2] not in (0, 2)):
            records = records[1:]

    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for values in records:
        if len(values) < 6:
            continue
        coords = values[4:]
        if len(coords) % 2 != 0:
            coords = coords[:-1]
        points = [(coords[i] * COMPONENT_POINT_SCALE, coords[i + 1] * COMPONENT_POINT_SCALE) for i in range(0, len(coords), 2)]
        if len(points) < 2:
            continue
        if len(points) % 2 != 0:
            points = points[:-1]
        for idx in range(0, len(points), 2):
            segments.append((points[idx], points[idx + 1]))
    return segments


def _collect_glyph_placements(payload: bytes) -> tuple[list[tuple[str, tuple[float, float, float, float, float]]], int]:
    width_hint, placements, place_offset = _parse_component_place_entries(payload)
    if placements:
        return placements, place_offset
    trailer_placements: list[tuple[str, tuple[float, float, float, float, float]]] = []
    trailers = list(iter_placement_trailers(payload))
    for trailer in trailers:
        for record in extract_glyph_records(trailer.payload):
            if len(record.values) != 5:
                continue
            tx, ty, rot, sx, sy = record.values
            trailer_placements.append((record.label, (tx, ty, rot, sx, sy)))
    return trailer_placements, -1


def _parse_carc_arcs(
    payload: bytes,
    labels: Sequence[str],
    segments: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]],
) -> dict[str, list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]]:
    arc_map: dict[str, list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]] = {}
    chunk = None
    for d in iter_component_definitions(payload):
        for lbl, ch in iter_label_chunks(payload, d):
            if lbl == "CArc":
                chunk = ch
                break
    if not chunk:
        return arc_map

    marker1 = [0, 0, 1, 0]
    marker2 = [-32760, 0, 0, 3]
    try:
        shorts = struct.unpack("<{}h".format(len(chunk) // 2), chunk)
    except struct.error:
        return arc_map
    recs: list[list[int]] = []
    cur: list[int] = []
    for s in shorts:
        if s == COMPONENT_SENTINEL:
            if cur:
                recs.append(cur)
                cur = []
        else:
            cur.append(s)
    if cur:
        recs.append(cur)

    triples: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
    for rec in recs:
        if len(rec) < 10:
            continue
        data = rec[4:]
        cleaned: list[int] = []
        i = 0
        while i < len(data):
            if data[i : i + 4] == marker1:
                i += 4
                continue
            if data[i : i + 4] == marker2:
                i += 4
                continue
            cleaned.append(data[i])
            i += 1
        for j in range(0, len(cleaned) - 5, 6):
            sub = cleaned[j : j + 6]
            if len(sub) < 6:
                continue
            pts = [
                (sub[k] * COMPONENT_POINT_SCALE, sub[k + 1] * COMPONENT_POINT_SCALE)
                for k in range(0, 6, 2)
            ]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            size = max(bbox[2] - bbox[0], bbox[3] - bbox[1])
            if size <= 3.0:
                triples.append(tuple(pts))

    if not triples:
        return arc_map

    inline_bboxes: dict[str, tuple[float, float, float, float]] = {}
    for lbl in labels:
        segs = segments.get(lbl)
        if not segs:
            continue
        xs = [p for seg in segs for p in (seg[0][0], seg[1][0])]
        ys = [p for seg in segs for p in (seg[0][1], seg[1][1])]
        inline_bboxes[lbl] = (min(xs), min(ys), max(xs), max(ys))

    placements, _ = _collect_glyph_placements(payload)
    if placements:
        ws_bboxes: dict[str, tuple[float, float, float, float]] = {}
        for lbl, vals in placements:
            tx, ty, rot, sx, sy = vals
            segs = segments.get(lbl, [])
            if not segs:
                continue
            xs: list[float] = []
            ys: list[float] = []
            for p1, p2 in segs:
                xs.extend([p1[0] * sx + tx, p2[0] * sx + tx])
                ys.extend([p1[1] * sy + ty, p2[1] * sy + ty])
            bbox = (min(xs), min(ys), max(xs), max(ys))
            prev = ws_bboxes.get(lbl)
            if prev is None:
                ws_bboxes[lbl] = bbox
            else:
                ws_bboxes[lbl] = (
                    min(prev[0], bbox[0]),
                    min(prev[1], bbox[1]),
                    max(prev[2], bbox[2]),
                    max(prev[3], bbox[3]),
                )

        for trip in triples:
            best_lbl = None
            best_overlap = -1.0
            best_dist = None
            for lbl, vals in placements:
                tx, ty, rot, sx, sy = vals
                pts_ws = [(px * sx + tx, py * sy + ty) for px, py in trip]
                arc_bbox = (
                    min(p[0] for p in pts_ws),
                    min(p[1] for p in pts_ws),
                    max(p[0] for p in pts_ws),
                    max(p[1] for p in pts_ws),
                )
                pb = ws_bboxes.get(lbl)
                if not pb:
                    continue
                ox = max(0.0, min(arc_bbox[2], pb[2]) - max(arc_bbox[0], pb[0]))
                oy = max(0.0, min(arc_bbox[3], pb[3]) - max(arc_bbox[1], pb[1]))
                overlap = ox * oy
                arc_cx = (arc_bbox[0] + arc_bbox[2]) * 0.5
                arc_cy = (arc_bbox[1] + arc_bbox[3]) * 0.5
                pb_cx = (pb[0] + pb[2]) * 0.5
                pb_cy = (pb[1] + pb[3]) * 0.5
                dist = abs(arc_cx - pb_cx) + abs(arc_cy - pb_cy)
                if overlap > best_overlap or (overlap == best_overlap and (best_dist is None or dist < best_dist)):
                    best_overlap = overlap
                    best_dist = dist
                    best_lbl = lbl
            if best_lbl:
                arc_map.setdefault(best_lbl, []).append(trip)
    else:
        arc_map["CArc"] = triples

    return arc_map


def _find_first_component_trailer_start(payload: bytes) -> int:
    magic = struct.pack("<I", 0x075BCD15)
    idx = 0
    best = None
    while True:
        hit = payload.find(magic, idx)
        if hit == -1:
            break
        start = None
        for name_len in range(1, 33):
            candidate = hit - 1 - name_len
            if candidate < 0:
                continue
            if payload[candidate] != name_len:
                continue
            name_bytes = payload[candidate + 1 : candidate + 1 + name_len]
            if not name_bytes:
                continue
            if not all(32 <= b < 127 for b in name_bytes):
                continue
            start = candidate
            break
        if start is not None:
            best = start if best is None else min(best, start)
        idx = hit + 1
    return -1 if best is None else best


def _compute_component_bbox(
    placements: Sequence[tuple[str, tuple[float, float, float, float, float]]],
    segments: dict[str, list[tuple[tuple[float, float], tuple[float, float]]]],
    *,
    use_scales: bool,
) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for label, values in placements:
        segs = segments.get(label)
        if not segs:
            continue
        tx, ty, _, sx, sy = values
        if not use_scales:
            sx = sy = 1.0
        for p1, p2 in segs:
            xs.extend([p1[0] * sx + tx, p2[0] * sx + tx])
            ys.extend([p1[1] * sy + ty, p2[1] * sy + ty])
    if not xs or not ys:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def _extract_inline_components(
    payload: bytes,
    labels: Sequence[str],
    *,
    stop_offset: int,
) -> dict[str, tuple[list[LineEntity], list[tuple[int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]]]]:
    markers: list[tuple[int, str, int]] = []
    for label in labels:
        marker = bytes([len(label)]) + label.encode("ascii", errors="ignore")
        pos = payload.find(marker, 0, stop_offset)
        if pos != -1:
            markers.append((pos, label, len(marker)))
    markers.sort()
    if not markers:
        return {}

    first_marker_pos = markers[0][0]
    search_back = min(first_marker_pos, 16_384)
    window_start = first_marker_pos - search_back
    best_start: int | None = None
    best_score = 0
    cursor = first_marker_pos
    while True:
        hit = payload.rfind(b"\x01\x80", window_start, cursor)
        if hit == -1:
            break
        bbox, lines, arcs = _parse_inline_float_records(payload[hit:first_marker_pos])
        score = len(lines) + len(arcs)
        if score > best_score:
            best_score = score
            best_start = hit
        cursor = hit
    stream_start = best_start if best_start is not None and best_score >= 4 else (best_start or window_start)

    extracted: dict[
        str,
        tuple[
            list[LineEntity],
            list[tuple[int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]],
        ],
    ] = {}

    prev_end = stream_start
    for pos, label, marker_len in markers:
        if pos <= prev_end:
            prev_end = pos + marker_len
            continue
        chunk = payload[prev_end:pos]
        prev_end = pos + marker_len

        bbox_hint, lines, arcs = _parse_inline_float_records(chunk)
        if bbox_hint is not None and (lines or arcs):
            extracted[label] = (lines, arcs)
            continue

        bbox_hint2, lines, arcs = _parse_inline_labeled_float_records(chunk)
        if bbox_hint2 is not None and (lines or arcs):
            extracted[label] = (lines, arcs)
            continue

        bbox = bbox_hint or bbox_hint2 or _extract_bbox_from_inline_chunk(chunk)
        if bbox is not None:
            dbl_lines, dbl_arcs = _scan_inline_double_records(chunk, bbox)
            if dbl_lines or dbl_arcs:
                extracted[label] = (dbl_lines, dbl_arcs)
                continue

        if chunk.find(struct.pack("<h", COMPONENT_SENTINEL)) != -1:
            segs, arc_triplets, _ = _decode_glyph_chunk(label, chunk)
            if not segs and not arc_triplets:
                continue
            fallback_lines = [LineEntity(layer=0, start=a, end=b) for a, b in segs if a != b]
            fallback_arcs = [(0, trip) for trip in arc_triplets]
            extracted[label] = (fallback_lines, fallback_arcs)

    return extracted


def _extract_inline_component_geometry(payload: bytes) -> tuple[list[LineEntity], list[InsertEntity], list[ArcEntity]]:
    placements, place_offset = _collect_glyph_placements(payload)
    if not placements:
        return [], [], []

    if place_offset == -1:
        definition_ids = {definition.component_id for definition in iter_component_definitions(payload)}
        trailer_ids = {trailer.component_id for trailer in iter_placement_trailers(payload)}
        if trailer_ids and definition_ids and trailer_ids.isdisjoint(definition_ids):
            return [], [], []

    labels = [label for label, _ in placements]
    stop_offset = place_offset if place_offset != -1 else len(payload)
    if place_offset == -1:
        trailer_start = _find_first_component_trailer_start(payload)
        if trailer_start != -1:
            name_len = payload[trailer_start] if trailer_start < len(payload) else 0
            marker_end = trailer_start + 1 + int(name_len)
            if marker_end > trailer_start:
                stop_offset = min(stop_offset, min(marker_end, len(payload)))
            else:
                stop_offset = min(stop_offset, trailer_start)
    label_list = list(dict.fromkeys(labels))

    extracted = _extract_inline_components(payload, label_list, stop_offset=stop_offset)

    lines: list[LineEntity] = []
    inserts: list[InsertEntity] = []
    arcs_out: list[ArcEntity] = []

    for label, values in placements:
        local_lines, local_arcs = extracted.get(label, ([], []))
        if not local_lines and not local_arcs:
            continue
        tx, ty, rot, sx, sy = values
        rotation_rad = math.radians(rot) if abs(rot) > 1e-6 else 0.0
        cos_theta = math.cos(rotation_rad) if rotation_rad else 1.0
        sin_theta = math.sin(rotation_rad) if rotation_rad else 0.0

        def xf(pt: tuple[float, float]) -> tuple[float, float]:
            x = pt[0] * sx
            y = pt[1] * sy
            if rotation_rad:
                x, y = (x * cos_theta - y * sin_theta, x * sin_theta + y * cos_theta)
            return (x + tx, y + ty)

        for ln in local_lines:
            a = xf(ln.start)
            b = xf(ln.end)
            if a != b and all(_looks_like_coordinate(v) for v in (*a, *b)):
                lines.append(LineEntity(layer=0, start=a, end=b))

        for layer, (p0, p1, p2) in local_arcs:
            ws0 = xf(p0)
            ws1 = xf(p1)
            ws2 = xf(p2)
            if not all(_looks_like_coordinate(v) for v in (*ws0, *ws1, *ws2)):
                continue
            solution = _circle_from_points(ws0, ws1, ws2)
            if solution is None:
                continue
            cx, cy, radius = solution
            if not math.isfinite(radius) or radius <= 1e-6:
                continue
            center = (cx, cy)

            def _angle(pt: tuple[float, float]) -> float:
                return math.atan2(pt[1] - cy, pt[0] - cx) % math.tau

            ang0 = _angle(ws0)
            ang1 = _angle(ws1)
            ang2 = _angle(ws2)
            sweep = (ang2 - ang0) % math.tau
            rel1 = (ang1 - ang0) % math.tau
            start = ws0
            end = ws2
            if rel1 > sweep + 1e-6:
                start, end = end, start

            arcs_out.append(ArcEntity(layer=0, center=center, start=start, end=end))

    rounded: dict[Tuple[int, Tuple[float, float], Tuple[float, float]], LineEntity] = {}
    for ln in lines:
        key = (
            ln.layer,
            (round(ln.start[0], 4), round(ln.start[1], 4)),
            (round(ln.end[0], 4), round(ln.end[1], 4)),
        )
        rounded[key] = ln
    lines = list(rounded.values())

    arc_index: dict[Tuple[int, float, float, float, float, float], ArcEntity] = {}
    for arc in arcs_out:
        cx, cy = arc.center
        radius = math.hypot(arc.start[0] - cx, arc.start[1] - cy)
        start_ang = math.atan2(arc.start[1] - cy, arc.start[0] - cx)
        end_ang = math.atan2(arc.end[1] - cy, arc.end[0] - cx)
        key = (
            arc.layer,
            round(cx, 5),
            round(cy, 5),
            round(radius, 5),
            round(start_ang, 5),
            round(end_ang, 5),
        )
        arc_index[key] = arc
    arcs_out = list(arc_index.values())

    if arcs_out:
        lines = _prune_lines_against_arcs(lines, arcs_out)

    return lines, inserts, arcs_out


__all__ = [
    "write_dxf",
    "_collect_component_circles",
    "_segments_from_inline_floats",
    "_parse_inline_float_records",
    "_extract_bbox_from_inline_chunk",
    "_scan_inline_double_records",
    "_parse_inline_labeled_float_records",
    "_decode_glyph_chunk",
    "_load_payload_glyph_components",
    "_looks_like_coordinate",
    "_parse_component_place_entries",
    "_component_segments",
    "_segments_from_inline_chunk",
    "_collect_glyph_placements",
    "_parse_carc_arcs",
    "_find_first_component_trailer_start",
    "_compute_component_bbox",
    "_extract_inline_components",
    "_extract_inline_component_geometry",
    "_extract_short_component_lines",
    "_segments_from_short_chunk",
    "_matches_component_helper",
]
