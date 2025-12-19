#!/usr/bin/env python3
"""
Brute-force extractor for MONU-CAD v9 .mcd files.

The format appears to embed a valid deflate stream at an arbitrary offset
inside an otherwise bogus gzip container.  This tool scans the file for
any byte offset that can be decompressed with raw zlib/deflate, looks for
LINE entity definitions inside the recovered payload, and emits a minimal
DXF file that recreates the vector geometry.

The heuristics are intentionally loose so the script keeps working even if
the vendor tweaks the binary padding.  It is not a full specification of
the .mcd format, but it is enough to reverse simple line work so the data
can be moved into another CAD package for inspection.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple
from monucad.deflate_io import DEFAULT_MIN_PAYLOAD, brute_force_deflate, collect_deflate_streams
from monucad.entities import ArcEntity, CircleEntity, DuplicateRecord, InsertEntity, LineEntity
from monucad.logging import ArcHelperLogger, log_duplicate_records
from monucad.geometry import (
    DUP_FINGERPRINT_PLACES,
    HELPER_AXIS_TOL,
    MAX_COORD_MAGNITUDE,
    circle_from_points as _circle_from_points,
    fuzzy_eq as _fuzzy_eq,
    is_alignment_helper as _is_alignment_helper,
    point_in_bbox as _point_in_bbox,
    points_match as _points_match,
    prune_lines_against_arcs as _prune_lines_against_arcs,
    record_fingerprint as _record_fingerprint,
)
from monucad.placement import collect_candidate_records, extract_new_style_component_lines as _extract_new_style_component_lines
from monucad.mcd import write_dxf
from monucad.fonts import (
    FontDefinition,
    FontManager,
    Glyph,
    TextEntity,
    GLYPH_COORD_SCALE,
    MAIN_GLYPH_MAP,
    PRINTABLE_ASCII,
    build_font_prefix_map as _build_font_prefix_map,
    build_vm_mapping as _build_vm_mapping,
    decode_glyph_label as _decode_glyph_label,
    derive_font_mapping as _derive_font_mapping,
    locate_metrics as _locate_metrics,
    looks_like_font_name as _looks_like_font_name,
    match_known_font as _match_known_font,
)
from component_parser import (
    ComponentSubBlock,
    CirclePrimitive,
    ComponentDefinition,
    extract_circle_primitives,
    iter_component_definitions,
    iter_label_chunks,
    iter_component_placements,
)
from font_components import iter_component_files, parse_component_bytes
from glyph_tlv_parser import parse_glyph_components, parse_glyph_components_from_blob
from extract_font_components import extract_components
from font_components import SENTINEL as COMPONENT_SENTINEL, POINT_SCALE as COMPONENT_POINT_SCALE
from placement_parser import (
    GlyphPlacementRecord,
    PlacementTrailer,
    extract_glyph_records,
    extract_glyph_records_with_offsets,
    iter_placement_trailers,
)

def _log_duplicate_records(records: Sequence[DuplicateRecord], destination: Path) -> None:
    # Backward shim for callers inside this module; actual implementation lives in monucad.logging.
    log_duplicate_records(records, destination)

def _looks_like_coordinate(value: float) -> bool:
    return math.isfinite(value) and abs(value) <= MAX_COORD_MAGNITUDE


def _fit_arc(points: Sequence[Tuple[float, float]]) -> Tuple[Tuple[float, float], float] | None:
    """Fit a circle to 3 points; return center and radius if valid."""

    if len(points) < 3:
        return None
    res = _circle_from_points(points[0], points[1], points[2])
    if res is None:
        return None
    cx, cy, r = res
    if not math.isfinite(r) or r <= 0:
        return None
    return (cx, cy), r


def _promote_line_chains_to_arcs(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    *,
    max_err: float = 0.0005,
    max_promote: int = 4,
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]]:
    """
    Given a list of line segments (glyph space), attempt to replace a few small
    chains with arcs when the three-point fit is extremely tight.
    Returns (remaining_segments, new_arc_triplets).
    """

    remaining = list(segments)
    new_arcs: list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]] = []
    if len(remaining) < 2 or max_promote <= 0:
        return remaining, new_arcs
    idx = 0
    promoted = 0
    while idx + 1 < len(remaining) and promoted < max_promote:
        chain = remaining[idx : idx + 2]
        pts = [chain[0][0], chain[0][1], chain[1][1]]
        fit = _fit_arc(pts)
        if fit:
            center, radius = fit
            errs = [abs(math.hypot(p[0] - center[0], p[1] - center[1]) - radius) for p in pts]
            if max(errs) <= max_err:
                new_arcs.append((pts[0], pts[1], pts[2]))
                del remaining[idx : idx + 2]
                promoted += 1
                continue
        idx += 1
    return remaining, new_arcs


def parse_entities(
    payload: bytes,
    *,
    helper_logger: ArcHelperLogger | None = None,
    duplicate_log: Path | None = None,
) -> Tuple[List[LineEntity], List[ArcEntity], List[CircleEntity], List[InsertEntity]]:
    """
    Walk the payload and collect both LINE (etype=2) and ARC (etype=3) records.
    Legacy files often place real geometry on layer 0, so we keep everything.
    """

    # Older payloads sometimes encode entity coordinates as float32 instead of
    # float64. Scan for both and dedupe later.
    placements = list(iter_placement_trailers(payload))

    double_records = collect_candidate_records(payload, coord_format="double")
    double_offsets = {offset for offset, *_ in double_records}
    float_records = [
        rec for rec in collect_candidate_records(payload, coord_format="float") if rec[0] not in double_offsets
    ]
    records = double_records + float_records
    tlv_lines = _extract_short_component_lines(payload)
    inline_lines, inline_inserts, inline_arcs = _extract_inline_component_geometry(payload)
    component_lines, component_inserts, component_arcs = _extract_new_style_component_lines(payload)
    if not component_lines and not component_arcs and placements:
        # Second-pass with arc-guess enabled to rescue icon tables that otherwise decode to nothing.
        component_lines, component_inserts, component_arcs = _extract_new_style_component_lines(
            payload, force_arc_guess=True
        )
    # Inline component payloads and inline instancing are mutually exclusive. If
    # either path produced geometry, suppress the legacy line/arc records so we
    # do not double-emit stroke data that lives in component space.
    suppress_inline_records = bool(inline_lines or inline_arcs or component_lines or component_arcs)

    trailer_start = _find_first_component_trailer_start(payload)

    # When inline component geometry is present, the payload often also
    # contains *component-local* line/arc records (coordinates ~[-1,1]) used to
    # describe glyph strokes. Those are not world-space entities and should not
    # be merged back into the drawing view. Filter them conservatively.
    inline_span: float | None = None
    if inline_lines:
        xs = [p for ln in inline_lines for p in (ln.start[0], ln.end[0])]
        ys = [p for ln in inline_lines for p in (ln.start[1], ln.end[1])]
        if xs and ys:
            inline_span = max(max(xs) - min(xs), max(ys) - min(ys))

    duplicate_records: List[DuplicateRecord] = []
    filtered_records: List[Tuple[int, int, int, float, float, float, float]] = []
    seen_fingerprints: dict[Tuple[int, int, float, float, float, float], int] = {}
    for offset, layer, etype, x1, y1, x2, y2 in records:
        if suppress_inline_records and layer == 0 and etype in (2, 3):
            continue
        if (
            trailer_start != -1
            and inline_span is not None
            and inline_span > 5.0
            and offset < trailer_start
            and max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0
        ):
            continue
        if (
            inline_span is not None
            and inline_span > 5.0
            and layer == 0
            and max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 2.0
        ):
            continue
        fingerprint = _record_fingerprint(layer, etype, x1, y1, x2, y2)
        prev = seen_fingerprints.get(fingerprint)
        if prev is not None:
            duplicate_records.append(
                DuplicateRecord(
                    offset=offset,
                    original_offset=prev,
                    layer=layer,
                    etype=etype,
                    start=(x1, y1),
                    end=(x2, y2),
                )
            )
            continue
        seen_fingerprints[fingerprint] = offset
        filtered_records.append((offset, layer, etype, x1, y1, x2, y2))
    records = filtered_records

    lines: List[LineEntity] = []
    arcs: List[ArcEntity] = list(inline_arcs)
    circles: List[CircleEntity] = []
    inserts: List[InsertEntity] = list(inline_inserts)
    arc_counter = 0

    if inline_lines:
        lines.extend(inline_lines)
    elif tlv_lines:
        # TLV payloads often coexist with legacy type=2/3 records in newer
        # drawings. Treat them as *additional* geometry instead of replacing
        # the record stream entirely so we keep arcs/circles when both encodings
        # appear. Legacy TLV-only files still work because `records` can be
        # empty.
        lines.extend(tlv_lines)

    # Helper index so we can look ahead for follow-up records (e.g., arc centers).
    by_offset = {offset: (layer, etype, x1, y1, x2, y2) for offset, layer, etype, x1, y1, x2, y2 in records}
    record_offsets = [offset for offset, *_ in records]

    # Identify circle candidates by looking for <type=2> records whose endpoints are mirrored
    # by a nearby <type=0> helper.
    consumed_as_circle: set[int] = set()
    for idx, offset in enumerate(record_offsets):
        layer, etype, x1, y1, x2, y2 = by_offset[offset]
        if etype != 2:
            continue
        center = (x1, y1)
        rim = (x2, y2)
        if not all(_looks_like_coordinate(v) for v in (*center, *rim)):
            continue
        for look_ahead in record_offsets[idx + 1 : idx + 1 + 25]:
            if look_ahead - offset > 200:
                break
            l2, t2, rx1, ry1, rx2, ry2 = by_offset[look_ahead]
            if t2 != 0:
                continue
            if _fuzzy_eq(rx1, rim[0]) and _fuzzy_eq(ry1, rim[1]) and _fuzzy_eq(rx2, center[0]) and _fuzzy_eq(
                ry2, center[1]
            ):
                radius = math.hypot(rim[0] - center[0], rim[1] - center[1])
                if 1e-6 < radius <= MAX_COORD_MAGNITUDE:
                    circles.append(CircleEntity(layer=layer, center=center, radius=radius))
                    consumed_as_circle.add(offset)
                break

    for idx, offset in enumerate(record_offsets):
        layer, etype, x1, y1, x2, y2 = by_offset[offset]

        if etype == 2:
            if offset in consumed_as_circle:
                continue
            if not all(_looks_like_coordinate(v) for v in (x1, y1, x2, y2)):
                continue
            if abs(x1 - x2) < 1e-9 and abs(y1 - y2) < 1e-9:
                continue
            candidate = LineEntity(layer=layer, start=(x1, y1), end=(x2, y2))
            if _is_alignment_helper(candidate):
                continue
            lines.append(candidate)
            continue

        if etype == 3:
            arc_counter += 1
            helper_records: List[Tuple[int, int, int, float, float, float, float]] = []
            if helper_logger:
                neighbor_offsets = record_offsets[idx + 1 : idx + 1 + helper_logger.window]
                helper_records = [
                    (n_offset, *by_offset[n_offset]) for n_offset in neighbor_offsets if n_offset in by_offset
                ]
            raw_end = (x1, y1)
            guide_point = (x2, y2)
            if not all(_looks_like_coordinate(v) for v in (*raw_end, *guide_point)):
                if helper_logger:
                    helper_logger.record(
                        seq=arc_counter,
                        arc_offset=offset,
                        layer=layer,
                        start=raw_end,
                        center=guide_point,
                        neighbors=helper_records,
                        note="skipped: invalid coordinates",
                    )
                continue

            radius = math.hypot(raw_end[0] - guide_point[0], raw_end[1] - guide_point[1])
            start_point: Tuple[float, float] | None = None

            def _helper_from_offset(source_offsets: Iterable[int]) -> Tuple[float, float] | None:
                for candidate in source_offsets:
                    if abs(candidate - offset) > 200:
                        break
                    layer2, etype2, ex1, ey1, ex2, ey2 = by_offset[candidate]
                    if etype2 != 0:
                        continue
                    valid_first = _looks_like_coordinate(ex1) and _looks_like_coordinate(ey1)
                    valid_second = _looks_like_coordinate(ex2) and _looks_like_coordinate(ey2)
                    if valid_first and valid_second and _fuzzy_eq(ex1, guide_point[0]) and _fuzzy_eq(
                        ey1, guide_point[1]
                    ):
                        return (ex2, ey2)
                    if valid_first:
                        dist = math.hypot(ex1 - guide_point[0], ey1 - guide_point[1])
                        if abs(dist - radius) <= max(1e-3, radius * 0.05):
                            return (ex1, ey1)
                return None

            forward_window = helper_logger.window if helper_logger else 20
            forward_offsets = record_offsets[idx + 1 : idx + 1 + forward_window]
            start_point = _helper_from_offset(forward_offsets)

            if start_point is None:
                backward_window = helper_logger.window if helper_logger else 20
                backward_slice = record_offsets[max(0, idx - backward_window) : idx]
                start_point = _helper_from_offset(reversed(backward_slice))
            if start_point is None:
                if helper_logger:
                    helper_logger.record(
                        seq=arc_counter,
                        arc_offset=offset,
                        layer=layer,
                        start=raw_end,
                        center=guide_point,
                        neighbors=helper_records,
                        note="skipped: missing start helper",
                    )
                continue

            vec1 = (start_point[0] - guide_point[0], start_point[1] - guide_point[1])
            vec2 = (raw_end[0] - guide_point[0], raw_end[1] - guide_point[1])
            cross = vec1[0] * vec2[1] - vec1[1] * vec2[0]
            if cross > 0:
                start_point, raw_end = raw_end, start_point

            solution = _circle_from_points(start_point, raw_end, guide_point)
            if solution is not None:
                actual_center = (solution[0], solution[1])
            else:
                actual_center = guide_point
            arcs.append(ArcEntity(layer=layer, center=actual_center, start=start_point, end=raw_end))
            note_text = "parsed successfully (start/end swapped)" if cross > 0 else "parsed successfully"
            if helper_logger:
                helper_logger.record(
                    seq=arc_counter,
                    arc_offset=offset,
                    layer=layer,
                    start=start_point,
                    center=actual_center,
                    neighbors=helper_records,
                    note=note_text,
                )

    component_circles, helper_segments = _collect_component_circles(payload)
    if component_circles:
        circles.extend(component_circles)

    if helper_segments:
        lines = [line for line in lines if not _matches_component_helper(line, helper_segments)]

    # New-style component blocks encode un-exploded lettering and should not be
    # mixed with already-decoded inline (exploded) component geometry. The
    # resaved MCPro9-era files often contain both placement trailers and
    # legacy inline component blobs; in those cases the new-style scanner can
    # false-positive and introduce large "starburst" artifacts.
    # Prefer placement-based instancing when available; fall back to inline blobs
    # if instancing produced nothing.
    if component_lines or component_arcs:
        lines.extend(component_lines)
        arcs.extend(component_arcs)
        inserts.extend(component_inserts)
    else:
        lines.extend(inline_lines)
        arcs.extend(inline_arcs)
        inserts.extend(inline_inserts)

    # If we already decoded inline glyph/component geometry, skip the heuristic
    # text scanners to avoid double-rendering and false positives.
    if not inline_lines and not component_lines and not component_arcs:
        text_lines, _missing_fonts = _render_text_lines(payload)
        if text_lines and len(text_lines) <= 20000:
            lines.extend(text_lines)

    # Prune instanced/inline lines that lie on emitted arcs to reduce duplication.
    if arcs and lines:
        lines = _prune_lines_against_arcs(lines, arcs)

    # Deduplicate line segments
    dedup_lines: dict[Tuple[int, Tuple[float, float], Tuple[float, float]], LineEntity] = {}
    for entity in lines:
        dedup_lines[(entity.layer, entity.start, entity.end)] = entity
    dedup_arcs: dict[Tuple[int, Tuple[float, float], Tuple[float, float]], ArcEntity] = {}
    for entity in arcs:
        dedup_arcs[(entity.layer, entity.start, entity.end)] = entity

    dedup_circles: dict[Tuple[int, Tuple[float, float]], CircleEntity] = {}
    for entity in circles:
        dedup_circles[(entity.layer, entity.center)] = entity

    if duplicate_records:
        sample = ", ".join(
            f"0x{entry.offset:04X}->0x{entry.original_offset:04X}/etype={entry.etype}"
            for entry in duplicate_records[:5]
        )
        print(
            f"[warn] Detected {len(duplicate_records)} duplicate geometry record(s); "
            f"sample: {sample}"
        )
        if duplicate_log:
            _log_duplicate_records(duplicate_records, duplicate_log)
            print(f"[i] Duplicate record log written to {duplicate_log}")

    return (
        list(dedup_lines.values()),
        list(dedup_arcs.values()),
        list(dedup_circles.values()),
        inserts,
    )




_FONT_MANAGER: FontManager | None = None
_LAST_MISSING_FONTS: set[str] = set()
_WARNED_MISSING_DTA: set[str] = set()
_WARNED_PUNCT_FLAG: set[str] = set()
_WARNED_SERIF_FLAG: set[str] = set()


def _get_font_manager() -> FontManager | None:
    global _FONT_MANAGER
    if _FONT_MANAGER is not None:
        return _FONT_MANAGER
    font_root = Path(__file__).resolve().parent / "FONTS"
    if not font_root.exists():
        return None
    manager = FontManager(font_root)
    if not manager.known_fonts():
        return None
    _FONT_MANAGER = manager
    return _FONT_MANAGER


def _render_text_lines(payload: bytes) -> Tuple[List[LineEntity], set[str]]:
    manager = _get_font_manager()
    if not manager:
        return [], set()
    entities: List[LineEntity] = []
    missing: set[str] = set()
    global _LAST_MISSING_FONTS
    _LAST_MISSING_FONTS = set()
    main_fallback = manager.get_font("MAIN")
    for text_entity in _iter_text_entities(payload, manager):
        font = manager.get_font(text_entity.font) or main_fallback
        if not font:
            missing.add(text_entity.font)
            continue
        rendered = font.render(text_entity.text, text_entity.metrics, layer=0)
        if not rendered and text_entity.text.strip():
            missing.add(text_entity.font)
            continue
        entities.extend(rendered)
    _LAST_MISSING_FONTS = set(missing)
    return entities, missing


TEXT_FLOAT_COUNT = 8


def _align4(value: int) -> int:
    return (value + 3) & ~3


def _read_ascii_lp_string(
    payload: bytes,
    offset: int,
    *,
    max_len: int = 96,
) -> tuple[str | None, int]:
    if offset >= len(payload):
        return None, offset
    length = payload[offset]
    if 1 <= length <= max_len and offset + 1 + length <= len(payload):
        chunk = payload[offset + 1 : offset + 1 + length]
        if all(32 <= byte < 127 for byte in chunk):
            return chunk.decode("ascii"), offset + 1 + length
    first = payload[offset]
    if first < 32 or first >= 127:
        return None, offset
    cursor = offset
    chars: list[int] = []
    while cursor < len(payload):
        byte = payload[cursor]
        if byte == 0:
            break
        if byte < 32 or byte >= 127:
            return None, offset
        chars.append(byte)
        cursor += 1
        if len(chars) >= max_len:
            break
    if not chars:
        return None, offset
    cursor = min(cursor + 1, len(payload))
    return bytes(chars).decode("ascii"), cursor


def _iter_text_entities(payload: bytes, manager: FontManager) -> Iterable[TextEntity]:
    known_fonts = manager.known_fonts()
    seen_offsets: set[int] = set()
    yield from _iter_length_pref_text(payload, known_fonts, seen_offsets)
    yield from _iter_cstring_text(payload, known_fonts, seen_offsets)
    yield from _iter_glyph_label_text(payload, manager)


def _iter_length_pref_text(
    payload: bytes,
    known_fonts: set[str],
    seen_offsets: set[int],
) -> Iterable[TextEntity]:
    idx = 0
    limit = len(payload)
    while idx < limit - 64:
        text_len = payload[idx]
        if not (1 <= text_len <= 64):
            idx += 1
            continue
        text_start = idx + 1
        text_end = text_start + text_len
        if text_end + TEXT_FLOAT_COUNT * 4 + 2 >= limit:
            break
        text_bytes = payload[text_start:text_end]
        if not text_bytes or any(byte < 32 or byte > 126 for byte in text_bytes):
            idx += 1
            continue
        metrics_offset, metrics = _locate_metrics(payload, _align4(text_end), TEXT_FLOAT_COUNT)
        if metrics_offset is None:
            idx = text_end
            continue
        cursor = metrics_offset + TEXT_FLOAT_COUNT * 4
        font_len_offset = cursor
        if font_len_offset >= limit:
            break
        raw_font_len = payload[font_len_offset]
        font_name: str | None = None
        cursor_after_font = font_len_offset + 1
        if 1 <= raw_font_len <= 64:
            font_end = cursor_after_font + raw_font_len
            if font_end <= limit:
                font_name = payload[cursor_after_font:font_end].decode("ascii", errors="ignore").upper()
                cursor_after_font = font_end
        if not font_name:
            candidate, next_cursor = _read_ascii_lp_string(payload, font_len_offset)
            if candidate:
                font_name = candidate.upper()
                cursor_after_font = next_cursor
        if not font_name:
            font_name, cursor_after_font = _match_known_font(payload, font_len_offset, known_fonts)
        if not font_name:
            idx = text_end
            continue
        if not _looks_like_font_name(font_name):
            idx = text_end
            continue
        if metrics_offset in seen_offsets:
            idx = text_end
            continue
        seen_offsets.add(metrics_offset)
        yield TextEntity(text=text_bytes.decode("ascii"), font=font_name, metrics=metrics)
        idx = cursor_after_font


def _iter_cstring_text(
    payload: bytes,
    known_fonts: set[str],
    seen_offsets: set[int],
) -> Iterable[TextEntity]:
    limit = len(payload)
    idx = 0
    while idx < limit - 64:
        if payload[idx] < 32 or payload[idx] > 126:
            idx += 1
            continue
        end = idx
        while end < limit and 32 <= payload[end] < 127 and (end - idx) < 64:
            end += 1
        if end == idx or end + 1 >= limit or payload[end : end + 2] != b"\x00\x00":
            idx = end + 1
            continue
        text_bytes = payload[idx:end]
        metrics_offset, metrics = _locate_metrics(payload, _align4(end + 2), TEXT_FLOAT_COUNT)
        if metrics_offset is None:
            idx = end + 1
            continue
        if metrics_offset in seen_offsets:
            idx = end + 1
            continue
        cursor = metrics_offset + TEXT_FLOAT_COUNT * 4
        while cursor < limit and payload[cursor] == 0:
            cursor += 1
        font_name: str | None = None
        cursor_after_font = cursor
        scan_limit = min(cursor + 32, limit)
        probe = cursor
        while probe < scan_limit:
            candidate, next_cursor = _read_ascii_lp_string(payload, probe)
            if candidate:
                candidate_upper = candidate.upper()
                cursor_after_font = next_cursor
                font_name = candidate_upper if candidate_upper in known_fonts else candidate_upper
                break
            probe += 1
        if font_name is None:
            font_name, cursor_after_font = _match_known_font(payload, cursor, known_fonts)
        if not font_name or not _looks_like_font_name(font_name):
            idx = end + 1
            continue
        seen_offsets.add(metrics_offset)
        yield TextEntity(text=text_bytes.decode("ascii"), font=font_name, metrics=metrics)
        idx = cursor_after_font


def _iter_glyph_label_text(payload: bytes, manager: FontManager) -> Iterable[TextEntity]:
    records = extract_glyph_records(payload)
    if not records:
        return
    prefix_map = _build_font_prefix_map(manager)
    if not prefix_map:
        return
    known_fonts = manager.known_fonts()
    seen_keys: set[Tuple[str, int, int]] = set()
    for record in records:
        font_name, char = _decode_glyph_label(record.label, prefix_map)
        if not font_name or not char or font_name not in known_fonts:
            continue
        if not char.strip():
            continue
        metrics = _metrics_from_glyph_record(record)
        if not metrics:
            continue
        key = (font_name, int(record.values[0] * 1e4), int(record.values[1] * 1e4))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        yield TextEntity(text=char, font=font_name, metrics=metrics)


def _metrics_from_glyph_record(record: GlyphPlacementRecord) -> Tuple[float, ...] | None:
    values = record.values
    if len(values) != 5:
        return None
    x, y, rotation, scale_x, scale_y = values
    if not all(math.isfinite(val) for val in (x, y, rotation, scale_x, scale_y)):
        return None
    if abs(x) > MAX_COORD_MAGNITUDE or abs(y) > MAX_COORD_MAGNITUDE:
        return None
    height = abs(scale_y)
    if not (1e-3 <= height <= 1e3):
        return None
    width_scale = abs(scale_x)
    width_value = width_scale / height if width_scale >= 1e-6 else 1.0
    metrics = (
        height,
        0.0,
        rotation,
        0.0,
        width_value,
        0.0,
        x,
        y,
    )
    return metrics


def _load_glyphs_from_components(directory: Path) -> dict[str, Glyph]:
    glyphs: dict[str, Glyph] = {}
    scale = 1.0 / 32768.0
    for definition in iter_component_files(directory):
        segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        bounds = [float("inf"), float("-inf"), float("inf"), float("-inf")]
        for record in definition.records:
            vals = record.values
            if len(vals) < 12:
                continue
            coords = vals[4:12]
            for idx in range(0, len(coords), 4):
                x1, y1, x2, y2 = coords[idx : idx + 4]
                if x1 == y1 == x2 == y2 == 0:
                    continue
                p1 = (x1 * scale, y1 * scale)
                p2 = (x2 * scale, y2 * scale)
                segments.append((p1, p2))
                bounds[0] = min(bounds[0], p1[0], p2[0])
                bounds[1] = max(bounds[1], p1[0], p2[0])
                bounds[2] = min(bounds[2], p1[1], p2[1])
                bounds[3] = max(bounds[3], p1[1], p2[1])
        if not segments:
            continue
        min_x, max_x, min_y, max_y = bounds[0], bounds[1], bounds[2], bounds[3]
        tuple_bounds = (min_x, min_y, max_x, max_y)
        advance = max(max_x - min_x, 0.0)
        glyphs[definition.label] = Glyph(
            label=definition.label,
            segments=segments,
            bounds=tuple_bounds,
            advance=advance,
            baseline=min_y,
        )
    return glyphs


def _glyph_from_component(component) -> Glyph:
    segments = [
        ((float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])))
        for p1, p2 in component.segments
    ]
    min_x, min_y, max_x, max_y = component.bbox
    advance = max(component.advance, max_x - min_x, 0.0)
    baseline = component.baseline if component.baseline is not None else min_y
    return Glyph(
        label=component.label,
        segments=segments,
        bounds=(min_x, min_y, max_x, max_y),
        advance=advance,
        baseline=baseline,
    )


def _glyph_from_component_definition(definition) -> Glyph | None:
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for record in definition.records:
        points = record.normalized_points()
        if len(points) < 2:
            continue
        if len(points) % 2 != 0:
            points = points[:-1]
        for idx in range(0, len(points), 2):
            start = (float(points[idx][0]), float(points[idx][1]))
            end = (float(points[idx + 1][0]), float(points[idx + 1][1]))
            if start == end:
                continue
            segments.append((start, end))
    bbox = definition.bounding_box()
    if not bbox or not segments:
        return None
    min_x, min_y, max_x, max_y = bbox
    advance = max(max_x - min_x, 0.0)
    return Glyph(
        label=definition.label,
        segments=segments,
        bounds=(min_x, min_y, max_x, max_y),
        advance=advance,
        baseline=min_y,
    )


def _load_line_entities_from_dxf(path: Path) -> list[LineEntity]:
    data = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines: list[LineEntity] = []
    idx = 0
    while idx + 1 < len(data):
        code = data[idx].strip()
        value = data[idx + 1].strip()
        idx += 2
        if code != "0" or value != "LINE":
            continue
        x1 = y1 = x2 = y2 = None
        layer = 0
        while idx + 1 < len(data):
            code = data[idx].strip()
            value = data[idx + 1].strip()
            idx += 2
            if code == "0":
                idx -= 2
                break
            if code == "8":
                try:
                    layer = int(value)
                except Exception:
                    layer = 0
            if code == "10":
                x1 = float(value)
            elif code == "20":
                y1 = float(value)
            elif code == "11":
                x2 = float(value)
            elif code == "21":
                y2 = float(value)
        if x1 is None or x2 is None or y1 is None or y2 is None:
            continue
        lines.append(LineEntity(layer=layer, start=(x1, y1), end=(x2, y2)))
    return lines




def _parse_component_place_entries(payload: bytes) -> tuple[float | None, list[tuple[str, tuple[float, float, float, float, float]]], int]:
    marker = b"CComponentPlace"
    idx = payload.find(marker)
    if idx == -1:
        return None, [], -1
    ptr = idx + len(marker)
    try:
        # The legacy format stores a small header after the class label.  In the
        # 2012-era files this appears to be two uint32 values followed by a
        # 5-float transform for the *first* placement, then a chain of
        # (name, unknown bytes, next_transform) tuples.  This means the
        # transform "belongs" to the previous label and the final label can be
        # a terminator with no trailing transform.
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
        # If we don't have enough bytes for the trailing unknown+transform,
        # treat this name as the terminator and reuse the last transform.
        if ptr + 9 + 20 > len(payload):
            entries.append((name, current))
            break
        ptr += 9  # skip unknown bytes
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
    # Inline component blobs in old .mcd files often omit the 28-short header and
    # are just sequences of records delimited by the -0x7FFD sentinel.
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

    # Heuristic: some blobs carry a leading header block before the first sentinel.
    # Drop the first record if it clearly looks like a header (large offsets or
    # unexpected type).
    if records:
        meta = records[0][:4]
        if any(abs(v) > 1024 for v in meta) or (len(meta) >= 3 and meta[2] not in (0, 2)):
            records = records[1:]

    POINT_SCALE = 1.0 / 32768.0
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for values in records:
        if len(values) < 6:
            continue
        coords = values[4:]
        if len(coords) % 2 != 0:
            coords = coords[:-1]
        points = [(coords[i] * POINT_SCALE, coords[i + 1] * POINT_SCALE) for i in range(0, len(coords), 2)]
        if len(points) < 2:
            continue
        # Treat each record as independent stroke segments (pairwise) to avoid
        # connecting unrelated strokes across the record.
        if len(points) % 2 != 0:
            points = points[:-1]
        for idx in range(0, len(points), 2):
            segments.append((points[idx], points[idx + 1]))
    return segments


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


def _point_in_bbox(
    pt: tuple[float, float],
    bbox: tuple[float, float, float, float],
    *,
    tol: float = 0.0,
) -> bool:
    x, y = pt
    min_x, min_y, max_x, max_y = bbox
    return (min_x - tol) <= x <= (max_x + tol) and (min_y - tol) <= y <= (max_y + tol)


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
                return None
            if not all(math.isfinite(v) for v in (min_x, min_y, max_x, max_y)):
                return None
            if any(abs(v) > coord_limit for v in (min_x, min_y, max_x, max_y)):
                return None
            if max_x < min_x or max_y < min_y:
                return None
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
    Some older inline glyph blobs (notably MCPro9-resaved files) store the same
    basic record layout as the float32 variant, but coordinates are float64 and
    record boundaries are not easily walked sequentially. Scan for candidate
    (layer,etype) headers and validate coordinates against the component bbox.
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
    # so scan both even and odd offsets. This still keeps the hot path fast
    # while avoiding the “every other record” drop caused by assuming 2-byte
    # alignment only.
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
    Some legacy inline components (notably the first component in some files)
    store float-record geometry inside TLV-ish string labels like 'CLine' and
    'CArc' without the leading 0x01 0x80 + bbox wrapper.
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
        # First try the common contiguous encoding (fast path).
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


def _decode_glyph_chunk(label: str, chunk: bytes) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]], tuple[float, float, float, float]]:
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


def _load_payload_glyph_components(payload: bytes) -> tuple[dict[str, Glyph], dict[str, list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]]]:
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


def _collect_glyph_placements(payload: bytes) -> tuple[list[tuple[str, tuple[float, float, float, float, float]]], int]:
    """Gather glyph placements from legacy inline ComponentPlace or placement trailers."""

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
    """Decode arcs from the CArc chunk, assign to glyph labels by bbox proximity."""

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

    # Build inline bboxes for assignment.
    inline_bboxes: dict[str, tuple[float, float, float, float]] = {}
    for lbl in labels:
        segs = segments.get(lbl)
        if not segs:
            continue
        xs = [p for seg in segs for p in (seg[0][0], seg[1][0])]
        ys = [p for seg in segs for p in (seg[0][1], seg[1][1])]
        inline_bboxes[lbl] = (min(xs), min(ys), max(xs), max(ys))

    # Assign arcs by world-space overlap (max area) with a nearest-center fallback.
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


def _extract_inline_components(
    payload: bytes,
    labels: Sequence[str],
    *,
    stop_offset: int,
) -> dict[str, tuple[list[LineEntity], list[tuple[int, tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]]]]:
    """
    Extract inline component geometry from legacy .mcd payloads.

    Important: in older (2012-era) files, the length-prefixed label markers
    (e.g. b"\\x04OUTG") appear *after* the component blob they label. This
    means chunk boundaries are inferred as:
      first_chunk = [stream_start : marker0_pos]
      next_chunk  = [marker0_end : marker1_pos]
      ...
      last_chunk  = [markerN-1_end : markerN_pos]

    Bytes after the final marker often belong to world-space geometry and/or
    placement metadata and should not be treated as component-local strokes.
    """

    markers: list[tuple[int, str, int]] = []
    for label in labels:
        marker = bytes([len(label)]) + label.encode("ascii", errors="ignore")
        pos = payload.find(marker, 0, stop_offset)
        if pos != -1:
            markers.append((pos, label, len(marker)))
    markers.sort()
    if not markers:
        return {}

    # Infer the start of the inline stream by searching backwards from the
    # first marker for the 0x01 0x80 header that precedes float-record chunks.
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

        # Fallback to the short/sentinel parser for older component blobs.
        if chunk.find(struct.pack("<h", COMPONENT_SENTINEL)) != -1:
            segs, arc_triplets, _ = _decode_glyph_chunk(label, chunk)
            if not segs and not arc_triplets:
                continue
            fallback_lines = [LineEntity(layer=0, start=a, end=b) for a, b in segs if a != b]
            fallback_arcs = [(0, trip) for trip in arc_triplets]
            extracted[label] = (fallback_lines, fallback_arcs)

    return extracted


def _find_first_component_trailer_start(payload: bytes) -> int:
    """Return the earliest offset of a 0x075BCD15 placement trailer, or -1."""

    magic = struct.pack("<I", 0x075BCD15)
    idx = 0
    best = None
    while True:
        hit = payload.find(magic, idx)
        if hit == -1:
            break
        # Trailer format: <len:1><name:len><magic:4>..., so backtrack to find
        # the length byte.
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


def _extract_inline_component_geometry(payload: bytes) -> tuple[list[LineEntity], list[InsertEntity], list[ArcEntity]]:
    placements, place_offset = _collect_glyph_placements(payload)
    if not placements:
        return [], [], []

    # New-style files (TW84/YIKES/etc.) often include a placement trailer whose
    # component_id does *not* match the component definition id embedded in the
    # payload. Treat those as non-inline to avoid starburst artifacts.
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
            # Allow markers that occur exactly at the trailer start (e.g. the
            # final '\x02SL' marker is also the start of the placement trailer).
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

    # Deduplicate lines with rounding to trim over-segmentation noise.
    rounded: dict[Tuple[int, Tuple[float, float], Tuple[float, float]], LineEntity] = {}
    for ln in lines:
        key = (
            ln.layer,
            (round(ln.start[0], 4), round(ln.start[1], 4)),
            (round(ln.end[0], 4), round(ln.end[1], 4)),
        )
        rounded[key] = ln
    lines = list(rounded.values())

    # Deduplicate arcs (helps when promotion/carc parsing produces overlaps).
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

    # Prune line segments that are represented by arcs.
    if arcs_out:
        lines = _prune_lines_against_arcs(lines, arcs_out)

    return lines, inserts, arcs_out


def _prune_lines_against_arcs(
    lines: list[LineEntity],
    arcs: list[ArcEntity],
    *,
    tol: float = 1e-3,
) -> list[LineEntity]:
    """Remove line segments that lie on emitted arcs (same layer) to reduce duplication."""

    def point_on_arc(pt, arc: ArcEntity) -> bool:
        cx, cy = arc.center
        r = math.hypot(arc.start[0] - cx, arc.start[1] - cy)
        if r <= 1e-9:
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


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompress a MONU-CAD .mcd file and spit out a DXF."
    )
    parser.add_argument("input", type=Path, help="Path to the source .mcd file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional DXF destination (defaults to <input>.dxf)",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Byte offset to start scanning for the hidden deflate stream",
    )
    parser.add_argument(
        "--stop-offset",
        type=int,
        default=None,
        help="Optional byte offset (exclusive) to stop scanning",
    )
    parser.add_argument(
        "--min-payload",
        type=int,
        default=DEFAULT_MIN_PAYLOAD,
        help="Ignore deflate candidates whose decompressed size is smaller than this",
    )
    parser.add_argument(
        "--dump-arc-helpers",
        type=Path,
        help="Write raw helper record context for each type-3 arc to this path",
    )
    parser.add_argument(
        "--arc-helper-window",
        type=int,
        default=16,
        help="Number of subsequent records to capture per arc when dumping helpers",
    )
    parser.add_argument(
        "--duplicate-log",
        type=Path,
        help="Write every duplicate geometry record to this path for Pack Data analysis",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    blob = args.input.read_bytes()

    print(f"[+] Loaded {args.input} ({len(blob)} bytes)")
    streams = collect_deflate_streams(
        blob,
        min_payload=args.min_payload,
        start_offset=args.start_offset,
        stop_offset=args.stop_offset,
    )
    if not streams:
        raise RuntimeError(
            "Unable to locate a deflate payload. "
            "Try passing --start-offset/--stop-offset to narrow the search window."
        )
    streams_sorted = sorted(streams, key=lambda item: len(item[1]), reverse=True)
    offset, payload = streams_sorted[0]
    print(f"[+] Found deflate stream at offset {offset} (payload {len(payload)} bytes)")
    if len(streams_sorted) > 1:
        best_size = len(payload)
        significant = [
            (ofs, data)
            for ofs, data in streams_sorted[1:]
            if best_size - len(data) >= 512
        ]
        if significant:
            preview = ", ".join(f"0x{ofs:X}/{len(data)}B" for ofs, data in significant[:5])
            print(f"[i] Additional deflate streams detected ({len(significant)} significant): {preview}")

    helper_logger = ArcHelperLogger(args.dump_arc_helpers, window=args.arc_helper_window) if args.dump_arc_helpers else None

    lines, arcs, circles, inserts = parse_entities(
        payload,
        helper_logger=helper_logger,
        duplicate_log=args.duplicate_log,
    )
    if helper_logger:
        helper_logger.flush()
    if not lines and not arcs and not circles:
        if _LAST_MISSING_FONTS:
            missing_list = ", ".join(sorted(_LAST_MISSING_FONTS))
            raise RuntimeError(
                f"No supported entities were discovered inside the payload (unsupported fonts: {missing_list})."
            )
        raise RuntimeError("No supported entities were discovered inside the payload.")
    print(
        f"[+] Extracted {len(lines)} line entities, {len(arcs)} arc entities, "
        f"and {len(circles)} circle entities"
    )

    output_path = args.output or args.input.with_suffix(".dxf")
    write_dxf(lines, arcs, circles, output_path, inserts=inserts)
    print(f"[+] DXF written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
