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
from monucad.mcd import (
    _collect_component_circles,
    _decode_glyph_chunk,
    _extract_bbox_from_inline_chunk,
    _extract_inline_component_geometry,
    _extract_short_component_lines,
    _find_first_component_trailer_start,
    _load_payload_glyph_components,
    _matches_component_helper,
    _parse_inline_float_records,
    _parse_inline_labeled_float_records,
    _scan_inline_double_records,
    _segments_from_inline_floats,
    write_dxf,
)
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
from font_components import iter_component_files
from placement_parser import GlyphPlacementRecord, extract_glyph_records, iter_placement_trailers

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
