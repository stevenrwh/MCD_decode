from __future__ import annotations

import math
import struct
from collections import Counter, defaultdict
from typing import Iterable, List, Sequence, Tuple

from component_parser import (
    ComponentDefinition,
    ComponentSubBlock,
    iter_component_definitions,
    iter_label_chunks,
)
from font_components import POINT_SCALE as COMPONENT_POINT_SCALE, parse_component_bytes
from monucad.entities import ArcEntity, InsertEntity, LineEntity
from monucad.fonts import GLYPH_COORD_SCALE
from monucad.geometry import MAX_COORD_MAGNITUDE, circle_from_points as _circle_from_points, point_in_bbox as _point_in_bbox
from placement_parser import (
    GlyphPlacementRecord,
    PlacementTrailer,
    extract_glyph_records,
    extract_glyph_records_with_offsets,
    iter_placement_trailers,
)


def collect_candidate_records(
    payload: bytes,
    *,
    coord_format: str = "double",
) -> List[Tuple[int, int, int, float, float, float, float]]:
    """Scan the payload for plausible line/arc records encoded as float/double."""

    if coord_format == "double":
        coord_unpack = "<dddd"
        coord_size = 8 * 4
    elif coord_format == "float":
        coord_unpack = "<ffff"
        coord_size = 4 * 4
    else:
        raise ValueError(f"Unsupported coord_format: {coord_format}")

    record_size = 8 + coord_size
    records: List[Tuple[int, int, int, float, float, float, float]] = []

    # Records are built from uint32 headers + float/double coordinates. In
    # practice the streams appear to be at least 2-byte aligned; scanning every
    # byte produces many false positives in mixed binary payloads.
    for offset in range(0, len(payload) - record_size + 1, 2):
        layer, etype = struct.unpack_from("<II", payload, offset)
        if etype not in (0, 2, 3):
            continue
        if layer > 512:
            continue
        try:
            x1, y1, x2, y2 = struct.unpack_from(coord_unpack, payload, offset + 8)
        except struct.error:
            continue
        if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
            continue
        if not all(abs(v) <= MAX_COORD_MAGNITUDE for v in (x1, y1, x2, y2)):
            continue
        records.append((offset, layer, etype, x1, y1, x2, y2))
    return records


def build_label_block_map(
    definition: ComponentDefinition,
    labels: Sequence[str],
    *,
    catalog_map: dict[str, int] | None = None,
    block_index_map: dict[int, int] | None = None,
    index_block_lists: dict[int, list[int]] | None = None,
    index_label_order: dict[int, list[str]] | None = None,
) -> dict[str, ComponentSubBlock]:
    label_set = set(labels)
    if not label_set:
        return {}
    blocks_4803 = [blk for blk in definition.sub_blocks if getattr(blk, "tag", None) == 0x4803]

    # Prefer the catalog map when available: map label -> block index.
    if catalog_map:
        result: dict[str, ComponentSubBlock] = {}
        # If we have ordered blocks per index, use positional pairing of labels within the same index.
        if index_block_lists and index_label_order:
            for idx, lbls in index_label_order.items():
                if not lbls:
                    continue
                blk_ord_list = index_block_lists.get(idx, [])
                if not blk_ord_list:
                    continue
                for pos, lbl in enumerate(lbls):
                    if lbl not in label_set:
                        continue
                    blk_ord = blk_ord_list[min(pos, len(blk_ord_list) - 1)]
                    if 0 <= blk_ord < len(blocks_4803):
                        result[lbl] = blocks_4803[blk_ord]
            if result and len(result) == len(label_set):
                return result
        for label in label_set:
            idx = catalog_map.get(label)
            if idx is None:
                continue
            blk_idx = None
            if block_index_map:
                blk_idx = block_index_map.get(idx)
            if blk_idx is None:
                blk_idx = idx
            if 0 <= blk_idx < len(blocks_4803):
                result[label] = blocks_4803[blk_idx]
        if result:
            return result

    encoded = {
        label: bytes([len(label)]) + label.encode("ascii", errors="ignore")
        for label in label_set
        if label
    }
    result: dict[str, ComponentSubBlock] = {}
    fallback_blocks: list[ComponentSubBlock] = list(blocks_4803)
    for block in fallback_blocks:
        if not block.payload:
            continue
        payload = block.payload
        for label, needle in encoded.items():
            if label in result:
                continue
            if payload.find(needle) != -1:
                result[label] = block
        if len(result) == len(encoded):
            break

    # If we failed to find all labels with the strict length-prefixed search,
    # try a looser substring search (labels have been observed without the
    # prefixed length byte in newer files).
    if result and len(result) < len(encoded):
        for block in fallback_blocks:
            payload = block.payload
            for label in label_set:
                if label in result:
                    continue
                if label and payload.find(label.encode("ascii", errors="ignore")) != -1:
                    result[label] = block
            if len(result) == len(encoded):
                break

    # Try TLV label chunks inside the definition body.
    if label_set:
        for lbl, payload in iter_label_chunks(definition.raw_payload, definition):
            if lbl in label_set and lbl not in result and payload:
                result[lbl] = ComponentSubBlock(tag=0x4803, dtype=0, count=0, payload=payload, offset=-1)

    # As a last resort, if labels exist and we still have no matches, map the
    # label occurrences (in order) to the 0x4803 blocks (in order).
    if not result and label_set and fallback_blocks:
        ordered_labels: list[str] = []
        data = definition.raw_payload
        idx = 0
        limit = len(data)
        while idx + 1 < limit:
            length = data[idx]
            if 1 <= length <= 16:
                end = idx + 1 + length
                if end <= limit:
                    try:
                        lbl = data[idx + 1 : end].decode("ascii")
                    except UnicodeDecodeError:
                        lbl = None
                    if lbl and lbl in label_set:
                        ordered_labels.append(lbl)
                        idx = end
                        continue
            idx += 1
        blocks_ordered = [blk for blk in definition.sub_blocks if getattr(blk, "tag", None) == 0x4803]
        for lbl, blk in zip(ordered_labels, blocks_ordered):
            result[lbl] = blk
    return result


def transform_point(
    point: Tuple[float, float],
    sx: float,
    sy: float,
    cos_theta: float,
    sin_theta: float,
    tx: float,
    ty: float,
    *,
    base_scale: float = GLYPH_COORD_SCALE,
) -> Tuple[float, float]:
    x_local = point[0] * base_scale * sx
    y_local = point[1] * base_scale * sy
    if abs(sin_theta) < 1e-9 and abs(cos_theta - 1.0) < 1e-9:
        return (x_local + tx, y_local + ty)
    x_rot = x_local * cos_theta - y_local * sin_theta
    y_rot = x_local * sin_theta + y_local * cos_theta
    return (x_rot + tx, y_rot + ty)


def instantiate_glyph_segments(
    segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
    values: Sequence[float],
    *,
    base_scale: float = GLYPH_COORD_SCALE,
    layer: int = 0,
) -> List[LineEntity]:
    if len(values) != 5:
        return []
    tx, ty, rot_deg, sx, sy = [float(v) for v in values]
    rotation_rad = math.radians(rot_deg) if abs(rot_deg) > 1e-9 else 0.0
    cos_theta = math.cos(rotation_rad) if rotation_rad else 1.0
    sin_theta = math.sin(rotation_rad) if rotation_rad else 0.0

    entities: List[LineEntity] = []
    for start, end in segments:
        start_pt = transform_point(start, sx, sy, cos_theta, sin_theta, tx, ty, base_scale=base_scale)
        end_pt = transform_point(end, sx, sy, cos_theta, sin_theta, tx, ty, base_scale=base_scale)
        entities.append(LineEntity(layer=layer, start=start_pt, end=end_pt))
    return entities


def instantiate_glyph_arcs(
    arcs: Sequence[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
    values: Sequence[float],
    *,
    base_scale: float = GLYPH_COORD_SCALE,
    layer: int = 0,
) -> List[ArcEntity]:
    if len(values) != 5:
        return []
    tx, ty, rot_deg, sx, sy = [float(v) for v in values]
    rotation_rad = math.radians(rot_deg) if abs(rot_deg) > 1e-9 else 0.0
    cos_theta = math.cos(rotation_rad) if rotation_rad else 1.0
    sin_theta = math.sin(rotation_rad) if rotation_rad else 0.0

    def xf(pt: Tuple[float, float]) -> Tuple[float, float]:
        return transform_point(pt, sx, sy, cos_theta, sin_theta, tx, ty, base_scale=base_scale)

    entities: List[ArcEntity] = []
    for p0, p1, p2 in arcs:
        ws0, ws1, ws2 = xf(p0), xf(p1), xf(p2)
        sol = _circle_from_points(ws0, ws1, ws2)
        if sol is None:
            continue
        cx, cy, radius = sol
        if radius <= 1e-6 or not math.isfinite(radius):
            continue

        def ang(pt: Tuple[float, float]) -> float:
            return math.atan2(pt[1] - cy, pt[0] - cx) % math.tau

        ang0 = ang(ws0)
        ang1 = ang(ws1)
        ang2 = ang(ws2)
        sweep = (ang2 - ang0) % math.tau
        rel1 = (ang1 - ang0) % math.tau
        start, end = ws0, ws2
        if rel1 > sweep + 1e-6:
            start, end = end, start
        entities.append(ArcEntity(layer=layer, center=(cx, cy), start=start, end=end))
    return entities


def decode_polyline_block(
    block,
    *,
    allow_arc_guess: bool = False,
    bbox: Tuple[float, float, float, float] | None = None,
) -> Tuple[
    List[Tuple[Tuple[float, float], Tuple[float, float]]],
    List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
    float,
]:
    """Decode newer 0x4803 blocks that store int16 polylines instead of TLV records."""

    payload = block.payload
    even_len = len(payload) // 2 * 2
    shorts = struct.unpack("<{}h".format(even_len // 2), payload[:even_len])
    base_scale = GLYPH_COORD_SCALE
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    arcs: List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = []

    def _decode_structured_shorts() -> tuple[list[Tuple[Tuple[float, float], Tuple[float, float]]], float] | None:
        # Treat payload as repeated 8-short records. Each record yields at most
        # one segment: (x0, y0) -> (x1, y1) when the secondary point is nonzero.
        even = len(payload) // 2 * 2
        if even < 16:
            return None
        shorts_local = struct.unpack("<{}h".format(even // 2), payload[:even])
        segments_local: list[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        rec_size = 8
        for i in range(0, len(shorts_local), rec_size):
            rec = shorts_local[i : i + rec_size]
            if len(rec) < rec_size:
                break
            x0, y0 = rec[2], rec[3]
            x1, y1 = rec[6], rec[7]
            if x1 or y1:
                p0 = (x0 * COMPONENT_POINT_SCALE, y0 * COMPONENT_POINT_SCALE)
                p1 = (x1 * COMPONENT_POINT_SCALE, y1 * COMPONENT_POINT_SCALE)
                segments_local.append((p0, p1))
        if not segments_local:
            return None
        return segments_local, 1.0

    # Not enough shorts to be useful -> try structured shorts immediately.
    if len(shorts) < 4:
        struct_result = _decode_structured_shorts()
        if struct_result is not None:
            segs_struct, base_scale = struct_result
            segments.extend(segs_struct)
            return segments, arcs, base_scale
        return [], [], GLYPH_COORD_SCALE

    def _emit(points: List[Tuple[float, float]]) -> None:
        nonlocal base_scale
        if len(points) < 2:
            return
        pts = points
        if bbox:
            min_x, min_y, max_x, max_y = bbox
            span_x = max(max_x - min_x, 1e-6)
            span_y = max(max_y - min_y, 1e-6)
            cx = (min_x + max_x) / 2.0
            cy = (min_y + max_y) / 2.0
            norm_scale_x = span_x / 2.0
            norm_scale_y = span_y / 2.0
            pts = [(cx + px * norm_scale_x, cy + py * norm_scale_y) for px, py in points]
            base_scale = 1.0
        for a, b in zip(pts, pts[1:]):
            if a != b:
                segments.append((a, b))
        if allow_arc_guess and len(pts) >= 3 and len(pts) % 3 == 0:
            for i in range(0, len(pts), 3):
                p0, p1, p2 = pts[i : i + 3]
                sol = _circle_from_points(p0, p1, p2)
                if sol is None:
                    continue
                cx, cy, r = sol
                if not math.isfinite(r) or r <= 1e-6:
                    continue
                arcs.append((p0, p1, p2))

    # Split on quadruple-zero sentinels.
    zero = (0, 0, 0, 0)
    header_len = 2  # drop two shorts of metadata per sub-polyline
    i = 0
    chunk_start = 0
    sentinels = 0
    while i <= len(shorts):
        if i + 3 < len(shorts) and shorts[i : i + 4] == zero:
            chunk = shorts[chunk_start:i]
            if len(chunk) > header_len:
                coords = chunk[header_len:]
                pts = [
                    (coords[j] * COMPONENT_POINT_SCALE, coords[j + 1] * COMPONENT_POINT_SCALE)
                    for j in range(0, len(coords) - 1, 2)
                ]
                _emit(pts)
            i += 4
            chunk_start = i
            sentinels += 1
            continue
        i += 1
    if sentinels == 0:
        coords = shorts[header_len:]
        pts = [
            (coords[j] * COMPONENT_POINT_SCALE, coords[j + 1] * COMPONENT_POINT_SCALE)
            for j in range(0, len(coords) - 1, 2)
        ]
        _emit(pts)
    elif chunk_start < len(shorts):
        chunk = shorts[chunk_start:]
        if len(chunk) > header_len:
            coords = chunk[header_len:]
            pts = [
                (coords[j] * COMPONENT_POINT_SCALE, coords[j + 1] * COMPONENT_POINT_SCALE)
                for j in range(0, len(coords) - 1, 2)
            ]
            _emit(pts)

    return segments, arcs, base_scale


def segments_from_component_block(
    block,
    label: str,
    *,
    allow_arc_guess: bool = False,
    bbox: Tuple[float, float, float, float] | None = None,
) -> Tuple[
    List[Tuple[Tuple[float, float], Tuple[float, float]]],
    List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
    float,
]:
    # Newer 0x4803 blocks sometimes store fixed-size polyline records instead
    # of the legacy sentinel format. When dtype/count are zero, decode them
    # as batches of int16 coordinate pairs or packed floats.
    if getattr(block, "dtype", None) == 0 and getattr(block, "count", None) == 0:
        return decode_polyline_block(block, allow_arc_guess=allow_arc_guess, bbox=bbox)

    try:
        component = parse_component_bytes(label, block.payload)
    except Exception:
        return [], [], GLYPH_COORD_SCALE
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    arcs: List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]] = []
    for record in component.records:
        points = list(record.normalized_points())
        if len(points) < 2:
            continue
        etype = None
        if len(record.metadata) >= 3:
            etype = record.metadata[2]
        # Heuristic: when type is unknown (0/None) or 3, promote triplets to arcs.
        if allow_arc_guess and etype in (0, None, 3) and len(points) >= 3:
            # Always emit line segments for connectivity, optionally add arcs when a valid circle exists.
            for idx in range(0, len(points) - 1):
                segments.append((points[idx], points[idx + 1]))
            triplet_count = len(points) // 3
            for i in range(triplet_count):
                p0, p1, p2 = points[i * 3 : i * 3 + 3]
                solution = _circle_from_points(p0, p1, p2)
                if solution is None:
                    continue
                cx, cy, radius = solution
                if not math.isfinite(radius) or radius <= 1e-6:
                    continue
                arcs.append((p0, p1, p2))
            continue
        # Treat even pairs as segments, and any remaining triplets as arcs.
        pair_count = (len(points) // 2) * 2
        for idx in range(0, pair_count, 2):
            segments.append((points[idx], points[idx + 1]))
        triplet_start = pair_count
        while triplet_start + 2 < len(points):
            p0, p1, p2 = points[triplet_start : triplet_start + 3]
            arcs.append((p0, p1, p2))
            triplet_start += 3
    return segments, arcs, GLYPH_COORD_SCALE


def extract_new_style_component_lines(
    payload: bytes,
    *,
    force_arc_guess: bool = False,
) -> tuple[List[LineEntity], List[InsertEntity], List[ArcEntity]]:
    definitions = {definition.component_id: definition for definition in iter_component_definitions(payload)}
    if not definitions:
        return [], [], []

    placements = list(iter_placement_trailers(payload))
    placement_records: list[tuple[PlacementTrailer, list[GlyphPlacementRecord]]] = []
    component_labels: dict[int, set[str]] = defaultdict(set)
    catalog_records = []
    catalog_map: dict[str, int] = {}
    index_label_order: dict[int, list[str]] = {}
    try:
        from diagnostics.dump_catalog_records import iter_catalog_records

        catalog_records = list(iter_catalog_records(payload))
        catalog_map = {rec.name: rec.index for rec in catalog_records}
        for rec in catalog_records:
            index_label_order.setdefault(rec.index, []).append(rec.name)
    except Exception:
        catalog_map = {}
        index_label_order = {}

    # If no placement trailers were found, attempt to harvest placement-like
    # records directly from 0x4803 sub-blocks. Skip 0x3805 tables to avoid the
    # fixed-stride label catalog. This keeps the scan bounded to block payloads
    # instead of the whole deflate stream.
    if not placements:
        def _block_candidate_records(definition: ComponentDefinition) -> Iterable[tuple[PlacementTrailer, list[GlyphPlacementRecord]]]:
            MIN_BLOCK_RECORDS = 5
            MAX_BLOCK_RECORDS = 200
            for blk_idx, blk in enumerate(definition.sub_blocks):
                if getattr(blk, "tag", None) != 0x4803 or not blk.payload:
                    continue
                records = extract_glyph_records(blk.payload, allow_spaces=True, max_label_len=80)
                if not records:
                    try:
                        records_with_offsets = extract_glyph_records_with_offsets(
                            blk.payload, allow_spaces=True, max_label_len=80
                        )
                        records = [rec for _, rec in records_with_offsets]
                    except Exception:
                        records = []
                if len(records) < MIN_BLOCK_RECORDS or len(records) > MAX_BLOCK_RECORDS:
                    continue
                comp_ids = Counter(rec.component_id for rec in records if rec.component_id)
                component_id = comp_ids.most_common(1)[0][0] if comp_ids else definition.component_id
                trailer = PlacementTrailer(
                    name=f"blk{blk_idx}",
                    component_id=component_id,
                    instance_id=blk_idx,
                    payload=b"",
                )
                yield trailer, records

        for definition in definitions.values():
            new_records = list(_block_candidate_records(definition))
            placement_records.extend(new_records)
            for trailer, records in new_records:
                component_labels[trailer.component_id].update(record.label for record in records)
        placements_from_blocks = bool(placement_records)

        if placements_from_blocks:
            # Heuristic: keep only the richest block-scoped placement sources to avoid
            # instancing every tiny 0x4803 block. Rank by distinct labels and cap the count.
            ranked: list[tuple[int, PlacementTrailer, list[GlyphPlacementRecord]]] = []
            for trailer, records in placement_records:
                label_count = len({rec.label for rec in records if rec.label})
                if label_count < 3:
                    continue
                ranked.append((label_count, trailer, records))
            ranked.sort(key=lambda item: item[0], reverse=True)
            MAX_BLOCK_CANDIDATES = 12
            selected = ranked[:MAX_BLOCK_CANDIDATES]
            placement_records = [(trailer, records) for _, trailer, records in selected]
            component_labels.clear()
            for trailer, records in placement_records:
                component_labels[trailer.component_id].update(record.label for record in records)
    else:
        placements_from_blocks = False

    block_index_maps: dict[int, dict[int, int]] = {}
    index_block_lists: dict[int, list[int]] = {}
    for definition in definitions.values():
        idx_map: dict[int, int] = {}
        needle = b"\x15\xcd\x5b\x07"
        for blk in definition.sub_blocks:
            if getattr(blk, "tag", None) != 0x4803:
                continue
            pos = blk.payload.find(needle)
            if pos == -1 or pos + 8 > len(blk.payload):
                continue
            idx_val = struct.unpack("<I", blk.payload[pos + 4 : pos + 8])[0]
            idx_map.setdefault(idx_val, blk.offset)
        if idx_map:
            offset_to_ord = {blk.offset: i for i, blk in enumerate(definition.sub_blocks) if getattr(blk, "tag", None) == 0x4803}
            resolved = {idx_val: offset_to_ord[offset] for idx_val, offset in idx_map.items() if offset in offset_to_ord}
            if resolved:
                block_index_maps[definition.component_id] = resolved
                for idx_val, offset in idx_map.items():
                    ord_val = resolved.get(idx_val)
                    if ord_val is not None:
                        index_block_lists.setdefault(idx_val, []).append(ord_val)
    for k, v in list(index_block_lists.items()):
        seen = set()
        ordered = []
        for ord_val in sorted(v):
            if ord_val in seen:
                continue
            seen.add(ord_val)
            ordered.append(ord_val)
        index_block_lists[k] = ordered

    MAX_TOTAL_RECORDS = 5000
    MAX_PER_LABEL = 5000

    for placement in placements:
        glyph_records = extract_glyph_records(placement.payload, allow_spaces=True, max_label_len=80)
        if not glyph_records:
            continue
        if len(glyph_records) > MAX_TOTAL_RECORDS:
            pass
        placement_records.append((placement, glyph_records))
        component_labels[placement.component_id].update(record.label for record in glyph_records)

    if not placement_records:
        return [], [], []

    only_definition: ComponentDefinition | None = None
    if len(definitions) == 1:
        only_definition = next(iter(definitions.values()))

    label_maps: dict[int, dict[str, ComponentSubBlock]] = {}
    glyph_cache: dict[Tuple[int, int], Tuple[
        List[Tuple[Tuple[float, float], Tuple[float, float]]],
        List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]],
        float,
    ]] = {}
    new_lines: List[LineEntity] = []
    new_arcs: List[ArcEntity] = []
    new_inserts: List[InsertEntity] = []

    total_records = sum(len(grs) for _, grs in placement_records)
    arc_guess = force_arc_guess or (not placements_from_blocks and total_records <= 500)
    arc_limit = 20_000 if force_arc_guess else 10_000
    line_cap = 200_000
    arc_cap = arc_limit

    all_labels = set().union(*component_labels.values()) if component_labels else set()

    for placement, glyph_records in placement_records:
        deduped: dict[Tuple[str, Tuple[int, int, int, int, int]], GlyphPlacementRecord] = {}
        for rec in glyph_records:
            tx, ty, rot, sx, sy = rec.values
            key = (
                rec.label,
                (
                    round(tx, 4),
                    round(ty, 4),
                    round(rot, 4),
                    round(sx, 4),
                    round(sy, 4),
                ),
            )
            deduped.setdefault(key, rec)
        glyph_records = list(deduped.values())
        per_label: dict[str, list[GlyphPlacementRecord]] = defaultdict(list)
        for rec in glyph_records:
            if len(per_label[rec.label]) < MAX_PER_LABEL:
                per_label[rec.label].append(rec)
        glyph_records = [rec for records in per_label.values() for rec in records]
        definition = definitions.get(placement.component_id)
        if not definition and only_definition is not None:
            definition = only_definition
        if not definition:
            continue
        label_map = label_maps.get(definition.component_id)
        if label_map is None:
            labels_for_def = component_labels.get(placement.component_id, set()) or component_labels.get(
                definition.component_id, set()
            ) or all_labels
            block_index_map = block_index_maps.get(definition.component_id)
            label_map = build_label_block_map(
                definition,
                labels_for_def,
                catalog_map=catalog_map,
                block_index_map=block_index_map,
                index_block_lists=index_block_lists,
                index_label_order=index_label_order,
            )
            label_maps[definition.component_id] = label_map
        if not label_map:
            continue
        allowed_labels = set(label_map.keys())
        for record in glyph_records:
            if allowed_labels and record.label not in allowed_labels:
                continue
            block = label_map.get(record.label)
            if not block:
                continue
            cache_key = (definition.component_id, block.offset)
            cached = glyph_cache.get(cache_key)
            if cached is None:
                cached = segments_from_component_block(
                    block,
                    record.label,
                    allow_arc_guess=arc_guess,
                    bbox=getattr(definition, "bbox", None),
                )
                glyph_cache[cache_key] = cached
            segments, arcs, base_scale = cached
            if not segments and not arcs:
                continue
            def_bbox = getattr(definition, "bbox", None)
            if def_bbox:
                min_x, min_y, max_x, max_y = def_bbox
                span = max(max_x - min_x, max_y - min_y, 1e-3)
                tol = max(span * 0.1, 1e-3)
                corners = [
                    (min_x, min_y),
                    (min_x, max_y),
                    (max_x, min_y),
                    (max_x, max_y),
                ]
                tx, ty, rot_deg, sx, sy = record.values
                rotation_rad = math.radians(rot_deg) if abs(rot_deg) > 1e-9 else 0.0
                cos_theta = math.cos(rotation_rad) if rotation_rad else 1.0
                sin_theta = math.sin(rotation_rad) if rotation_rad else 0.0

                def _xf(pt):
                    return transform_point(pt, sx, sy, cos_theta, sin_theta, tx, ty)

                ws_corners = [_xf(pt) for pt in corners]
                ws_min_x = min(pt[0] for pt in ws_corners)
                ws_max_x = max(pt[0] for pt in ws_corners)
                ws_min_y = min(pt[1] for pt in ws_corners)
                ws_max_y = max(pt[1] for pt in ws_corners)
                ws_bbox = (ws_min_x - tol, ws_min_y - tol, ws_max_x + tol, ws_max_y + tol)
            else:
                ws_bbox = None

            layer_hint = 0
            if getattr(record, "field2", 0):
                layer_hint = max(0, int(record.field2) // 256 - 1)
            instantiated_lines: list[LineEntity] = []
            if segments:
                instantiated_lines = instantiate_glyph_segments(
                    segments,
                    record.values,
                    base_scale=base_scale,
                    layer=layer_hint,
                )
                if ws_bbox:
                    instantiated_lines = [
                        ln
                        for ln in instantiated_lines
                        if _point_in_bbox(ln.start, ws_bbox) and _point_in_bbox(ln.end, ws_bbox)
                    ]
            if instantiated_lines:
                room = max(0, line_cap - len(new_lines))
                instantiated_lines = instantiated_lines[:room]
                new_lines.extend(instantiated_lines)
            if arcs:
                instantiated_arcs = instantiate_glyph_arcs(
                    arcs,
                    record.values,
                    base_scale=base_scale,
                    layer=layer_hint,
                )
                if ws_bbox:
                    instantiated_arcs = [
                        arc
                        for arc in instantiated_arcs
                        if _point_in_bbox(arc.start, ws_bbox) and _point_in_bbox(arc.end, ws_bbox)
                    ]
                if instantiated_arcs:
                    room = max(0, arc_limit - len(new_arcs))
                    if room <= 0:
                        instantiated_arcs = []
                    else:
                        instantiated_arcs = instantiated_arcs[:room]
                if instantiated_arcs:
                    room = max(0, arc_cap - len(new_arcs))
                    instantiated_arcs = instantiated_arcs[:room]
                new_arcs.extend(instantiated_arcs)

    return new_lines, new_inserts, new_arcs


__all__ = [
    "collect_candidate_records",
    "build_label_block_map",
    "decode_polyline_block",
    "segments_from_component_block",
    "transform_point",
    "instantiate_glyph_segments",
    "instantiate_glyph_arcs",
    "extract_new_style_component_lines",
]
