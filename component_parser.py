#!/usr/bin/env python3
"""
Utilities for parsing the embedded CComponentDefinition blobs that Monu-CAD
stores inside .mcd/.mcc payloads.  The helpers here are intentionally
lightweight so they can be reused by both mcd_to_dxf.py and the standalone
component_inspector.py tool.
"""

from __future__ import annotations

import collections
import math
import struct
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Sequence

COMPONENT_MARKER = b"CComponentDefinition"
KNOWN_BLOCK_TAGS = (0x3805, 0x4803)
MAX_COORD = 1e5
TLV_STRING = 0xFFFF
TLV_CLASS_LABELS = (b"CLine", b"CArc", b"CCircle")


@dataclass(frozen=True)
class ComponentSubBlock:
    tag: int
    dtype: int
    count: int
    payload: bytes
    offset: int


@dataclass(frozen=True)
class ComponentDefinition:
    offset: int
    length: int
    component_id: int
    bbox: tuple[float, float, float, float]
    header_values: tuple[int, ...]
    sub_blocks: tuple[ComponentSubBlock, ...]
    raw_payload: bytes
    tlv_primitives: tuple[TLVPrimitive, ...] = ()


@dataclass(frozen=True)
class CirclePrimitive:
    center: tuple[float, float]
    radius: float
    rim: tuple[float, float]


@dataclass(frozen=True)
class ComponentPlacement:
    name: str
    component_id: int
    instance_id: int


def _find_all_indices(blob: bytes, needle: bytes) -> Iterator[int]:
    start = 0
    while True:
        idx = blob.find(needle, start)
        if idx == -1:
            break
        yield idx
        start = idx + len(needle)


def _parse_sub_blocks(body: bytes) -> list[ComponentSubBlock]:
    blocks: list[ComponentSubBlock] = []
    idx = 0
    tag_bytes = [struct.pack("<H", tag) for tag in KNOWN_BLOCK_TAGS]

    while idx + 2 <= len(body):
        tag = struct.unpack_from("<H", body, idx)[0]
        if tag not in KNOWN_BLOCK_TAGS:
            idx += 1
            continue
        if idx + 12 > len(body):
            break
        dtype, count, declared_size = struct.unpack_from("<HII", body, idx + 2)
        data_start = idx + 12
        next_idx = len(body)
        for needle in tag_bytes:
            pos = body.find(needle, data_start)
            if pos != -1 and pos < next_idx:
                next_idx = pos
        payload = body[data_start:next_idx]
        blocks.append(
            ComponentSubBlock(
                tag=tag,
                dtype=dtype,
                count=count,
                payload=payload,
                offset=idx,
            )
        )
        if next_idx <= idx:
            break
        idx = next_idx
    return blocks

def _iter_tlv_fields(blob: bytes, *, start: int = 0) -> Iterator[tuple[int, int, int, int, bytes]]:
    """Generic TLV iterator used to decode the older component structures."""

    offset = start
    while offset + 8 <= len(blob):
        tag, dtype, count, size = struct.unpack_from("<HHHH", blob, offset)
        payload_start = offset + 8
        payload_end = payload_start + size
        if payload_end > len(blob) or size > len(blob):
            offset += 2
            continue
        payload = blob[payload_start:payload_end]
        yield offset, tag, dtype, count, size, payload
        offset = payload_end


@dataclass(frozen=True)
class TLVPrimitive:
    kind: str
    values: tuple[float, ...]


def _parse_tlv_primitives(blob: bytes) -> List[TLVPrimitive]:
    primitives: List[TLVPrimitive] = []

    last_label: str | None = None
    for offset, tag, dtype, count, size, payload in _iter_tlv_fields(blob):
        if dtype == TLV_STRING and payload:
            try:
                last_label = payload.decode("ascii")
            except UnicodeDecodeError:
                last_label = None
            continue
        if not last_label:
            continue
        # Coordinates are typically emitted as doubles (size divisible by 8).
        if size % 8 != 0:
            last_label = None
            continue
        doubles = struct.unpack("<{}d".format(size // 8), payload)
        primitives.append(TLVPrimitive(kind=last_label, values=doubles))
        last_label = None

    return primitives


def iter_label_chunks(blob: bytes, definition: ComponentDefinition) -> Iterator[tuple[str, bytes]]:
    """
    Walk the TLV-like class chunks embedded inside a component definition and
    yield (label, payload) pairs (e.g., ('CLine', <binary blob>)).
    """

    start = definition.offset
    end = definition.offset + definition.length
    data = blob[start:end]

    for label_bytes in TLV_CLASS_LABELS:
        search_pos = 0
        while True:
            pos = data.find(label_bytes, search_pos)
            if pos == -1:
                break
            payload = _extract_labeled_payload(data, pos, label_bytes)
            if payload:
                yield label_bytes.decode("ascii"), payload
            search_pos = pos + len(label_bytes)


def _extract_labeled_payload(data: bytes, label_pos: int, label: bytes) -> bytes | None:
    header_offset = label_pos - 8
    if header_offset < 0:
        return None
    try:
        _, dtype, _, size = struct.unpack_from("<HHHH", data, header_offset)
    except struct.error:
        return None
    if dtype != TLV_STRING or size != len(label):
        return None
    search = label_pos + len(label)
    while search + 8 <= len(data):
        try:
            _, _, _, payload_size = struct.unpack_from("<HHHH", data, search)
        except struct.error:
            return None
        payload_start = search + 8
        payload_end = payload_start + payload_size
        if payload_end > len(data):
            search += 2
            continue
        if payload_size == 0:
            search += 2
            continue
        return data[payload_start:payload_end]
    return None


def iter_component_definitions(blob: bytes) -> Iterable[ComponentDefinition]:
    marker_offsets = list(_find_all_indices(blob, COMPONENT_MARKER))
    if not marker_offsets:
        return
    total_len = len(blob)
    for idx, marker_offset in enumerate(marker_offsets):
        next_offset = marker_offsets[idx + 1] if idx + 1 < len(marker_offsets) else total_len
        data_start = marker_offset + len(COMPONENT_MARKER)
        data_end = max(data_start, next_offset)
        data = blob[data_start:data_end]
        if len(data) < 44:
            continue
        bbox = struct.unpack_from("<4f", data, 0)
        header_values = struct.unpack_from("<7I", data, 16)
        body = data[44:]
        sub_blocks = tuple(_parse_sub_blocks(body))
        tlv_primitives: tuple[TLVPrimitive, ...] = ()
        if not sub_blocks:
            tlv_primitives = tuple(_parse_tlv_primitives(body))
        yield ComponentDefinition(
            offset=marker_offset,
            length=data_end - marker_offset,
            component_id=header_values[2],
            bbox=tuple(float(v) for v in bbox),
            header_values=tuple(int(v) for v in header_values),
            sub_blocks=sub_blocks,
            raw_payload=body,
            tlv_primitives=tlv_primitives,
        )


def iter_component_placements(blob: bytes) -> Iterable[ComponentPlacement]:
    """
    Best-effort scanner that locates the length-prefixed component trailers
    (e.g., b\"\\x04FACE\").  The placement payload format is still evolving, so
    this helper currently returns a simplified view that exposes the component
    id and instance id for downstream wiring.
    """

    max_len = 32
    idx = 0
    while idx < len(blob):
        name_len = blob[idx]
        if not (1 <= name_len <= max_len):
            idx += 1
            continue
        end = idx + 1 + name_len
        if end + 4 > len(blob):
            break
        name_bytes = blob[idx + 1 : end]
        try:
            name = name_bytes.decode("ascii")
        except UnicodeDecodeError:
            idx += 1
            continue
        magic = struct.unpack_from("<I", blob, end)[0]
        if magic != 0x075BCD15:
            idx += 1
            continue
        if end + 12 > len(blob):
            break
        instance_id = struct.unpack_from("<I", blob, end + 4)[0]
        component_id = struct.unpack_from("<I", blob, end + 8)[0]
        yield ComponentPlacement(name=name, component_id=component_id, instance_id=instance_id)
        idx = end + 1


def extract_circle_primitives(definition: ComponentDefinition) -> list[CirclePrimitive]:
    """
    Decode the 0x0538 sub-blocks that store circle metadata.  Each block encodes
    a center (x, y) pair followed by a rim point (x, y) in local component
    coordinates.  Additional padding bytes are ignored.
    """

    circles: list[CirclePrimitive] = []
    circle_tag = 0x3805
    for block in definition.sub_blocks:
        if block.tag != circle_tag or not block.payload:
            continue
        payload = block.payload
        if len(payload) % 8 != 0:
            payload = payload[1:]
        usable = len(payload) // 8 * 8
        if usable < 32:
            continue
        doubles = struct.unpack_from("<{}d".format(usable // 8), payload)
        for idx in range(0, len(doubles) - 3, 4):
            cx, cy, px, py = doubles[idx : idx + 4]
            radius = math.hypot(px - cx, py - cy)
            if (
                radius <= 1e-6
                or radius > MAX_COORD
                or not all(math.isfinite(val) for val in (cx, cy, px, py))
                or any(abs(val) > MAX_COORD for val in (cx, cy, px, py))
            ):
                break
            circles.append(CirclePrimitive(center=(cx, cy), radius=radius, rim=(px, py)))
    return circles
