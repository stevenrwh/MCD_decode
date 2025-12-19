from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Iterator, List

MATRIX_PATTERN = b"\x00\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00"
MATRIX_CHUNK_SIZE = 192  # 48 uint32 slots -> 3x4 rows of float32 fixed-point
MATRIX_VALUE_SCALE = 1 / 2**32
VALUES_PADDING_BYTES = 12  # padding between the 5 doubles and the length byte


@dataclass(frozen=True)
class GlyphPlacementRecord:
    label: str
    values: tuple[float, float, float, float, float]
    component_id: int
    field1: int
    field2: int
    field3: int
    data_offset: int


@dataclass(frozen=True)
class PlacementTrailer:
    name: str
    component_id: int
    instance_id: int
    payload: bytes


def iter_placement_trailers(blob: bytes, *, max_label_len: int = 48) -> Iterator[PlacementTrailer]:
    """
    Walk the raw payload and yield each placement trailer discovered in the blob.
    """

    idx = 0
    length = len(blob)
    magic = 0x075BCD15
    while idx + 1 < length:
        name_len = blob[idx]
        if not (1 <= name_len <= max_label_len):
            idx += 1
            continue
        label_start = idx + 1
        label_end = label_start + name_len
        if label_end + 12 > length:
            break
        name_bytes = blob[label_start:label_end]
        try:
            name = name_bytes.decode("ascii")
        except UnicodeDecodeError:
            idx += 1
            continue
        trailer_magic = struct.unpack_from("<I", blob, label_end)[0]
        if trailer_magic != magic:
            idx += 1
            continue
        instance_id = struct.unpack_from("<I", blob, label_end + 4)[0]
        component_id = struct.unpack_from("<I", blob, label_end + 8)[0]
        payload_start = label_end + 12

        search = payload_start
        next_offset = None
        while search + 1 < length:
            cand_len = blob[search]
            if 1 <= cand_len <= max_label_len:
                cand_label_end = search + 1 + cand_len
                if cand_label_end + 12 > length:
                    break
                cand_bytes = blob[search + 1 : cand_label_end]
                try:
                    cand_bytes.decode("ascii")
                except UnicodeDecodeError:
                    search += 1
                    continue
                cand_magic = struct.unpack_from("<I", blob, cand_label_end)[0]
                if cand_magic == magic:
                    next_offset = search
                    break
            search += 1

        payload_end = next_offset if next_offset is not None else length
        payload = blob[payload_start:payload_end]
        if payload:
            # Skip obvious inline tables (e.g., fixed 71-byte entries with repeated labels and no sentinel)
            if payload.find(b"\x15\xcd\x5b\x07") == -1 and payload.count(b"\x00\x00\x00\x00") > 1000:
                idx = payload_end if payload_end > payload_start else payload_start + 1
                continue
            yield PlacementTrailer(
                name=name,
                component_id=component_id,
                instance_id=instance_id,
                payload=payload,
            )
        idx = payload_end if payload_end > payload_start else payload_start + 1


def extract_matrices(blob: bytes, *, max_matrices: int = 4) -> list[list[list[float]]]:
    idx = blob.find(MATRIX_PATTERN)
    if idx < 0:
        return []

    matrices: list[list[list[float]]] = []
    offset = idx + len(MATRIX_PATTERN)
    while offset + MATRIX_CHUNK_SIZE <= len(blob) and len(matrices) < max_matrices:
        chunk = blob[offset : offset + MATRIX_CHUNK_SIZE]
        values: List[float] = []
        for slot in range(0, MATRIX_CHUNK_SIZE, 4):
            raw = int.from_bytes(chunk[slot : slot + 4], "little", signed=False)
            values.append(raw * MATRIX_VALUE_SCALE)
        rows = [values[i : i + 4] for i in range(0, len(values), 4)]
        matrices.append(rows)
        offset += MATRIX_CHUNK_SIZE
    return matrices


def extract_glyph_records(
    blob: bytes,
    *,
    allow_spaces: bool = True,
    max_label_len: int = 80,
) -> list[GlyphPlacementRecord]:
    return _extract_glyph_records_internal(blob, allow_spaces=allow_spaces, max_label_len=max_label_len, return_offsets=False)


def extract_glyph_records_with_offsets(
    blob: bytes,
    *,
    allow_spaces: bool = True,
    max_label_len: int = 80,
) -> list[tuple[int, GlyphPlacementRecord]]:
    """
    Variant of ``extract_glyph_records`` that also returns the record start
    offset (relative to ``blob``) for diagnostics.
    """

    return _extract_glyph_records_internal(blob, allow_spaces=allow_spaces, max_label_len=max_label_len, return_offsets=True)


def _extract_glyph_records_internal(
    blob: bytes,
    *,
    allow_spaces: bool,
    max_label_len: int,
    return_offsets: bool,
) -> list[GlyphPlacementRecord] | list[tuple[int, GlyphPlacementRecord]]:
    """
    Extract placement records from the extra payload stored in a placement trailer.

    Empirically, records are laid out as:
      [5 float64 values][12 padding bytes][u8 label_len][label bytes][5 u32 footer]

    Where the 5 float64 values behave like ``(tx, ty, rot_deg, sx, sy)``.
    Labels are length-prefixed printable ASCII strings and may include spaces.
    """

    if len(blob) < 40 + VALUES_PADDING_BYTES + 2:
        return []

    # Heuristic: skip the fixed-stride catalog/table (e.g., 0x3805 table) that
    # lacks the placement sentinel and is packed with length-prefixed labels.
    if blob.find(b"\x15\xcd\x5b\x07") == -1:
        import re

        sample = blob[:1024]
        hits = [m.start() for m in re.finditer(rb"[A-Z0-9_]{3,8}", sample)]
        if len(hits) > 5:
            deltas = [hits[i + 1] - hits[i] for i in range(len(hits) - 1)]
            if deltas:
                common = max(set(deltas), key=deltas.count)
                if common in (70, 71, 72):
                    return []

    padding_sentinel = b"\x01" + (b"\x00" * (VALUES_PADDING_BYTES - 1))
    records: list[tuple[int, GlyphPlacementRecord]] = []
    seen_record_starts: set[int] = set()
    blob_len = len(blob)

    sentinel_positions = []
    search = 0
    while True:
        pos = blob.find(padding_sentinel, search)
        if pos == -1:
            break
        sentinel_positions.append(pos)
        search = pos + 1

    for sentinel_pos in sentinel_positions:
        record_start = sentinel_pos - 40
        if record_start < 0 or record_start in seen_record_starts:
            continue
        length_pos = sentinel_pos + VALUES_PADDING_BYTES  # offset of label_len byte
        if length_pos >= blob_len:
            continue
        label_len = blob[length_pos]
        if not (1 <= label_len <= max_label_len):
            continue
        label_start = length_pos + 1
        label_end = label_start + label_len
        if label_end > blob_len:
            continue
        label_bytes = blob[label_start:label_end]
        if not label_bytes.strip():
            continue
        if any(byte < 32 or byte > 126 for byte in label_bytes):
            continue
        if not allow_spaces and any(byte == 32 for byte in label_bytes):
            continue
        try:
            values = struct.unpack_from("<5d", blob, record_start)
        except struct.error:
            continue
        tx, ty, rot_deg, sx, sy = values
        if not all(math.isfinite(val) for val in values):
            continue
        if abs(tx) > 1e7 or abs(ty) > 1e7:
            continue
        if abs(rot_deg) > 1e5:
            continue
        if abs(sx) > 1e6 or abs(sy) > 1e6:
            continue
        if abs(sx) < 1e-12 and abs(sy) < 1e-12:
            continue
        footer_offset = label_end
        if footer_offset + 20 <= blob_len:
            footer = struct.unpack_from("<IIIII", blob, footer_offset)
        else:
            if footer_offset != blob_len:
                continue
            footer = (0, 0, 0, 0, 0)
        try:
            label = label_bytes.decode("ascii")
        except UnicodeDecodeError:
            continue
        record = GlyphPlacementRecord(
            label=label,
            values=values,
            component_id=footer[0],
            field1=footer[1],
            field2=footer[2],
            field3=footer[3],
            data_offset=footer[4],
        )
        records.append((record_start, record))
        seen_record_starts.add(record_start)

    records.sort(key=lambda item: item[0])
    if return_offsets:
        return records
    return [record for _, record in records]
