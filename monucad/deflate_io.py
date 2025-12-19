from __future__ import annotations

import zlib
from typing import List, Tuple

DEFAULT_MIN_PAYLOAD = 128


def collect_deflate_streams(
    blob: bytes,
    *,
    min_payload: int = DEFAULT_MIN_PAYLOAD,
    start_offset: int = 0,
    stop_offset: int | None = None,
) -> List[Tuple[int, bytes]]:
    """
    Identify every raw deflate stream embedded inside ``blob``. Many Monu-CAD
    resources (notably .fnt archives) pack multiple streams back-to-back, so we
    need to keep scanning after the first hit.
    """

    limit = len(blob) if stop_offset is None else min(stop_offset, len(blob))
    if start_offset < 0 or start_offset >= limit:
        raise ValueError("start_offset is outside the readable range")

    streams: List[Tuple[int, bytes]] = []
    seen_offsets: set[int] = set()
    overlap_window = 512
    offset = start_offset
    mv = memoryview(blob)
    while offset < limit:
        slice_ = mv[offset:limit]
        if not slice_:
            break
        obj = zlib.decompressobj(-zlib.MAX_WBITS)
        try:
            payload = obj.decompress(slice_)
        except zlib.error:
            offset += 1
            continue
        consumed = len(slice_) - len(obj.unused_data)
        if consumed <= 0:
            offset += 1
            continue
        if len(payload) >= min_payload and offset not in seen_offsets:
            streams.append((offset, payload))
            seen_offsets.add(offset)
        nested_start = offset + 1
        nested_limit = min(offset + overlap_window, offset + consumed, limit)
        while nested_start < nested_limit:
            if nested_start in seen_offsets:
                nested_start += 1
                continue
            nested_slice = mv[nested_start:limit]
            nested_obj = zlib.decompressobj(-zlib.MAX_WBITS)
            try:
                nested_payload = nested_obj.decompress(nested_slice)
            except zlib.error:
                nested_start += 1
                continue
            nested_consumed = len(nested_slice) - len(nested_obj.unused_data)
            if nested_consumed <= 0:
                nested_start += 1
                continue
            if len(nested_payload) >= min_payload:
                streams.append((nested_start, nested_payload))
                seen_offsets.add(nested_start)
            nested_start += 1
        offset += max(consumed, 1)
    return streams


def brute_force_deflate(
    blob: bytes,
    *,
    min_payload: int = DEFAULT_MIN_PAYLOAD,
    start_offset: int = 0,
    stop_offset: int | None = None,
) -> Tuple[int, bytes]:
    """
    Locate the most useful deflate stream embedded inside ``blob``. If multiple
    candidates exist we prefer the one with the largest decompressed size,
    which usually corresponds to the real geometry payload.
    """

    streams = collect_deflate_streams(
        blob,
        min_payload=min_payload,
        start_offset=start_offset,
        stop_offset=stop_offset,
    )
    if not streams:
        raise RuntimeError(
            "Unable to locate a deflate payload. "
            "Try passing --start-offset/--stop-offset to narrow the search window."
        )
    best_offset, best_payload = max(streams, key=lambda item: len(item[1]))
    return best_offset, best_payload
