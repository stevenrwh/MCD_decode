#!/usr/bin/env python3
"""
High-level inspector for Monu-CAD `.mcd` drawings.

This tool reuses the existing `mcd_to_dxf.py` + `component_parser.py` helpers to:
  * locate and decompress the hidden deflate payload
  * split the payload into the textual config prelude + binary drawing data
  * summarize recovered line/arc/circle entities
  * list embedded CComponentDefinition blocks and FACE/placement trailers

The output can be a pretty console report and/or a JSON blob so we can diff,
version, or feed the structure into other tooling without reopening MCPro9.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from component_parser import (
    ComponentDefinition,
    extract_circle_primitives,
    iter_component_definitions,
    iter_component_placements,
    COMPONENT_MARKER,
)
from mcd_to_dxf import brute_force_deflate, parse_entities
from glyph_tlv_parser import GlyphComponent, parse_glyph_components_from_blob


@dataclass
class GeometrySummary:
    line_count: int
    arc_count: int
    circle_count: int
    bbox: Tuple[float, float, float, float] | None


HEADER_MARKER = b"PolishTile = "
SECTION_COUNT = 9
ARRAY_COUNT = 7
TABLE_ARRAY_BYTES = SECTION_COUNT * ARRAY_COUNT * 4
DEFAULT_TABLE_BYTES = 0x100
DEFAULT_TABLE_TAIL = max(0, DEFAULT_TABLE_BYTES - TABLE_ARRAY_BYTES)


@dataclass(frozen=True)
class SectionHeader:
    section_count: int
    field1: int
    field2: int
    config_offset: int | None
    config_length: int | None
    table_bytes: int
    table_tail: bytes
    metadata_bytes: bytes
    arrays: Tuple[Tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if self.section_count <= 0:
            raise ValueError("section_count must be positive")

    @property
    def array_count(self) -> int:
        return len(self.arrays)

    @property
    def sections(self) -> Tuple[Tuple[int, ...], ...]:
        rows: List[Tuple[int, ...]] = []
        for sec_idx in range(self.section_count):
            row = tuple(row[sec_idx] for row in self.arrays if sec_idx < len(row))
            rows.append(row)
        return tuple(rows)

    def serialize_table(self) -> bytes:
        data = []
        for array in self.arrays:
            for value in array:
                data.append(value.to_bytes(4, "little"))
        return b"".join(data) + self.table_tail

    def to_dict(self) -> Dict:
        return {
            "section_count": self.section_count,
            "field1": self.field1,
            "field2": self.field2,
            "config_offset": self.config_offset,
            "config_length": self.config_length,
            "table_bytes": self.table_bytes,
            "array_count": self.array_count,
            "arrays": [list(array) for array in self.arrays],
            "sections": [list(section) for section in self.sections],
            "table_tail": self.table_tail.hex(),
            "metadata": self.metadata_bytes.hex(),
        }


def _split_payload(payload: bytes, header: SectionHeader | None) -> Tuple[str, bytes]:
    """
    Separate the textual configuration (INI-like region) from the binary data.
    Prefer offsets provided by `_parse_section_header`; fall back to the first
    double-NULL delimiter when the header cannot be located.
    """

    if header and header.config_offset is not None:
        start = header.config_offset
        length = header.config_length or 0
        text = payload[start : start + length].decode("ascii", errors="ignore").strip()
        return text, payload[start + length :]

    sentinel = b"\x00\x00"
    idx = payload.find(sentinel)
    if idx == -1:
        return payload.decode("ascii", errors="ignore"), b""
    config_blob = payload[:idx]
    binary_blob = payload[idx + len(sentinel) :]
    text = config_blob.decode("ascii", errors="ignore").strip()
    return text, binary_blob


def _load_payload(path: Path) -> Tuple[int, bytes]:
    blob = path.read_bytes()
    if path.suffix.lower() == ".decompressed":
        return 0, blob
    return brute_force_deflate(blob)


def _detect_payload_type(path: Path, payload: bytes, override: str) -> str:
    if override != "auto":
        return override
    suffixes = [token.lower() for token in path.suffixes]
    if ".fnt" in suffixes or ".dta" in suffixes:
        return "font"
    if ".mcc" in suffixes:
        return "component"
    if ".mcd" in suffixes:
        return "drawing"
    if HEADER_MARKER in payload[: DEFAULT_TABLE_BYTES * 2]:
        return "drawing"
    if COMPONENT_MARKER in payload[: DEFAULT_TABLE_BYTES * 2]:
        return "component"
    return "drawing"


def _print_font_preview(glyphs: Sequence["GlyphComponent"], limit: int) -> None:
    if not glyphs:
        print("\n[font] no glyph components parsed")
        return
    print(f"\n[font] glyphs={len(glyphs)} preview_limit={limit}")
    for glyph in glyphs[: max(limit, 0)]:
        bbox = glyph.bbox
        print(
            f"  {glyph.label:<12} segments={len(glyph.segments):3d} "
            f"baseline={glyph.baseline:+.4f} advance={glyph.advance:.4f} "
            f"bbox=({bbox[0]:+.4f},{bbox[1]:+.4f})-({bbox[2]:+.4f},{bbox[3]:+.4f})"
        )


def _summarize_geometry(payload: bytes) -> GeometrySummary:
    lines, arcs, circles, _inserts = parse_entities(payload)
    xs: List[float] = []
    ys: List[float] = []
    for line in lines:
        xs.extend((line.start[0], line.end[0]))
        ys.extend((line.start[1], line.end[1]))
    for arc in arcs:
        xs.extend((arc.center[0], arc.start[0], arc.end[0]))
        ys.extend((arc.center[1], arc.start[1], arc.end[1]))
    for circle in circles:
        xs.append(circle.center[0])
        ys.append(circle.center[1])
    bbox = None
    if xs and ys:
        bbox = (min(xs), min(ys), max(xs), max(ys))
    return GeometrySummary(
        line_count=len(lines),
        arc_count=len(arcs),
        circle_count=len(circles),
        bbox=bbox,
    )


def _summarize_component(defn: ComponentDefinition) -> Dict:
    circles = extract_circle_primitives(defn)
    return {
        "offset": defn.offset,
        "component_id": defn.component_id,
        "bbox": list(defn.bbox),
        "header_values": list(defn.header_values),
        "circle_count": len(circles),
        "sub_block_count": len(defn.sub_blocks),
    }


def _summarize_instances(payload: bytes) -> List[Dict]:
    return [
        {
            "name": placement.name,
            "component_id": placement.component_id,
            "instance_id": placement.instance_id,
        }
        for placement in iter_component_placements(payload)
    ]


def _parse_section_header(payload: bytes) -> SectionHeader | None:
    if len(payload) < TABLE_ARRAY_BYTES:
        return None

    table_bytes = min(DEFAULT_TABLE_BYTES, len(payload))
    array_blob = payload[:TABLE_ARRAY_BYTES]
    tail_end = min(table_bytes, len(payload))
    table_tail = payload[TABLE_ARRAY_BYTES:tail_end]

    marker_offset = payload.find(HEADER_MARKER)
    component_offset = payload.find(COMPONENT_MARKER)
    data_start = table_bytes
    if marker_offset != -1 and marker_offset >= table_bytes:
        data_start = marker_offset
    elif component_offset != -1 and component_offset >= table_bytes:
        data_start = component_offset
    metadata = payload[table_bytes:data_start]

    section_count = SECTION_COUNT
    field1 = 1
    field2 = 1
    if len(metadata) >= 4:
        candidate = int.from_bytes(metadata[0:4], "little")
        if 0 < candidate <= 64:
            section_count = candidate
    if len(metadata) >= 8:
        field1 = int.from_bytes(metadata[4:8], "little")
    if len(metadata) >= 12:
        field2 = int.from_bytes(metadata[8:12], "little")

    int_values = [
        int.from_bytes(array_blob[offset : offset + 4], "little")
        for offset in range(0, len(array_blob) - (len(array_blob) % 4), 4)
    ]
    if len(int_values) < section_count:
        return None

    array_count = max(1, len(int_values) // section_count)
    arrays = tuple(
        tuple(int_values[idx * section_count : idx * section_count + section_count])
        for idx in range(array_count)
    )

    config_length = None
    if marker_offset != -1 and marker_offset >= 2:
        config_length = (payload[marker_offset - 2] << 8) | payload[marker_offset - 1]

    return SectionHeader(
        section_count=section_count,
        field1=field1,
        field2=field2,
        config_offset=marker_offset if marker_offset != -1 else None,
        config_length=config_length,
        table_bytes=table_bytes,
        table_tail=table_tail,
        metadata_bytes=metadata,
        arrays=arrays,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect the hidden payload inside a .mcd file.")
    parser.add_argument(
        "input",
        type=Path,
        help="Path to a .mcd/.mcc/.fnt (or .decompressed) payload",
    )
    parser.add_argument("--json", type=Path, help="Optional JSON destination for structured output")
    parser.add_argument(
        "--type",
        choices=("auto", "drawing", "component", "font"),
        default="auto",
        help="Force how we interpret the payload (default: auto detect)",
    )
    parser.add_argument(
        "--glyph-limit",
        type=int,
        default=5,
        help="Font preview cap (applies to --type font or detection)",
    )
    parser.add_argument(
        "--instances",
        action="store_true",
        help="Include FACE/component placement trailers in the summary",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    offset, payload = _load_payload(args.input)
    payload_type = _detect_payload_type(args.input, payload, args.type)
    section_header = _parse_section_header(payload)
    text, binary_blob = _split_payload(payload, section_header)
    print(
        f"[+] Hidden deflate stream located at 0x{offset:X} ({len(payload)} bytes) "
        f"(type={payload_type})"
    )
    print(f"[+] Textual config bytes: {len(text.encode('ascii', errors='ignore'))}")
    print(f"[+] Binary payload bytes: {len(binary_blob)}")
    if text and payload_type == "drawing":
        preview = "\n".join(line for line in text.splitlines()[:10])
        print("\n[config preview]\n" + preview)

    if section_header:
        config_off = (
            f"0x{section_header.config_offset:X}"
            if section_header.config_offset is not None
            else "N/A"
        )
        config_len = section_header.config_length if section_header.config_length else 0
        print(
            "\n[header] sections="
            f"{section_header.section_count} "
            f"field1={section_header.field1} "
            f"field2={section_header.field2} "
            f"config_len={config_len} "
            f"config_off={config_off} "
            f"arrays={section_header.array_count} "
            f"table_bytes=0x{section_header.table_bytes:X} "
            f"tail={len(section_header.table_tail)} "
            f"meta={len(section_header.metadata_bytes)}"
        )
    else:
        print("\n[header] unable to locate section/INI metadata marker")

    geom: GeometrySummary | None = None
    if payload_type == "drawing":
        geom = _summarize_geometry(payload)
        print(
            f"\n[geometry] lines={geom.line_count} arcs={geom.arc_count} "
            f"circles={geom.circle_count} bbox={geom.bbox}"
        )

    comp_defs = list(iter_component_definitions(payload))
    if comp_defs:
        print(f"\n[components] detected {len(comp_defs)} definition block(s)")
        if payload_type != "font":
            for idx, defn in enumerate(comp_defs, start=1):
                summary = _summarize_component(defn)
                print(
                    f"  #{idx}: id=0x{summary['component_id']:08X} "
                    f"off=0x{summary['offset']:X} circles={summary['circle_count']} "
                    f"sub_blocks={summary['sub_block_count']}"
                )
        else:
            print("  (component details suppressed for font payloads)")
    else:
        print("\n[components] no embedded definitions detected")

    instances: List[Dict] = []
    if args.instances:
        instances = _summarize_instances(payload)
        if instances:
            print(f"\n[instances] detected {len(instances)} placement trailer(s)")
            for inst in instances[:10]:
                print(
                    f"  name={inst['name']} component=0x{inst['component_id']:08X} "
                    f"instance=0x{inst['instance_id']:08X}"
                )
        else:
            print("\n[instances] no placement trailers detected")

    glyphs: List[GlyphComponent] = []
    if payload_type == "font":
        glyphs = parse_glyph_components_from_blob(payload)
        glyphs_sorted = sorted(glyphs, key=lambda g: g.label)
        _print_font_preview(glyphs_sorted, args.glyph_limit)
    else:
        glyphs_sorted = []

    if args.json:
        summary = {
            "input": str(args.input),
            "payload_type": payload_type,
            "deflate_offset": offset,
            "config_text": text,
            "geometry": asdict(geom) if geom else None,
            "components": [_summarize_component(defn) for defn in comp_defs],
            "instances": instances if args.instances else [],
            "section_header": section_header.to_dict() if section_header else None,
            "glyphs": [glyph.to_dict() for glyph in glyphs_sorted],
        }
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\n[+] JSON summary written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
