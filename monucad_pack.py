#!/usr/bin/env python3
"""
Pack/unpack Monu-CAD `.mcd` / `.mcc` payloads.

This lets us edit the deflated payloads (or craft new ones) in plain files,
then wrap them back into the fake-gzip container expected by MCPro9.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path
from typing import Any, Dict, List, Sequence

from mcd_to_dxf import brute_force_deflate

FAKE_GZIP_HEADER = bytes.fromhex("1F 8B 08 00 00 00 00 00 00 0B")
DEFAULT_BANNER = b"MCD2\x00\x00Created with MONU-CAD Pro 9.2.10\x00"
DEFAULT_PAYLOAD_OFFSET = 0x45
HEADER_TABLE_BYTES = 0x100


def _build_stub(offset: int = DEFAULT_PAYLOAD_OFFSET, banner: bytes = DEFAULT_BANNER) -> bytes:
    stub = FAKE_GZIP_HEADER + banner
    if len(stub) < offset:
        stub += b"\x00" * (offset - len(stub))
    return stub


def pack_payload(payload: bytes, *, offset: int = DEFAULT_PAYLOAD_OFFSET, banner: bytes = DEFAULT_BANNER) -> bytes:
    stub = _build_stub(offset=offset, banner=banner)
    compressor = zlib.compressobj(level=9, wbits=-15)
    compressed = compressor.compress(payload) + compressor.flush()
    return stub + compressed


def unpack_payload(blob: bytes) -> tuple[int, bytes]:
    offset, payload = brute_force_deflate(blob)
    return offset, payload


def _normalize_config_text(text: str | None, lines: Sequence[str] | None) -> bytes:
    if text is None and lines:
        text = "\n".join(lines)
    if text is None:
        text = ""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    if normalized and not normalized.endswith("\n"):
        normalized += "\n"
    normalized = normalized.replace("\n", "\r\n")
    return normalized.encode("ascii", errors="ignore") + b"\x00\x00"


def _load_section_header_from_file(source: Path) -> Dict[str, Any]:
    data = json.loads(source.read_text(encoding="utf-8"))
    header = data.get("section_header")
    if isinstance(header, dict):
        return header
    return data


def _section_header_bytes(header_spec: Dict[str, Any]) -> tuple[bytes, bytes]:
    arrays = header_spec.get("arrays", [])
    header_blob = bytearray()
    for array in arrays:
        for value in array:
            header_blob.extend(struct.pack("<I", int(value) & 0xFFFFFFFF))
    table_tail = header_spec.get("table_tail", "")
    if table_tail:
        header_blob.extend(bytes.fromhex(table_tail))
    if len(header_blob) < HEADER_TABLE_BYTES:
        header_blob.extend(b"\x00" * (HEADER_TABLE_BYTES - len(header_blob)))
    metadata_hex = header_spec.get("metadata", "")
    metadata = bytes.fromhex(metadata_hex) if metadata_hex else b""
    return bytes(header_blob[:HEADER_TABLE_BYTES]), metadata


def _resolve_section_header_spec(
    spec: Dict[str, Any],
    args: argparse.Namespace,
    spec_root: Path,
) -> Dict[str, Any] | None:
    if args.section_header:
        return _load_section_header_from_file(args.section_header)
    header_spec = spec.get("section_header")
    if isinstance(header_spec, dict):
        return header_spec
    header_path = spec.get("section_header_path") or spec.get("section_header_file")
    if header_path:
        path = Path(header_path)
        if not path.is_absolute():
            path = spec_root / path
        return _load_section_header_from_file(path)
    return None


def _pack_record(layer: int, etype: int, coords: Sequence[float]) -> bytes:
    if len(coords) != 4:
        raise ValueError("record coordinates must contain four values (x1, y1, x2, y2)")
    return struct.pack("<IIdddd", layer, etype, float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))


def build_payload_from_spec(
    spec: Dict[str, Any],
    *,
    section_header: Dict[str, Any] | None = None,
) -> bytes:
    config_blob = _normalize_config_text(spec.get("config_text"), spec.get("config_lines"))

    body = bytearray()
    for entry in spec.get("lines", []):
        layer = int(entry.get("layer", 0))
        start = entry.get("start")
        end = entry.get("end")
        if start is None or end is None:
            raise ValueError("line entries must provide 'start' and 'end'")
        coords = (float(start[0]), float(start[1]), float(end[0]), float(end[1]))
        body.extend(_pack_record(layer, 2, coords))

    for entry in spec.get("records", []):
        layer = int(entry.get("layer", 0))
        etype = int(entry.get("etype", 0))
        coords = entry.get("points")
        if coords is None:
            raise ValueError("record entries must include 'points'")
        body.extend(_pack_record(layer, etype, coords))

    append_hex = spec.get("append_hex")
    if append_hex:
        body.extend(bytes.fromhex(append_hex))

    payload = config_blob + bytes(body)
    if section_header:
        header_blob, metadata = _section_header_bytes(section_header)
        return header_blob + metadata + payload
    return payload


def _read_payload(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix.lower() == ".decompressed":
        return data
    _, payload = unpack_payload(data)
    return payload


def pack_command(args: argparse.Namespace) -> int:
    payload = _read_payload(args.payload)
    banner = args.banner.encode("ascii") + b"\x00" if args.banner else DEFAULT_BANNER
    blob = pack_payload(payload, offset=args.offset, banner=banner)
    args.output.write_bytes(blob)
    print(f"[+] Packed payload ({len(payload)} bytes) into {args.output} (offset=0x{args.offset:X})")
    return 0


def unpack_command(args: argparse.Namespace) -> int:
    blob = args.input.read_bytes()
    offset, payload = unpack_payload(blob)
    output = args.output or args.input.with_suffix(".decompressed")
    output.write_bytes(payload)
    print(f"[+] Extracted deflate payload ({len(payload)} bytes) at 0x{offset:X} -> {output}")
    return 0


def build_command(args: argparse.Namespace) -> int:
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    spec_path = args.spec.resolve()
    header_spec = _resolve_section_header_spec(spec, args, spec_path.parent)
    payload = build_payload_from_spec(spec, section_header=header_spec)
    if args.payload_out:
        args.payload_out.write_bytes(payload)
        print(f"[+] Payload bytes written to {args.payload_out}")
    banner = spec.get("banner")
    banner_bytes = (banner.encode("ascii") + b"\x00") if isinstance(banner, str) else DEFAULT_BANNER
    offset = spec.get("offset", args.offset)
    if args.output:
        blob = pack_payload(payload, offset=offset, banner=banner_bytes)
        args.output.write_bytes(blob)
        print(f"[+] Packed JSON spec into {args.output} (payload {len(payload)} bytes, offset=0x{offset:X})")
    elif not args.payload_out:
        print(f"[i] Built payload of {len(payload)} bytes (no output file requested)")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack/unpack Monu-CAD payloads.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_p = subparsers.add_parser("pack", help="Wrap a .decompressed payload into a fake-gzip container.")
    pack_p.add_argument("payload", type=Path, help="Source .decompressed payload or existing .mcd/.mcc")
    pack_p.add_argument("-o", "--output", required=True, type=Path, help="Destination .mcd/.mcc path")
    pack_p.add_argument("--offset", type=lambda v: int(v, 0), default=DEFAULT_PAYLOAD_OFFSET, help="Byte offset where the deflate stream should start (default 0x45)")
    pack_p.add_argument("--banner", help="Optional ASCII banner to embed between the fake gzip header and payload")
    pack_p.set_defaults(func=pack_command)

    unpack_p = subparsers.add_parser("unpack", help="Extract the hidden deflate payload from a .mcd/.mcc")
    unpack_p.add_argument("input", type=Path, help="Source .mcd/.mcc file")
    unpack_p.add_argument("-o", "--output", type=Path, help="Destination .decompressed path (defaults to <input>.decompressed)")
    unpack_p.set_defaults(func=unpack_command)

    build_p = subparsers.add_parser("build", help="Generate a payload from a JSON spec and optionally pack it.")
    build_p.add_argument("spec", type=Path, help="Path to the JSON spec describing config + records")
    build_p.add_argument("-o", "--output", type=Path, help="Optional destination .mcd/.mcc file")
    build_p.add_argument("--payload-out", type=Path, help="Optional destination for the raw .decompressed payload")
    build_p.add_argument("--offset", type=lambda v: int(v, 0), default=DEFAULT_PAYLOAD_OFFSET, help="Deflate offset when packing (default 0x45)")
    build_p.add_argument("--section-header", type=Path, help="JSON file describing the 7×9 header table (arrays/tail/metadata)")
    build_p.set_defaults(func=build_command)

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
