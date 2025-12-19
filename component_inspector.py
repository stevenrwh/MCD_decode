#!/usr/bin/env python3
"""
Inspect TLV blocks that embed CComponentDefinition payloads inside a Monu-CAD
.mcd/.mcc file (or their already-deflated *.decompressed equivalents).

Example usage:

    python component_inspector.py MCD_CONTAINS_FACE_COMPONENT.decompressed \
        --json FACE_component_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

from component_parser import (
    ComponentDefinition,
    CirclePrimitive,
    extract_circle_primitives,
    iter_component_definitions,
    iter_component_placements,
)

try:
    from mcd_to_dxf import brute_force_deflate
except ImportError:  # pragma: no cover - optional dependency when only inspecting blobs
    brute_force_deflate = None  # type: ignore[assignment]


def load_payload(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix == ".decompressed" or brute_force_deflate is None:
        return data
    try:
        _, payload = brute_force_deflate(data)
    except Exception as exc:  # pragma: no cover - interactive aid
        raise SystemExit(f"Unable to locate deflate stream inside {path}: {exc}") from exc
    return payload


def _summarize_circles(circles: Sequence[CirclePrimitive]) -> list[dict]:
    return [
        {
            "center": [circle.center[0], circle.center[1]],
            "radius": circle.radius,
            "rim": [circle.rim[0], circle.rim[1]],
        }
        for circle in circles
    ]


def summarize_definition(definition: ComponentDefinition) -> dict:
    circles = extract_circle_primitives(definition)
    return {
        "offset": definition.offset,
        "length": definition.length,
        "component_id": definition.component_id,
        "bbox": list(definition.bbox),
        "header_ints": list(definition.header_values),
        "sub_blocks": [
            {
                "tag": hex(block.tag),
                "dtype": block.dtype,
                "count": block.count,
                "payload_size": len(block.payload),
            }
            for block in definition.sub_blocks
        ],
        "circles": _summarize_circles(circles),
    }


def summarize_placement(placement) -> dict:
    return {
        "name": placement.name,
        "component_id": placement.component_id,
        "instance_id": placement.instance_id,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize component TLVs inside a Monu-CAD payload.")
    parser.add_argument("input", type=Path, help=".mcd/.mcc/.decompressed blob to inspect")
    parser.add_argument("--json", type=Path, help="Optional path to write a JSON summary")
    parser.add_argument(
        "--instances",
        action="store_true",
        help="Include FACE (and other) placement trailers in the JSON summary",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = load_payload(args.input)
    definitions = list(iter_component_definitions(payload))
    if not definitions:
        print("No CComponentDefinition payloads were detected.")
    else:
        print(f"Discovered {len(definitions)} component block(s):")
        for definition in definitions:
            summary = summarize_definition(definition)
            print(
                f"  off=0x{summary['offset']:X} len={summary['length']} "
                f"id=0x{summary['component_id']:08X} "
                f"bbox={summary['bbox']} "
                f"circles={len(summary['circles'])}"
            )
    instances: list[dict] = []
    if args.instances:
        placements = list(iter_component_placements(payload))
        instances = [summarize_placement(item) for item in placements]
        if instances:
            print(f"\nDetected {len(instances)} placement trailer(s):")
            for inst in instances:
                print(f"  name={inst['name']} component_id=0x{inst['component_id']:08X}")
    if args.json:
        summary = {
            "source": str(args.input),
            "component_blocks": [summarize_definition(defn) for defn in definitions],
            "instance_blocks": instances if args.instances else [],
        }
        args.json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"\nSummary written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
