#!/usr/bin/env python3
"""
Render MONU-CAD .mcd/.mcc geometry to PNG previews without relying on CAD.

The script reuses mcd_to_dxf's parser so the geometry matches the DXF output,
then rasterizes the result with Pillow.  Example:

    python render_component_png.py FACE.mcc \
        --preview FACE_thumb.png --preview-size 256 \
        --hires FACE_full.png --hires-size 2048
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Sequence, Tuple

from PIL import Image, ImageDraw

from mcd_to_dxf import (
    ArcEntity,
    CircleEntity,
    LineEntity,
    brute_force_deflate,
    parse_entities,
)


def _load_payload(path: Path) -> bytes:
    data = path.read_bytes()
    if path.suffix == ".decompressed":
        return data
    _, payload = brute_force_deflate(data)
    return payload


def load_geometry(path: Path) -> Tuple[Sequence[LineEntity], Sequence[ArcEntity], Sequence[CircleEntity]]:
    payload = _load_payload(path)
    lines, arcs, circles, _inserts = parse_entities(payload)
    if not (lines or arcs or circles):
        raise RuntimeError("No renderable entities were discovered inside the payload.")
    return lines, arcs, circles


def _angle(center: Tuple[float, float], point: Tuple[float, float]) -> float:
    return math.atan2(point[1] - center[1], point[0] - center[0])


def _normalize_angle(angle: float) -> float:
    two_pi = math.tau if hasattr(math, "tau") else 2 * math.pi
    return angle % two_pi


def _sample_arc_points(arc: ArcEntity) -> list[Tuple[float, float]]:
    center = arc.center
    radius = math.hypot(arc.start[0] - center[0], arc.start[1] - center[1])
    if radius <= 0:
        return []

    theta_start = _normalize_angle(_angle(center, arc.start))
    theta_end = _normalize_angle(_angle(center, arc.end))
    two_pi = math.tau if hasattr(math, "tau") else 2 * math.pi
    sweep = (theta_end - theta_start) % two_pi
    if math.isclose(sweep, 0.0):
        sweep = two_pi

    segments = max(16, int(sweep / (math.pi / 64)))
    points: list[Tuple[float, float]] = []
    for step in range(segments + 1):
        t = step / segments
        angle = theta_start + t * sweep
        points.append(
            (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
            )
        )
    return points


def _sample_circle_points(circle: CircleEntity, segments: int = 128) -> list[Tuple[float, float]]:
    if circle.radius <= 0:
        return []
    points: list[Tuple[float, float]] = []
    for step in range(segments):
        angle = 2 * math.pi * step / segments
        points.append(
            (
                circle.center[0] + circle.radius * math.cos(angle),
                circle.center[1] + circle.radius * math.sin(angle),
            )
        )
    return points


def _trim_range(values: Sequence[float], trim_ratio: float = 0.005) -> Tuple[float, float]:
    if not values:
        raise RuntimeError("No values available for trimming.")
    raw_min = min(values)
    raw_max = max(values)
    if len(values) < 20:
        return raw_min, raw_max
    sorted_vals = sorted(values)
    cut = int(len(sorted_vals) * trim_ratio)
    if cut == 0 or cut >= len(sorted_vals) // 2:
        return raw_min, raw_max
    trimmed_min = sorted_vals[cut]
    trimmed_max = sorted_vals[-cut - 1]
    trimmed_range = trimmed_max - trimmed_min
    raw_range = raw_max - raw_min
    if raw_range <= 0 or trimmed_range <= 0:
        return raw_min, raw_max
    if trimmed_range <= raw_range * 0.9:
        buffer = (raw_range - trimmed_range) * 0.05
        return trimmed_min - buffer, trimmed_max + buffer
    return raw_min, raw_max


def _collect_bounds(
    lines: Sequence[LineEntity],
    arcs: Sequence[ArcEntity],
    circles: Sequence[CircleEntity],
) -> Tuple[float, float, float, float]:
    points: list[Tuple[float, float]] = []
    for line in lines:
        points.append(line.start)
        points.append(line.end)
    for arc in arcs:
        points.extend(_sample_arc_points(arc))
    for circle in circles:
        points.extend(_sample_circle_points(circle))
    if not points:
        raise RuntimeError("Unable to compute bounds for the requested geometry.")
    xs = [pt[0] for pt in points]
    ys = [pt[1] for pt in points]
    min_x, max_x = _trim_range(xs)
    min_y, max_y = _trim_range(ys)
    return min_x, max_x, min_y, max_y


def _build_transform(
    bounds: Tuple[float, float, float, float],
    size_px: int,
    padding_ratio: float,
):
    min_x, max_x, min_y, max_y = bounds
    width = max(max_x - min_x, 1e-9)
    height = max(max_y - min_y, 1e-9)
    pad = max(width, height) * padding_ratio

    world_min_x = min_x - pad
    world_max_x = max_x + pad
    world_min_y = min_y - pad
    world_max_y = max_y + pad

    world_width = world_max_x - world_min_x
    world_height = world_max_y - world_min_y

    scale = min(size_px / world_width, size_px / world_height)
    offset_x = (size_px - world_width * scale) / 2.0
    offset_y = (size_px - world_height * scale) / 2.0

    def transform(point: Tuple[float, float]) -> Tuple[float, float]:
        x, y = point
        px = (x - world_min_x) * scale + offset_x
        py = size_px - ((y - world_min_y) * scale + offset_y)
        return px, py

    return transform, scale


def render_png(
    lines: Sequence[LineEntity],
    arcs: Sequence[ArcEntity],
    circles: Sequence[CircleEntity],
    destination: Path,
    size_px: int,
    *,
    padding_ratio: float = 0.05,
) -> None:
    bounds = _collect_bounds(lines, arcs, circles)
    transform, scale = _build_transform(bounds, size_px, padding_ratio)

    image = Image.new("RGBA", (size_px, size_px), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    stroke = max(1, int(size_px / 256))

    for line in lines:
        start = transform(line.start)
        end = transform(line.end)
        draw.line([start, end], fill="black", width=stroke)

    for arc in arcs:
        points = _sample_arc_points(arc)
        if len(points) < 2:
            continue
        transformed = [transform(pt) for pt in points]
        draw.line(transformed, fill="black", width=stroke)

    for circle in circles:
        if circle.radius <= 0:
            continue
        center_px = transform(circle.center)
        radius_px = circle.radius * scale
        bbox = [
            center_px[0] - radius_px,
            center_px[1] - radius_px,
            center_px[0] + radius_px,
            center_px[1] + radius_px,
        ]
        draw.ellipse(bbox, outline="black", width=stroke)

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render MONU-CAD .mcd/.mcc geometry to PNG.")
    parser.add_argument("input", type=Path, help="Source .mcd/.mcc/.decompressed payload")
    parser.add_argument("--preview", type=Path, help="Path for the low-res preview PNG")
    parser.add_argument("--preview-size", type=int, default=256, help="Preview size in pixels (square)")
    parser.add_argument("--hires", type=Path, help="Path for the high-res PNG")
    parser.add_argument("--hires-size", type=int, default=2048, help="High-res size in pixels (square)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.preview and not args.hires:
        raise SystemExit("Specify --preview and/or --hires to render a PNG.")

    lines, arcs, circles = load_geometry(args.input)
    if args.preview:
        render_png(lines, arcs, circles, args.preview, args.preview_size)
        print(f"[+] Preview PNG written to {args.preview}")
    if args.hires:
        render_png(lines, arcs, circles, args.hires, args.hires_size)
        print(f"[+] High-res PNG written to {args.hires}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
