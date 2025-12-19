#!/usr/bin/env python3
"""
Diagnostic helper that aligns raw type-3 arc records from a MONU-CAD .mcd payload
with the corresponding ARC/CIRCLE entities from a trusted DXF export.  The goal
is to highlight which helper records reproduce the DXF start/center/end points so
we can reverse-engineer the newer Monu-CAD arc/circle encoding.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import mcd_to_dxf

TOLERANCE = 1e-3


@dataclass
class DxfArc:
    kind: str  # "ARC" or "CIRCLE"
    center: Tuple[float, float]
    radius: float
    start_angle: float | None = None
    end_angle: float | None = None

    def start_point(self) -> Tuple[float, float] | None:
        if self.kind != "ARC" or self.start_angle is None:
            return None
        return _point_at_angle(self.center, self.radius, self.start_angle)

    def end_point(self) -> Tuple[float, float] | None:
        if self.kind != "ARC" or self.end_angle is None:
            return None
        return _point_at_angle(self.center, self.radius, self.end_angle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare raw .mcd arc helper records with a reference DXF export."
    )
    parser.add_argument("mcd", type=Path, help="Path to the source .mcd file")
    parser.add_argument("reference_dxf", type=Path, help="DXF exported from Monu-CAD for the same drawing")
    parser.add_argument(
        "--window",
        type=int,
        default=16,
        help="Number of records after each type-3 arc to treat as helper candidates",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional destination for the textual report (defaults to <mcd>.arc_helper_report.txt)",
    )
    return parser.parse_args()


def _point_at_angle(center: Tuple[float, float], radius: float, angle_deg: float) -> Tuple[float, float]:
    angle_rad = math.radians(angle_deg)
    return (
        center[0] + radius * math.cos(angle_rad),
        center[1] + radius * math.sin(angle_rad),
    )


def _close(p1: Tuple[float, float], p2: Tuple[float, float], tol: float = TOLERANCE) -> bool:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1]) <= tol


def parse_dxf_arcs(path: Path) -> List[DxfArc]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    entities: List[DxfArc] = []
    i = 0
    n = len(lines)
    while i < n - 1:
        code = lines[i].strip()
        value = lines[i + 1].strip()
        if code == "0" and value in {"ARC", "CIRCLE"}:
            entity = value
            data = {}
            i += 2
            while i < n - 1 and lines[i].strip() != "0":
                data_code = lines[i].strip()
                data_val = lines[i + 1].strip()
                data[data_code] = data_val
                i += 2
            center = (float(data.get("10", "0")), float(data.get("20", "0")))
            radius = float(data.get("40", "0"))
            if entity == "ARC":
                start_angle = float(data.get("50", "0"))
                end_angle = float(data.get("51", "0"))
                entities.append(DxfArc("ARC", center, radius, start_angle, end_angle))
            else:
                entities.append(DxfArc("CIRCLE", center, radius))
            continue
        i += 2
    return entities


def classify_point(
    point: Tuple[float, float],
    labels: Sequence[Tuple[str, Tuple[float, float]]],
    *,
    tol: float = TOLERANCE,
) -> str:
    hits = [name for name, ref in labels if _close(point, ref, tol)]
    return "|".join(hits) if hits else "-"


def gather_arc_context(
    records: Sequence[Tuple[int, int, int, float, float, float, float]],
    window: int,
) -> List[Tuple[int, int, float, float, float, float, List[Tuple[int, int, int, float, float, float, float]]]]:
    by_offset = {offset: (layer, etype, x1, y1, x2, y2) for offset, layer, etype, x1, y1, x2, y2 in records}
    record_offsets = [offset for offset, *_ in records]
    result: List[
        Tuple[int, int, float, float, float, float, List[Tuple[int, int, int, float, float, float, float]]]
    ] = []
    for idx, offset in enumerate(record_offsets):
        layer, etype, x1, y1, x2, y2 = by_offset[offset]
        if etype != 3:
            continue
        neighbor_offsets = record_offsets[idx + 1 : idx + 1 + window]
        neighbors = [
            (n_offset, *by_offset[n_offset])
            for n_offset in neighbor_offsets
            if n_offset in by_offset
        ]
        result.append((offset, layer, x1, y1, x2, y2, neighbors))
    return result


def match_arc_records(
    arc_records: Sequence[Tuple[int, int, float, float, float, float, List[Tuple]]],
    dxf_entities: Sequence[DxfArc],
) -> List[Tuple[int | None, str | None]]:
    used: set[int] = set()
    matches: List[Tuple[int | None, str | None]] = []
    for _, _, sx, sy, *_ in arc_records:
        start_point = (sx, sy)
        match_idx: int | None = None
        match_role: str | None = None
        for idx, entity in enumerate(dxf_entities):
            if entity.kind != "ARC" or idx in used:
                continue
            dxf_start = entity.start_point()
            dxf_end = entity.end_point()
            if dxf_start and _close(start_point, dxf_start):
                match_idx = idx
                match_role = "matches DXF start"
                break
            if dxf_end and _close(start_point, dxf_end):
                match_idx = idx
                match_role = "matches DXF end"
                break
        if match_idx is not None:
            used.add(match_idx)
        matches.append((match_idx, match_role))
    return matches


def main() -> int:
    args = parse_args()
    blob = args.mcd.read_bytes()
    offset, payload = mcd_to_dxf.brute_force_deflate(blob)
    records = mcd_to_dxf.collect_candidate_records(payload)
    arc_context = gather_arc_context(records, window=args.window)
    dxf_entities = parse_dxf_arcs(args.reference_dxf)

    matches = match_arc_records(arc_context, dxf_entities)
    output_path = args.output or args.mcd.with_suffix(".arc_helper_report.txt")

    circle_entity = next((entity for entity in dxf_entities if entity.kind == "CIRCLE"), None)

    report_lines: List[str] = []
    report_lines.append(f"[+] Decompressed payload offset {offset} ({len(payload)} bytes)")
    report_lines.append(
        f"[+] Found {len(arc_context)} type-3 records; matched "
        f"{sum(1 for m, _ in matches if m is not None)} DXF arcs"
    )
    report_lines.append("")

    for idx, ((arc_offset, layer, sx, sy, cx, cy, neighbors), (match_idx, match_role)) in enumerate(
        zip(arc_context, matches), start=1
    ):
        radius = math.hypot(sx - cx, sy - cy)
        report_lines.append(
            f"ARC #{idx}: offset=0x{arc_offset:04X} layer={layer} "
            f"start=({sx:.6f},{sy:.6f}) center=({cx:.6f},{cy:.6f}) radius={radius:.6f}"
        )
        labels: List[Tuple[str, Tuple[float, float]]] = [
            ("payload_start", (sx, sy)),
            ("payload_center", (cx, cy)),
        ]
        if match_idx is not None:
            entity = dxf_entities[match_idx]
            role_suffix = f" ({match_role})" if match_role else ""
            report_lines.append(
                f"  DXF match: center=({entity.center[0]:.6f},{entity.center[1]:.6f}) "
                f"radius={entity.radius:.6f} angles=({entity.start_angle:.6f},{entity.end_angle:.6f}){role_suffix}"
            )
            dxf_start = entity.start_point()
            dxf_end = entity.end_point()
            if dxf_start and dxf_end:
                report_lines.append(
                    f"  DXF start=({dxf_start[0]:.6f},{dxf_start[1]:.6f}) "
                    f"end=({dxf_end[0]:.6f},{dxf_end[1]:.6f})"
                )
            elif dxf_start:
                report_lines.append(f"  DXF start=({dxf_start[0]:.6f},{dxf_start[1]:.6f}) end=(unknown)")
            if dxf_start:
                labels.append(("dxf_start", dxf_start))
            if dxf_end:
                labels.append(("dxf_end", dxf_end))
        else:
            report_lines.append("  DXF match: none")

        if not neighbors:
            report_lines.append("  (no helper records captured)")
            report_lines.append("")
            continue

        report_lines.append("  helper records:")
        for helper_offset, helper_layer, etype, x1, y1, x2, y2 in neighbors:
            label_a = classify_point((x1, y1), labels)
            label_b = classify_point((x2, y2), labels)
            report_lines.append(
                f"    off=0x{helper_offset:04X} layer={helper_layer:<10} etype={etype:<10} "
                f"p1=({x1:.6f},{y1:.6f}) [{label_a}] "
                f"p2=({x2:.6f},{y2:.6f}) [{label_b}]"
            )
        report_lines.append("")

    if circle_entity:
        report_lines.append("CIRCLE diagnostics:")
        report_lines.append(
            f"  DXF circle center=({circle_entity.center[0]:.6f},{circle_entity.center[1]:.6f}) "
            f"radius={circle_entity.radius:.6f}"
        )
        circle_candidates = []
        for offset, layer, etype, x1, y1, x2, y2 in records:
            center_hit = _close((x1, y1), circle_entity.center) or _close((x2, y2), circle_entity.center)
            touches_radius = abs(math.hypot(x1, y1) - circle_entity.radius) <= TOLERANCE or abs(
                math.hypot(x2, y2) - circle_entity.radius
            ) <= TOLERANCE
            if center_hit or touches_radius:
                circle_candidates.append((offset, layer, etype, x1, y1, x2, y2))
            if len(circle_candidates) >= 10:
                break
        if circle_candidates:
            report_lines.append("  potential helper records:")
            for offset, layer, etype, x1, y1, x2, y2 in circle_candidates:
                report_lines.append(
                    f"    off=0x{offset:04X} layer={layer:<10} etype={etype:<10} "
                    f"p1=({x1:.6f},{y1:.6f}) p2=({x2:.6f},{y2:.6f})"
                )
        else:
            report_lines.append("  (no obvious helper records referencing the circle)")
    else:
        report_lines.append("No DXF circle entity detected.")

    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[+] Report written to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
