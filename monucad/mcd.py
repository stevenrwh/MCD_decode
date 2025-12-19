from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence, Tuple

from .entities import ArcEntity, CircleEntity, InsertEntity, LineEntity


def write_dxf(
    lines: Sequence[LineEntity],
    arcs: Sequence[ArcEntity],
    circles: Sequence[CircleEntity],
    destination: Path,
    inserts: Sequence[InsertEntity] | None = None,
) -> None:
    """
    Emit a bare-bones DXF file that places every entity on its original layer
    (named MCD_Layer_<id>) and keeps Z at 0.
    """

    def emit(code: str, value: str) -> str:
        return f"{code}\n{value}\n"

    chunks: list[str] = []

    # Optional placeholder BLOCK definitions for any inserts we plan to emit.
    if inserts:
        block_names = {ins.name for ins in inserts}
        chunks.append(emit("0", "SECTION"))
        chunks.append(emit("2", "BLOCKS"))
        for name in sorted(block_names):
            chunks.append(emit("0", "BLOCK"))
            chunks.append(emit("8", "0"))
            chunks.append(emit("2", name))
            chunks.append(emit("70", "0"))
            chunks.append(emit("10", "0.0"))
            chunks.append(emit("20", "0.0"))
            chunks.append(emit("30", "0.0"))
            chunks.append(emit("3", name))
            chunks.append(emit("1", name))
            chunks.append(emit("0", "ENDBLK"))
        chunks.append(emit("0", "ENDSEC"))

    chunks.append(emit("0", "SECTION"))
    chunks.append(emit("2", "ENTITIES"))

    for line in lines:
        layer_name = f"MCD_Layer_{line.layer}"
        chunks.append(emit("0", "LINE"))
        chunks.append(emit("8", layer_name))
        chunks.append(emit("10", f"{line.start[0]:.6f}"))
        chunks.append(emit("20", f"{line.start[1]:.6f}"))
        chunks.append(emit("30", "0.0"))
        chunks.append(emit("11", f"{line.end[0]:.6f}"))
        chunks.append(emit("21", f"{line.end[1]:.6f}"))
        chunks.append(emit("31", "0.0"))

    for arc in arcs:
        layer_name = f"MCD_Layer_{arc.layer}"
        radius = math.hypot(arc.start[0] - arc.center[0], arc.start[1] - arc.center[1])
        if radius < 1e-9:
            continue

        def _angle(pt: Tuple[float, float]) -> float:
            return math.degrees(math.atan2(pt[1] - arc.center[1], pt[0] - arc.center[0])) % 360.0

        start_ang = _angle(arc.start)
        end_ang = _angle(arc.end)

        chunks.append(emit("0", "ARC"))
        chunks.append(emit("8", layer_name))
        chunks.append(emit("10", f"{arc.center[0]:.6f}"))
        chunks.append(emit("20", f"{arc.center[1]:.6f}"))
        chunks.append(emit("40", f"{radius:.6f}"))
        chunks.append(emit("50", f"{start_ang:.6f}"))
        chunks.append(emit("51", f"{end_ang:.6f}"))

    for circle in circles:
        layer_name = f"MCD_Layer_{circle.layer}"
        if circle.radius < 1e-9:
            continue
        chunks.append(emit("0", "CIRCLE"))
        chunks.append(emit("8", layer_name))
        chunks.append(emit("10", f"{circle.center[0]:.6f}"))
        chunks.append(emit("20", f"{circle.center[1]:.6f}"))
        chunks.append(emit("40", f"{circle.radius:.6f}"))

    if inserts:
        for ins in inserts:
            layer_name = f"MCD_Layer_{ins.layer}"
            chunks.append(emit("0", "INSERT"))
            chunks.append(emit("8", layer_name))
            chunks.append(emit("2", ins.name))
            chunks.append(emit("10", f"{ins.position[0]:.6f}"))
            chunks.append(emit("20", f"{ins.position[1]:.6f}"))
            chunks.append(emit("30", "0.0"))
            chunks.append(emit("41", f"{ins.scale[0]:.6f}"))
            chunks.append(emit("42", f"{ins.scale[1]:.6f}"))
            chunks.append(emit("43", "1.0"))
            if abs(ins.rotation) > 1e-9:
                chunks.append(emit("50", f"{ins.rotation:.6f}"))

    chunks.append(emit("0", "ENDSEC"))
    chunks.append(emit("0", "EOF"))
    destination.write_text("".join(chunks), encoding="ascii")


__all__ = ["write_dxf"]
