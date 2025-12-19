from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

from .entities import ArcEntity, DuplicateRecord, LineEntity


def log_duplicate_records(records: Sequence[DuplicateRecord], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for idx, entry in enumerate(records, start=1):
        lines.append(
            f"#{idx:04d} offset=0x{entry.offset:04X} original=0x{entry.original_offset:04X} "
            f"layer={entry.layer} etype={entry.etype}"
        )
        lines.append(
            f"       start=({entry.start[0]:.6f},{entry.start[1]:.6f}) "
            f"end=({entry.end[0]:.6f},{entry.end[1]:.6f})"
        )
    destination.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


@dataclass
class ArcHelperLogger:
    destination: Path
    window: int = 16

    def __post_init__(self) -> None:
        self._lines: List[str] = []

    def record(
        self,
        *,
        seq: int,
        arc_offset: int,
        layer: int,
        start: Tuple[float, float],
        center: Tuple[float, float],
        neighbors: Sequence[Tuple[int, int, int, float, float, float, float]],
        note: str | None = None,
    ) -> None:
        header = (
            f"Arc #{seq} offset=0x{arc_offset:04X} layer={layer} "
            f"start=({start[0]:.6f},{start[1]:.6f}) center=({center[0]:.6f},{center[1]:.6f})"
        )
        if note:
            header += f" | {note}"
        self._lines.append(header)
        if not neighbors:
            self._lines.append("  (no trailing helper records captured)")
            return
        for rel_idx, (offset, n_layer, etype, x1, y1, x2, y2) in enumerate(neighbors, start=1):
            self._lines.append(
                f"  helper[{rel_idx:02}] off=0x{offset:04X} layer={n_layer:<10} "
                f"etype={etype:<10} "
                f"p1=({x1:.6f},{y1:.6f}) p2=({x2:.6f},{y2:.6f})"
            )

    def flush(self) -> None:
        if not self._lines:
            return
        text = "\n".join(self._lines) + "\n"
        self.destination.write_text(text, encoding="utf-8")
