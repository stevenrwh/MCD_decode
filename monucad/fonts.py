from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from .entities import LineEntity

PRINTABLE_ASCII = tuple(chr(i) for i in range(32, 127))
GLYPH_COORD_SCALE = 64.0


@dataclass(frozen=True)
class TextEntity:
    text: str
    font: str
    metrics: Tuple[float, ...]


@dataclass
class Glyph:
    label: str
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    bounds: Tuple[float, float, float, float]
    advance: float
    baseline: float


@dataclass
class FontDefinition:
    name: str
    glyphs: dict[str, Glyph]
    space_advance: float
    width_scale: float = 1.0
    kerning: dict[Tuple[str, str], float] = field(default_factory=dict)

    def render(
        self,
        text: str,
        metrics: Tuple[float, ...],
        *,
        layer: int = 0,
    ) -> List[LineEntity]:
        if not metrics:
            return []
        height = max(metrics[0], 1e-6) if len(metrics) > 0 else 1.0
        width_scale = metrics[4] if len(metrics) > 4 and metrics[4] > 0 else self.width_scale
        tracking = metrics[5] if len(metrics) > 5 else 0.0
        origin_x = metrics[6] if len(metrics) > 6 else 0.0
        origin_y = metrics[7] if len(metrics) > 7 else 0.0
        entities: List[LineEntity] = []
        cursor_x = origin_x
        prev_char: str | None = None
        for ch in text:
            kerning_shift = self._kerning_adjust(prev_char, ch, height, width_scale)
            cursor_x += kerning_shift
            glyph = self.glyphs.get(ch)
            if not glyph:
                cursor_x += self.space_advance * width_scale + tracking
                prev_char = ch
                continue
            min_x, min_y, max_x, max_y = glyph.bounds
            glyph_width = max(glyph.advance, max_x - min_x, 1e-6)
            glyph_height = max(max_y - min_y, 1e-6)
            scale = height / glyph_height
            scale_x = scale * width_scale
            scale_y = scale
            baseline = glyph.baseline
            base_y = origin_y - baseline * scale_y
            base_x = cursor_x - min_x * scale_x
            for p1, p2 in glyph.segments:
                start = (base_x + p1[0] * scale_x, base_y + p1[1] * scale_y)
                end = (base_x + p2[0] * scale_x, base_y + p2[1] * scale_y)
                if _points_match(start, end):
                    continue
                entities.append(LineEntity(layer=layer, start=start, end=end))
            cursor_x += glyph_width * scale_x + tracking
            prev_char = ch
        return entities

    def _kerning_adjust(
        self,
        left: str | None,
        right: str,
        height: float,
        width_scale: float,
    ) -> float:
        if not left or not self.kerning:
            return 0.0
        delta_em = self.kerning.get((left, right))
        if not delta_em:
            return 0.0
        return delta_em * height * width_scale


class FontManager:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._fonts: dict[str, FontDefinition] = {}
        self._config = self._load_config()
        self._kerning_cache: dict[str, dict[Tuple[str, str], float]] = {}

    def _load_config(self) -> dict[str, dict[str, str]]:
        config_path = self.root / "mcfonts.lst"
        if not config_path.exists():
            defaults: dict[str, dict[str, str]] = {}
            def _maybe_add(name: str, fontfile: str, dtafile: str | None = None) -> None:
                defaults[name] = {"fontfile": fontfile}
                if dtafile:
                    defaults[name]["dtafile"] = dtafile
            _maybe_add("MAIN", "Mcalf092_exported_from_Monucad_unexploded.dxf")
            _maybe_add("VERMARCO", "Mcalf020_exported_from_monucad.dxf", "M92_kerning.json")
            _maybe_add("VM", "VM_exported_from_monucad.dxf")
            _maybe_add("SANS", "PressModifiedRoman_exported_from_monucad.dxf")
            return defaults
        config: dict[str, dict[str, str]] = {}
        for line in config_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            name, fontfile = parts[0], parts[1]
            entry: dict[str, str] = {"fontfile": fontfile}
            if len(parts) >= 3:
                entry["dtafile"] = parts[2]
            config[name] = entry
        return config

    def get(self, name: str) -> FontDefinition | None:
        key = name.upper()
        if key in self._fonts:
            return self._fonts[key]
        entry = self._config.get(key)
        if not entry:
            return None
        definition = self._load_font(entry)
        if definition:
            self._fonts[key] = definition
        return definition

    def render_text(self, text: str, metrics: Tuple[float, ...], *, layer: int = 0) -> List[LineEntity]:
        if not text:
            return []
        font_key = None
        if metrics and len(metrics) >= 8:
            font_key = metrics[8] if len(metrics) > 8 else None
        font = None
        if isinstance(font_key, str):
            font = self.get(font_key)
        if not font:
            font = self.get("MAIN")
        if not font:
            return []
        return font.render(text, metrics, layer=layer)

    def _load_font(self, entry: dict[str, str]) -> FontDefinition | None:
        fontfile = entry.get("fontfile")
        if not fontfile:
            return None
        dtafile = entry.get("dtafile")
        # JSON glyph map path (preferred when present)
        json_candidate = (self.root / Path(fontfile).stem).with_suffix(".json")
        if json_candidate.exists():
            glyphs = self._load_glyphs_from_json(json_candidate)
            kerning = self._load_kerning(dtafile) if dtafile else {}
            if glyphs:
                return FontDefinition(
                    name=Path(fontfile).stem,
                    glyphs=glyphs,
                    space_advance=1.0,
                    kerning=kerning,
                )
        dxf_path = self.root / fontfile
        glyphs = self._load_glyphs_from_dxf(dxf_path)
        if not glyphs:
            return None
        kerning = self._load_kerning(dtafile) if dtafile else {}
        space_advance = 1.0
        return FontDefinition(
            name=Path(fontfile).stem,
            glyphs=glyphs,
            space_advance=space_advance,
            kerning=kerning,
        )

    def _load_kerning(self, dtafile: str) -> dict[Tuple[str, str], float]:
        path = self.root / dtafile
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        entries: dict[Tuple[str, str], float] = {}
        for key, value in data.items():
            if not isinstance(key, str) or len(key) != 2:
                continue
            left, right = key[0], key[1]
            try:
                delta = float(value)
            except Exception:
                continue
            entries[(left, right)] = delta
        return entries

    def _load_glyphs_from_json(self, path: Path) -> dict[str, Glyph]:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        glyphs: dict[str, Glyph] = {}
        for label, entry in raw.items():
            segments = []
            for line in entry.get("lines", []):
                start = tuple(float(v) for v in line.get("start", (0.0, 0.0)))
                end = tuple(float(v) for v in line.get("end", (0.0, 0.0)))
                segments.append((start, end))
            bbox = entry.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            if len(bbox) < 4:
                continue
            min_x, min_y, max_x, max_y = (float(b) for b in bbox[:4])
            advance = max(max_x - min_x, 0.0)
            glyphs[label] = Glyph(
                label=label,
                segments=segments,
                bounds=(min_x, min_y, max_x, max_y),
                advance=advance,
                baseline=min_y,
            )
        total_segments = sum(len(glyph.segments) for glyph in glyphs.values())
        if len(glyphs) < 40 or total_segments < 200:
            return {}
        return glyphs

    def _load_glyphs_from_dxf(self, path: Path) -> dict[str, Glyph]:
        if not path.exists():
            return {}
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception:
            return {}
        glyphs: dict[str, Glyph] = {}
        idx = 0
        in_blocks = False
        while idx + 1 < len(lines):
            code = lines[idx].strip()
            value = lines[idx + 1].strip()
            idx += 2
            if not in_blocks:
                if code == "0" and value == "SECTION":
                    if idx + 1 < len(lines) and lines[idx].strip() == "2" and lines[idx + 1].strip() == "BLOCKS":
                        in_blocks = True
                        idx += 2
                continue
            if code == "0" and value == "ENDSEC":
                break
            if code == "0" and value == "BLOCK":
                block_name: str | None = None
                segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
                while idx + 1 < len(lines):
                    code = lines[idx].strip()
                    value = lines[idx + 1].strip()
                    idx += 2
                    if code == "2" and block_name is None:
                        block_name = value
                        continue
                    if code == "0" and value == "ENDBLK":
                        if block_name and segments:
                            bounds = _compute_bounds(segments)
                            min_x, min_y, max_x, max_y = bounds
                            advance = max(max_x - min_x, 0.0)
                            glyphs[block_name] = Glyph(
                                label=block_name,
                                segments=segments,
                                bounds=bounds,
                                advance=advance,
                                baseline=min_y,
                            )
                        break
                    if code == "0":
                        entity_type = value
                        new_segments, idx = _parse_block_entity(entity_type, lines, idx)
                        if new_segments:
                            segments.extend(new_segments)
        return glyphs


def _points_match(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    tol: float = 1e-6,
) -> bool:
    return abs(p1[0] - p2[0]) <= tol and abs(p1[1] - p2[1]) <= tol


def _compute_bounds(segments: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]]) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for p1, p2 in segments:
        xs.extend((p1[0], p2[0]))
        ys.extend((p1[1], p2[1]))
    return min(xs), min(ys), max(xs), max(ys)


def _parse_block_entity(
    entity_type: str,
    data: List[str],
    idx: int,
) -> Tuple[List[Tuple[Tuple[float, float], Tuple[float, float]]], int]:
    attrs: dict[str, float] = {}
    while idx + 1 < len(data):
        code = data[idx].strip()
        value = data[idx + 1].strip()
        idx += 2
        if code == "0":
            idx -= 2
            break
        try:
            attrs[code] = float(value)
        except ValueError:
            attrs[code] = 0.0
    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    if entity_type == "LINE":
        segments.append(
            (
                (attrs.get("10", 0.0), attrs.get("20", 0.0)),
                (attrs.get("11", 0.0), attrs.get("21", 0.0)),
            )
        )
    return segments, idx
