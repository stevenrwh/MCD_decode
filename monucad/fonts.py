from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from .entities import ArcEntity, CircleEntity, LineEntity

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

    def _approximate_arc(
        center: Tuple[float, float],
        radius: float,
        start_deg: float,
        end_deg: float,
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        start_rad = math.radians(start_deg)
        end_rad = math.radians(end_deg)
        while end_rad < start_rad:
            end_rad += math.tau
        sweep = end_rad - start_rad
        steps = max(6, int(abs(sweep) / (math.pi / 18)))
        points: List[Tuple[float, float]] = []
        for step in range(steps + 1):
            t = step / steps
            angle = start_rad + sweep * t
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            points.append((x, y))
        segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for p1, p2 in zip(points, points[1:]):
            segments.append((p1, p2))
        return segments

    segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    if entity_type == "LINE":
        start = (attrs.get("10", 0.0), attrs.get("20", 0.0))
        end = (attrs.get("11", 0.0), attrs.get("21", 0.0))
        segments.append((start, end))
    elif entity_type == "ARC":
        center = (attrs.get("10", 0.0), attrs.get("20", 0.0))
        radius = attrs.get("40", 0.0)
        start_ang = attrs.get("50", 0.0)
        end_ang = attrs.get("51", 0.0)
        if radius > 0:
            segments.extend(_approximate_arc(center, radius, start_ang, end_ang))
    elif entity_type == "CIRCLE":
        center = (attrs.get("10", 0.0), attrs.get("20", 0.0))
        radius = attrs.get("40", 0.0)
        if radius > 0:
            segments.extend(_approximate_arc(center, radius, 0.0, 360.0))

    return segments, idx


# -------- Font discovery and mapping helpers (shared with converters) --------
PUNCT_SUFFIX_MAP = {
    "PERID": ".",
    "PERIOD": ".",
    "DOT": ".",
    "COMMA": ",",
    "COLN": ":",
    "COLON": ":",
    "SEMI": ";",
    "APOST": "'",
    "QUOTE": '"',
    "QUOT": '"',
    "QUOT2": '"',
    "DASH": "-",
    "HYPHEN": "-",
    "MINUS": "-",
    "PLUS": "+",
    "SPACE": " ",
    "SP": " ",
    "AMP": "&",
    "AMPERSAND": "&",
    "AND": "&",
    "AT": "@",
    "ATS": "@",
    "EXCL": "!",
    "QUES": "?",
    "PERC": "%",
    "PERCENT": "%",
    "STAR": "*",
    "ASTER": "*",
    "SLASH": "/",
    "FSLASH": "/",
    "BSLASH": "\\",
    "LBS": "#",
    "HASH": "#",
    "POUND": "#",
    "LPAREN": "(",
    "RPAREN": ")",
}


def build_font_prefix_map(manager: FontManager) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for name, config in manager.iter_font_configs():
        tokens = set()
        upper_name = name.upper()
        tokens.add(upper_name)
        tokens.add(upper_name.replace(" ", ""))
        dta = config.get("dtafile", "").upper()
        if dta:
            tokens.add(dta)
            tokens.add(f"F{dta}")
        fontfile = config.get("fontfile", "").upper()
        if fontfile:
            tokens.add(fontfile.replace("MCALF", "M"))
            tokens.add(fontfile.replace(".", ""))
        for token in tokens:
            if not token:
                continue
            mapping.setdefault(token, name)
    return mapping


def decode_glyph_label(label: str, prefix_map: dict[str, str]) -> Tuple[str | None, str | None]:
    upper = label.upper()
    for split in range(len(upper), 0, -1):
        prefix = upper[:split]
        suffix = upper[split:]
        font_name = prefix_map.get(prefix)
        if not font_name or not suffix:
            continue
        char = char_from_suffix(suffix)
        if char:
            return font_name, char
    return None, None


def match_known_font(
    payload: bytes,
    offset: int,
    known_fonts: set[str],
) -> tuple[str | None, int]:
    limit = len(payload)
    for name in known_fonts:
        encoded = name.encode("ascii", errors="ignore")
        if not encoded:
            continue
        end = offset + len(encoded)
        if end > limit:
            continue
        if payload[offset:end] == encoded:
            return name, end
    return None, offset


def looks_like_font_name(name: str, *, max_len: int = 32) -> bool:
    if not name or len(name) > max_len:
        return False
    return all(32 <= ord(ch) < 127 for ch in name)


def locate_metrics(
    payload: bytes,
    start: int,
    count: int,
) -> tuple[int | None, tuple[float, ...]]:
    limit = len(payload)
    max_offset = min(start + 8, limit)
    for candidate in range(start, max_offset):
        end = candidate + count * 4
        if end > limit:
            break
        values = struct.unpack_from(f"<{count}f", payload, candidate)
        if not all(math.isfinite(val) and abs(val) <= 1e6 for val in values):
            continue
        if max(abs(val) for val in values) < 1e-3:
            continue
        return candidate, values
    return None, ()


def char_from_suffix(suffix: str) -> str | None:
    if not suffix:
        return None
    suffix = suffix.upper()
    if len(suffix) == 1 and suffix.isalpha():
        return suffix
    if len(suffix) == 1 and suffix.isdigit():
        return suffix
    if suffix.isdigit():
        try:
            code = int(suffix)
        except ValueError:
            return None
        if 32 <= code <= 126:
            return chr(code)
        return None
    mapped = PUNCT_SUFFIX_MAP.get(suffix)
    if mapped:
        return mapped
    return None


def _longest_alpha_prefix(labels: Sequence[str]) -> str:
    if not labels:
        return ""
    prefix = labels[0].upper()
    for label in labels[1:]:
        candidate = label.upper()
        while prefix and not candidate.startswith(prefix):
            prefix = prefix[:-1]
    while prefix and not prefix[-1].isalpha():
        prefix = prefix[:-1]
    return prefix


def candidate_prefixes(config: dict[str, str] | None, labels: Sequence[str]) -> List[str]:
    prefixes: List[str] = []
    if config:
        raw_prefix = config.get("dtafile", "").strip()
        token = re.sub(r"[^0-9A-Za-z]", "", raw_prefix).upper()
        if token:
            prefixes.append(token)
            prefixes.append(f"F{token}")
        fontfile = config.get("fontfile", "")
        if fontfile:
            token_ff = re.sub(r"[^0-9A-Za-z]", "", fontfile).upper()
            if token_ff:
                prefixes.append(token_ff)
    inferred = _longest_alpha_prefix(labels)
    if inferred:
        prefixes.append(inferred)
    prefixes.append("")
    seen: List[str] = []
    for prefix in prefixes:
        if prefix not in seen:
            seen.append(prefix)
    return seen


def build_vm_mapping() -> List[Tuple[str, str]]:
    mapping: List[Tuple[str, str]] = []
    for i in range(26):
        ch = chr(ord("A") + i)
        mapping.append((ch, f"VM{ch}"))
    for digit in "0123456789":
        mapping.append((digit, f"VM{digit}"))
    mapping.extend(
        [
            ("-", "VM-"),
            (".", "VMPERID"),
            (",", "VMCOMMA"),
            (";", "VMSEMI"),
            (":", "VMCOLN"),
            ("'", "VMAPOST"),
            ('"', "VMQUOTE"),
            ("&", "VM&"),
        ]
    )
    return mapping


def derive_font_mapping(
    font_name: str,
    glyphs: dict[str, Glyph],
    config: dict[str, str] | None,
) -> List[Tuple[str, str]]:
    upper_name = font_name.upper()
    if upper_name == "MAIN":
        return list(MAIN_GLYPH_MAP)
    if upper_name == "VERMARCO":
        return build_vm_mapping()
    labels = list(glyphs.keys())
    if not labels:
        return []
    prefixes = candidate_prefixes(config, labels)
    allow_lowercase = True
    if config:
        lc_flag = config.get("lowercase")
        if lc_flag is not None:
            allow_lowercase = lc_flag.strip().lower() == "yes"
    mapping: dict[str, str] = {}
    for label in labels:
        label_upper = label.upper()
        suffix = label_upper
        for prefix in prefixes:
            if prefix and label_upper.startswith(prefix):
                suffix = label_upper[len(prefix) :]
                break
        char = char_from_suffix(suffix)
        if char and char not in mapping:
            mapping[char] = label
    ordered_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if allow_lowercase:
        ordered_chars += "abcdefghijklmnopqrstuvwxyz"
    ordered_chars += "0123456789"
    ordered_chars += " -.,'\"!@#$%^&*()/?"
    result: List[Tuple[str, str]] = []
    for ch in ordered_chars:
        if ch in mapping:
            result.append((ch, mapping[ch]))
    for ch, label in mapping.items():
        if ch not in ordered_chars:
            result.append((ch, label))
    return result


# VM/MONUCAD reference mapping used by the converters
MAIN_GLYPH_MAP: List[Tuple[str, str]] = [
    ("A", "A"),
    ("B", "B"),
    ("C", "C"),
    ("D", "D"),
    ("E", "E"),
    ("F", "F"),
    ("G", "G"),
    ("H", "H"),
    ("I", "I"),
    ("J", "J"),
    ("K", "K"),
    ("L", "L"),
    ("M", "M"),
    ("N", "N"),
    ("O", "O"),
    ("P", "P"),
    ("Q", "Q"),
    ("R", "R"),
    ("S", "S"),
    ("T", "T"),
    ("U", "U"),
    ("V", "V"),
    ("W", "W"),
    ("X", "X"),
    ("Y", "Y"),
    ("Z", "Z"),
    ("1", "1"),
    ("2", "2"),
    ("3", "3"),
    ("4", "4"),
    ("5", "5"),
    ("6", "6"),
    ("7", "7"),
    ("8", "8"),
    ("9", "9"),
    ("0", "0"),
    ("-", "DASH"),
    (".", "PERIOD"),
    (",", "COMMA"),
    (";", "SEMI"),
    (":", "COLN"),
    ("'", "APOST"),
    ('"', "QUOT"),
    ("&", "AMP"),
    ("/", "SLASH"),
]


# -------- Reference DXF helpers (diagnostic) --------
REFERENCE_DXF_MAP = {
    "MAIN": Path("FONTS/Mcalf092_exported_from_Monucad_unexploded.dxf"),
    "VERMARCO": Path("FONTS/Mcalf020_exported_from_monucad.dxf"),
}


def load_glyphs_from_reference(font_name: str) -> dict[str, Glyph]:
    path = REFERENCE_DXF_MAP.get(font_name.upper())
    if not path or not path.exists():
        return {}
    return _parse_unexploded_blocks(path)


def _parse_unexploded_blocks(path: Path) -> dict[str, Glyph]:
    data = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    glyphs: dict[str, Glyph] = {}
    idx = 0
    in_blocks = False
    while idx + 1 < len(data):
        code = data[idx].strip()
        value = data[idx + 1].strip()
        idx += 2
        if not in_blocks:
            if code == "0" and value == "SECTION":
                if idx + 1 < len(data) and data[idx].strip() == "2" and data[idx + 1].strip() == "BLOCKS":
                    in_blocks = True
                    idx += 2
            continue
        if code == "0" and value == "ENDSEC":
            break
        if code == "0" and value == "BLOCK":
            block_name: str | None = None
            segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
            while idx + 1 < len(data):
                code = data[idx].strip()
                value = data[idx + 1].strip()
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
                    new_segments, idx = _parse_block_entity(entity_type, data, idx)
                    if new_segments:
                        segments.extend(new_segments)
                # other codes are metadata we can skip
    return glyphs


def parse_kerning_file(path: Path) -> dict[Tuple[str, str], float]:
    entries: dict[Tuple[str, str], float] = {}
    try:
        data = path.read_bytes()
    except OSError:
        return entries
    expected = 95 * 95 * 2
    if len(data) < expected:
        return entries
    offset = 0
    for row in range(95):
        for col in range(95):
            raw = int.from_bytes(data[offset : offset + 2], "little", signed=False)
            offset += 2
            if row == 0 and col == 0:
                continue
            if raw in (0, 1, 1000):
                continue
            signed = raw if raw < 0x8000 else raw - 0x10000
            if signed == 0:
                continue
            first = PRINTABLE_ASCII[row] if row < len(PRINTABLE_ASCII) else chr(32 + row)
            second = PRINTABLE_ASCII[col] if col < len(PRINTABLE_ASCII) else chr(32 + col)
            entries[(first, second)] = signed / 1000.0
    return entries


def load_reference_entities(path: Path) -> Tuple[List[LineEntity], List[ArcEntity], List[CircleEntity]]:
    data = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    idx = 0
    lines: List[LineEntity] = []
    arcs: List[ArcEntity] = []
    circles: List[CircleEntity] = []

    while idx + 1 < len(data):
        code = data[idx].strip()
        value = data[idx + 1].strip()
        idx += 2
        if code != "0":
            continue
        if value == "LINE":
            attrs, idx = _collect_entity_attrs(data, idx)
            lines.append(
                LineEntity(
                    layer=0,
                    start=(attrs.get("10", 0.0), attrs.get("20", 0.0)),
                    end=(attrs.get("11", 0.0), attrs.get("21", 0.0)),
                )
            )
        elif value == "ARC":
            attrs, idx = _collect_entity_attrs(data, idx)
            center = (attrs.get("10", 0.0), attrs.get("20", 0.0))
            radius = attrs.get("40", 0.0)
            start_ang = math.radians(attrs.get("50", 0.0))
            end_ang = math.radians(attrs.get("51", 0.0))
            start = (center[0] + radius * math.cos(start_ang), center[1] + radius * math.sin(start_ang))
            end = (center[0] + radius * math.cos(end_ang), center[1] + radius * math.sin(end_ang))
            arcs.append(ArcEntity(layer=0, center=center, start=start, end=end))
        elif value == "CIRCLE":
            attrs, idx = _collect_entity_attrs(data, idx)
            center = (attrs.get("10", 0.0), attrs.get("20", 0.0))
            radius = attrs.get("40", 0.0)
            circles.append(CircleEntity(layer=0, center=center, radius=radius))
    return lines, arcs, circles


def _collect_entity_attrs(lines: List[str], idx: int) -> Tuple[dict[str, float], int]:
    attrs: dict[str, float] = {}
    while idx + 1 < len(lines):
        code = lines[idx].strip()
        value = lines[idx + 1].strip()
        idx += 2
        if code == "0":
            idx -= 2
            break
        try:
            attrs[code] = float(value)
        except ValueError:
            attrs[code] = 0.0
    return attrs, idx
