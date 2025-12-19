#!/usr/bin/env python3
"""
Convert a Monu-CAD *.fnt (or *.fnt.decompressed) font archive into a DXF that
lays out every glyph as straight line segments.  This is primarily a debugging
aid so we can inspect the glyph strokes outside of Monu-CAD.
"""

from __future__ import annotations

import argparse
import sys
import re
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from glyph_tlv_parser import parse_glyph_components_from_blob

REPO_ROOT = Path(__file__).resolve().parents[0]
FONT_ROOT = (REPO_ROOT / "FONTS").resolve()
VM_TAIL_RECORDS_PATH = FONT_ROOT / "VM_tail_records.json"
VM_COMPONENTS_DIR = FONT_ROOT / "components_Mcalf020"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mcd_to_dxf import (
    FontDefinition,
    Glyph,
    LineEntity,
    MAIN_GLYPH_MAP,
    _build_vm_mapping,
    _glyph_from_component,
    _derive_font_mapping,
    brute_force_deflate,
    write_dxf,
    FontManager,
)
from font_components import iter_component_files

DEFAULT_TEXT = "ABCDEFGHIJKLMNOPQRSTUVWXYZ 0123456789-.,;:'\"&"
_VM_OUTLINE_CACHE: Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]] | None = None
_VM_TAIL_MAP: Dict[str, Tuple[int, int, int, int]] | None = None


def _read_font_payload(path: Path) -> bytes:
    data = path.read_bytes()
    if path.name.endswith(".decompressed"):
        return data
    try:
        _, payload = brute_force_deflate(data)
        return payload
    except Exception:
        return data


def _is_vm_font_name(font_name: str | None) -> bool:
    if not font_name:
        return False
    upper = font_name.upper()
    return "VERMARCO" in upper or "MCALF020" in upper or upper.startswith("VM")


def _segment_bounds(
    segments: Iterable[Tuple[Tuple[float, float], Tuple[float, float]]]
) -> Tuple[float, float, float, float]:
    xs: List[float] = []
    ys: List[float] = []
    for start, end in segments:
        xs.extend((start[0], end[0]))
        ys.extend((start[1], end[1]))
    if not xs or not ys:
        return (0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), max(xs), max(ys))


def _normalized_segment_key(
    start: Tuple[float, float], end: Tuple[float, float], *, precision: int = 6
) -> Tuple[float, float, float, float]:
    ordered = (start, end)
    if ordered[0] > ordered[1]:
        ordered = (ordered[1], ordered[0])
    return (
        round(ordered[0][0], precision),
        round(ordered[0][1], precision),
        round(ordered[1][0], precision),
        round(ordered[1][1], precision),
    )


def _load_vm_outline_segments() -> Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]]:
    global _VM_OUTLINE_CACHE
    if _VM_OUTLINE_CACHE is not None:
        return _VM_OUTLINE_CACHE

    tail_map = _load_vm_tail_map()
    if not tail_map:
        _VM_OUTLINE_CACHE = {}
        return _VM_OUTLINE_CACHE

    tail_path = VM_TAIL_RECORDS_PATH
    if not tail_path.exists():
        _VM_OUTLINE_CACHE = {}
        return _VM_OUTLINE_CACHE
    try:
        tail_data = json.loads(tail_path.read_text(encoding="utf-8"))
    except Exception:
        _VM_OUTLINE_CACHE = {}
        return _VM_OUTLINE_CACHE

    tail_lookup: Dict[Tuple[int, int, int, int], List[dict]] = {}
    for entry in tail_data.get("tails", []):
        tail = entry.get("tail")
        if not tail:
            continue
        offsets = []
        for rec in entry.get("records", []):
            if "x1" in rec and "y1" in rec and "x2" in rec and "y2" in rec:
                offsets.append(rec)
        tail_lookup[tuple(int(v) for v in tail)] = offsets

    outlines: Dict[str, List[Tuple[Tuple[float, float], Tuple[float, float]]]] = {}
    for label, tail in tail_map.items():
        records = tail_lookup.get(tail)
        if not records:
            continue
        seen: set[Tuple[float, float, float, float]] = set()
        label_segments: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
        for rec in records:
            start_pt = (float(rec.get("x1", 0.0)), float(rec.get("y1", 0.0)))
            end_pt = (float(rec.get("x2", 0.0)), float(rec.get("y2", 0.0)))
            key = _normalized_segment_key(start_pt, end_pt)
            if key in seen:
                continue
            seen.add(key)
            label_segments.append((start_pt, end_pt))
        if label_segments:
            outlines[label] = label_segments
    _VM_OUTLINE_CACHE = outlines
    return outlines


def _load_vm_tail_map() -> Dict[str, Tuple[int, int, int, int]]:
    global _VM_TAIL_MAP
    if _VM_TAIL_MAP is not None:
        return _VM_TAIL_MAP
    if not VM_COMPONENTS_DIR.exists():
        _VM_TAIL_MAP = {}
        return _VM_TAIL_MAP
    mapping: Dict[str, Tuple[int, int, int, int]] = {}
    try:
        for definition in iter_component_files(VM_COMPONENTS_DIR):
            if not definition.records:
                continue
            tail = definition.records[0].values[-4:]
            if len(tail) < 4:
                continue
            mapping[definition.label] = tuple(int(val) for val in tail[-4:])
    except Exception:
        mapping = {}
    _VM_TAIL_MAP = mapping
    return mapping


def _glyph_from_outline(component, segments: List[Tuple[Tuple[float, float], Tuple[float, float]]]) -> Glyph:
    bounds = _segment_bounds(segments)
    advance = float(component.advance) if component.advance else max(bounds[2] - bounds[0], 1e-3)
    baseline = component.baseline if component.baseline is not None else bounds[1]
    usable_segments = [
        ((float(start[0]), float(start[1])), (float(end[0]), float(end[1])))
        for start, end in segments
    ]
    return Glyph(
        label=component.label,
        segments=usable_segments,
        bounds=bounds,
        advance=advance,
        baseline=baseline,
    )


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
    "HASH": "#",
    "POUND": "#",
    "LPAREN": "(",
    "RPAREN": ")",
}


def _char_from_suffix(suffix: str) -> str | None:
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
    if suffix and suffix[-1].isalpha():
        return suffix[-1]
    return None


def _guess_char(label: str) -> str | None:
    upper = label.upper()
    match = re.search(r"([A-Z0-9]+)$", upper)
    suffix = match.group(1) if match else upper
    ch = _char_from_suffix(suffix)
    if ch:
        return ch
    for character in reversed(upper):
        if character.isalpha():
            return character
    return None


def _known_mapping(font_name: str) -> List[tuple[str, str]] | None:
    upper = font_name.upper()
    if "VERMARCO" in upper or "MCALF020" in upper or upper.startswith("VM"):
        return _build_vm_mapping()
    if "MAIN" in upper or "MCALF092" in upper or upper.startswith("M92"):
        return MAIN_GLYPH_MAP
    return None


def _auto_order_pairs(glyphs, *, mode: str) -> List[tuple[str, Glyph]]:
    glyph_map_case = {glyph.label: glyph for glyph in glyphs}
    if mode == "label":
        return [(glyph.label, glyph) for glyph in sorted(glyphs, key=lambda g: g.label)]

    desired = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
        " -.,'\"!@#$%^&*()/?"
    )
    pairs: List[tuple[str, Glyph]] = []
    assigned_labels: set[str] = set()
    sorted_labels = sorted(glyph_map_case.keys())
    for ch in desired:
        for label in sorted_labels:
            if label in assigned_labels:
                continue
            actual_label = glyph_map_case[label].label
            if _guess_char(actual_label) == ch:
                pairs.append((ch, glyph_map_case[label]))
                assigned_labels.add(label)
                break
    leftovers = [label for label in sorted_labels if label not in assigned_labels]
    for idx, label in enumerate(leftovers):
        actual_label = glyph_map_case[label].label
        fallback_char = _guess_char(actual_label) or chr(ord("⍰") + idx)
        pairs.append((fallback_char, glyph_map_case[label]))
    return pairs


def _map_glyphs(font_name: str, glyphs, *, mode: str) -> List[tuple[str, Glyph]]:
    glyph_dict = {glyph.label.upper(): glyph for glyph in glyphs}
    mapping = _derive_font_mapping(font_name.upper(), glyph_dict, {})
    pairs: List[tuple[str, Glyph]] = []
    if mapping:
        for char, label in mapping:
            glyph = glyph_dict.get(label.upper())
            if glyph:
                pairs.append((char, glyph))
    if not pairs:
        known = _known_mapping(font_name)
        if known:
            for char, label in known:
                glyph = glyph_dict.get(label.upper())
                if glyph:
                    pairs.append((char, glyph))
    if not pairs:
        pairs = _auto_order_pairs(glyphs, mode=mode)
    return pairs


def _infer_font_name(manager: FontManager | None, font_path: Path) -> str | None:
    stem = font_path.stem
    candidates = {
        stem,
        stem.upper(),
        stem.lower(),
    }
    if stem.lower().endswith(".fnt"):
        stripped = stem[:-4]
        candidates.update({stripped, stripped.upper(), stripped.lower()})
    for name, config in getattr(manager, "_config", {}).items():
        fontfile = (config.get("fontfile") or "").strip()
        if not fontfile:
            continue
        if fontfile.upper() in candidates or (fontfile + ".fnt").upper() in candidates:
            return name
    return None


def _glyph_width(component) -> float:
    min_x, _, max_x, _ = component.bbox
    width = max(max_x - min_x, component.advance if component.advance else 0.0, 1e-6)
    return float(width)


def _component_to_glyph(component) -> Glyph:
    return _glyph_from_component(component)


def _build_font_definition(font_name: str, pairs: List[tuple[str, Glyph]]) -> FontDefinition:
    glyphs: dict[str, Glyph] = {}
    advances: List[float] = []
    for char, glyph in pairs:
        glyphs[char] = glyph
        advances.append(glyph.advance)
    space_advance = sum(advances) / len(advances) if advances else 1.0
    return FontDefinition(name=font_name, glyphs=glyphs, space_advance=space_advance)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a Monu-CAD .fnt font archive into DXF lines.")
    parser.add_argument("input", type=Path, help="Path to the .fnt or .fnt.decompressed file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Destination DXF path")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of glyphs to include (post-filter)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated list of glyph labels to include (e.g., A,B,C)",
    )
    parser.add_argument(
        "--order",
        choices=["auto", "label"],
        default="auto",
        help="How to order glyphs (auto tries to map suffixes to ASCII characters)",
    )
    parser.add_argument(
        "--height",
        type=float,
        default=6.0,
        help="Nominal text height for rendering (default: 6.0 units)",
    )
    parser.add_argument(
        "--width-scale",
        type=float,
        default=1.0,
        help="Horizontal stretch factor applied to each glyph (default: 1.0)",
    )
    parser.add_argument(
        "--tracking",
        type=float,
        default=0.0,
        help="Additional spacing (in same units as height) inserted between glyphs",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Optional custom text string to render instead of the inferred alphabet/number sequence",
    )
    parser.add_argument(
        "--font-name",
        type=str,
        default=None,
        help="Explicit font name to load via mcfonts.lst (e.g., VERMARCO, MAIN).",
    )
    parser.add_argument(
        "--disable-font-manager",
        action="store_true",
        help="Skip mcfonts.lst lookups and force raw .fnt decoding (useful for outline experiments).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    font_root = FONT_ROOT if FONT_ROOT.exists() else REPO_ROOT
    manager = None if args.disable_font_manager else FontManager(font_root)
    requested_font_name = args.font_name.upper() if args.font_name else None
    inferred_name = requested_font_name or _infer_font_name(manager, args.input)
    font_def: FontDefinition | None = None
    if manager and inferred_name:
        font_def = manager.get_font(inferred_name.upper())

    if font_def:
        text = args.text if args.text is not None else DEFAULT_TEXT
        if args.limit is not None:
            text = text[: args.limit]
        supported_chars = "".join(
            ch for ch in text if ch == " " or ch in font_def.glyphs
        )
        if not supported_chars:
            raise SystemExit(
                "None of the requested characters exist in the font definition. "
                "Try passing --text or --limit with supported characters."
            )
        metrics = (
            max(args.height, 1e-3),
            0.0,
            0.0,
            0.0,
            max(args.width_scale, 1e-3),
            args.tracking,
            0.0,
            0.0,
        )
        lines = font_def.render(supported_chars, metrics, layer=0)
        if not lines:
            raise SystemExit("Font rendering produced no geometry (text string empty?)")
        write_dxf(lines, [], [], args.output)
        print(
            f"[+] Wrote {len(lines)} line entities using FontManager font '{font_def.name}' "
            f"to {args.output} (text='{supported_chars}')"
        )
        return 0

    # Fallback: parse the .fnt directly and build a temporary FontDefinition.
    blob = _read_font_payload(args.input)
    only = {label.strip().upper() for label in args.only.split(",")} if args.only else None
    glyph_components = parse_glyph_components_from_blob(blob)
    filtered_components = [
        component for component in glyph_components if not only or component.label.upper() in only
    ]
    if not filtered_components:
        raise SystemExit("No glyph records were parsed from the font payload.")

    font_name = inferred_name or args.input.stem
    if ".fnt" in font_name.lower():
        font_name = font_name.split(".")[0]
    vm_outline_segments = _load_vm_outline_segments() if _is_vm_font_name(font_name) else {}
    glyph_objects: List[Glyph] = []
    for component in filtered_components:
        outline_segments = vm_outline_segments.get(component.label) if vm_outline_segments else None
        if outline_segments and len(outline_segments) >= len(component.segments):
            glyph_objects.append(_glyph_from_outline(component, outline_segments))
        else:
            glyph_objects.append(_component_to_glyph(component))
    ordered_pairs = _map_glyphs(font_name, glyph_objects, mode=args.order)
    if args.limit is not None:
        ordered_pairs = ordered_pairs[: args.limit]
    if not ordered_pairs:
        raise SystemExit("Unable to infer character order for the requested glyphs.")

    text = args.text if args.text is not None else "".join(char for char, _ in ordered_pairs)
    font = _build_font_definition(font_name, ordered_pairs)
    metrics = (
        max(args.height, 1e-3),
        0.0,
        0.0,
        0.0,
        max(args.width_scale, 1e-3),
        args.tracking,
        0.0,
        0.0,
    )
    lines = font.render(text, metrics, layer=0)
    if not lines:
        raise SystemExit("Font rendering produced no geometry (text string empty?)")
    write_dxf(lines, [], [], args.output)
    print(
        f"[+] Wrote {len(lines)} line entities from {len(ordered_pairs)} glyph(s) to {args.output} "
        f"(text='{text}')"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
