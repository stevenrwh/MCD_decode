# MonuCad Reverse Engineering – Current State (Single Source of Truth)

This file replaces the scattered notes. Archived docs now live in `docs_archive/`.

## 1) File Types & Detection
- **New-style MCD**: Contains `CComponentDefinition` marker and typically a single deflate payload embedded in a “gzip-looking” file. Example: `new_style_mcd_files/TW84.mcd` (110,584 bytes; payload ≈ 632 KB).
- **Old-style MCD**: Usually a single deflate payload with **no** `CComponentDefinition` marker. Example: `old_style_mcd_files/TW84.mcd` (95,195 bytes; payload ≈ 338 KB).
- Heuristic: use `diagnostics/detect_format.py` (checks markers + deflate streams).

## 2) Catalog Table (new-style TW84)
- Structured records live in the decompressed payload; not visible in raw bytes.
- Pattern (big-endian header, then little-endian sentinel/index):
  - `u32be 1`, `u32be 0`, `u32be 0`
  - `u32be name_len`, ASCII name (no padding)
  - `u16be 0x0180`
  - `float32le x0, y0, x1, y1` (component metrics/bounds)
  - `u32le 0x075BCD15` sentinel
  - `u32le index` (observed 0–38)
  - `u16be tail2` (often 568/840/1336, etc.)
  - `u32be tail3` (commonly 2 or 0)
- Tool: `diagnostics/dump_catalog_records.py` (outputs offset, name, index, floats, tails). New-style TW84: **106 catalog entries**, names match MonuCad’s component list (FSK*, MAIN*, M38*, VM*, etc.). Old-style TW84: this pattern does **not** appear.

## 3) Component Geometry Blocks (0x4803)
- New-style TW84 has **3,943** 0x4803 blocks inside the lone component definition (ID 14338). Many blocks are short (len 65) containing int16 polyline records; some longer blocks carry embedded sentinel + index.
- Block-to-index mapping: some 0x4803 payloads include `0x075BCD15` followed by a uint32 index (same index space as the catalog). Observed indices: 1–38 map to specific blocks; far fewer indices than blocks.
- Current strategy (WIP):
  - Build `index -> block` using the embedded sentinel indices.
  - Build `label -> index` from the catalog table.
  - Map `label -> block` via `label -> index -> sentinel-bearing block`.
  - Fallbacks (substring search, ordered mapping) remain but are noisy.

## 4) Placement Data
- `iter_placement_trailers` currently finds **zero** trailers in new-style TW84; placement records live inside specific 0x4803 sub-blocks of the lone component definition instead of a standalone trailer.
- `diagnostics/dump_placement_blocks.py` reports block-local candidates. For TW84 (`--raw`):
  - 0x3805 block idx 780: 623 records, looks like a fixed-stride label table (ignore for placement).
  - 0x4803 MAIN* blocks: idx 1638 (489 recs, tx≈[-147,-94], ty≈[-20,-11]); idx 1659 (493 recs, tx≈[-27,26], ty≈[-18,-6]); idx 3786/3879/3889 (≈490 recs each at other offsets); idx 1675 (126 recs, mostly M38* labels).
  - Smaller 0x4803 blocks: idx 2907 (69 recs, VM* at y≈1), idx 3806 (69 recs, VM* at y≈50.75), idx 2927 (68 recs, FSKC* spanning y≈4–64).
- A full payload scan produces ~3.5k glyph-like hits; block scoping is required to avoid instancing everything.

## 5) Current Converter Status (TW84)
- Reference DXF (`TW84_exported_from_MCPRO9.dxf`): ~1,195 lines / 13 arcs / 3 circles.
- Current run (Dec 19 block-fallback build): 4,906 lines / 0 arcs / 2 circles. Block-scoped placement parsing instantiates far too many lines and suppresses arc promotion.
- Legacy inline-only path (no placements): 611 lines / 11 arcs / 3 circles. Old-style TW84 path: ~298 lines (too low).

## 6) Diagnostics Toolkit
- `diagnostics/detect_format.py` – old vs new-style hint.
- `diagnostics/dump_catalog_records.py` – decode catalog entries (name → index).
- `diagnostics/map_catalog_ids.py` / `scan_label_catalog.py` / `extract_label_catalog.py` – legacy label scans (length-prefixed ASCII in payload).
- `diagnostics/dump_label_context.py` – hex/int16 context around label hits.
- `diagnostics/search_labels*.py` – quick label presence checks.
- `component_parser.py` – parses component definitions, TLV primitives, and sub-blocks.
- `placement_parser.py` – extracts placement trailers and glyph records.

## 7) Known Structures (new-style TW84)
- Component definitions: 1 definition (ID 14338). Sub-block counts: 3,943 × 0x4803 + 2 × 0x3805. Definition bbox is present.
- 0x4803 short blocks: repeated 8-short records interpreted as normalized coord pairs (with occasional sentinels). Some blocks store sentinel + index internally.
- Catalog floats: plausible component metric/bounds (often symmetric small values).

## 8) Next Steps (no guessing)
1) **Placement source selection**: Identify which 0x4803 blocks hold the real placement tables (likely the small VM/FSKC blocks) and wire `mcd_to_dxf.py` to consume only those, not the full 3.5k payload scan or the fixed-stride 0x3805 table.
2) **Block/caption mapping**: Keep catalog-first label→block pairing, but restrict geometry instancing to the chosen placement blocks to avoid duplication.
3) **Arc recovery**: Once instancing is correct, re-enable arc reconstruction where triplets make sense (with bbox sanity).
4) **Old-style path**: Build an equivalent catalog/label parser for old-style TW84 (different pattern), then map to geometry blocks.
5) **Validation**: After fixes, compare entity counts/PNGs against `TW84_exported_from_MCPRO9.dxf`; repeat for other samples (JF6050, GROVES, etc.).

## 9) Misc Facts
- Catalog names are only in the decompressed payload (not raw file).
- BBox sanity: use drawing/component bbox to filter out geometry that exceeds expected extents.
- `diagnostics/dump_placement_blocks.py` is the quickest way to see which 0x4803 blocks carry placement-style records.
- Deflate scanning lives in `monucad/deflate_io.py`; `monucad/entities.py` now holds core entity dataclasses as part of the ongoing modularization of `mcd_to_dxf.py`.
- Font logic now lives in `monucad/fonts.py`; `mcd_to_dxf.py` imports TextEntity/Glyph/FontDefinition/FontManager and related constants from there (inline font classes removed).

Keep this file as the canonical knowledge base. All older docs live in `docs_archive/`. Update this file as discoveries are made; avoid adding new scattered notes.
