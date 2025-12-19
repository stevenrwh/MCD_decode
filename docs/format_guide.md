# MonuCAD Format Guide (WIP)

Quick reference for the structures we actively parse. Keep this updated as we firm up counts/offset rules and retire heuristics.

## Old-style `.mcd`
- Detection: single deflate payload, **no** `CComponentDefinition` marker. `diagnostics/detect_format.py` flags these as “old”.
- Structure: geometry and labels live inline in the decompressed stream; no catalog pattern (`diagnostics/dump_catalog_records.py` finds none). Parsing today relies on TLV scans and inline component chunks rather than sub-block tables.
- Helpers: `mcd_to_dxf.py` legacy path + `_parse_component_place_entries()` for embedded placement hints.

## New-style `.mcd`
- Detection: contains `CComponentDefinition`; usually one large deflate payload. `diagnostics/detect_format.py` and `brute_force_deflate()` locate the stream.
- Component definition: bbox (4×float32) + 7×uint32 header (header[2] = component_id), followed by sub-blocks. Observed tags: `0x4803` (geometry/placement) and `0x3805` (tables). See `component_parser.py`.
- Catalog table: fixed header + ASCII name + `0x075BCD15` sentinel + index (u32le) + tails. 106 entries in new-style TW84. Tool: `diagnostics/dump_catalog_records.py`.
- Placement data: no standalone trailer in TW84; placement-like records live inside specific `0x4803` blocks. Large `0x3805`/`0x4803` tables look fixed-stride; smaller `0x4803` blocks (VM/FSKC/M38) carry the transforms. Tool: `diagnostics/dump_placement_blocks.py --raw`.

## `.mcc` (component libraries)
- Also deflate-backed; contains `CComponentDefinition` blocks with geometry only (no placements). Parsed via the same component/sub-block machinery.
- Used as glyph/component sources in `component_inspector.py`, `render_component_png.py`, and `mcc_server_rendering/`.

## `.fnt` and `.dta`
- MonuCAD font archives; often store multiple deflate streams back-to-back. Locate payloads with `monucad.deflate_io`.
- Sections parsed by `fnt_section_parser.py` / `fnt_to_dxf.py`: glyph strokes (coords scaled by 64), metrics/advances, kerning tables (e.g., `M92_kerning.json` output).
- `font_components.py` and `FontManager` stitch parsed glyphs into renderable line segments; reference DXFs for MAIN/VERMARCO live under `FONTS/`.

## Tooling pointers
- Deflate scan: `monucad.deflate_io` (`collect_deflate_streams`, `brute_force_deflate`).
- Catalog/label: `diagnostics/dump_catalog_records.py`, `diagnostics/dump_label_context.py`, `diagnostics/dump_3805_table.py`.
- Placement: `placement_parser.py`, `diagnostics/dump_placement_blocks.py`.
- Components: `component_parser.py`, `tools/dump_component_blocks.py`.
