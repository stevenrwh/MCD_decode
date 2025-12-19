## 2025-03-02

- Wired up `tools/catalog_dxf_inserts.py` to snapshot BLOCK/INSERT usage from the stored Monu-CAD DXFs. The fully exploded references for TW84/TW87/WI36/WI37 come back empty, so we saved the output under `notes/component_catalogs/` for T3.
- Extended `mcd_to_dxf.py` with `--duplicate-log`, unified the duplicate filter, and capped each entry with offsets/layer info so Pack Data noise can be diffed run-to-run.
- Added glyph-label heuristics (TLV placement parser) with sanity-checked metrics to `_iter_text_entities()` and taught `FontManager` to fall back to the sliced component bins when `.fnt` parsing fails. This keeps MAIN/VERMARCO plus Press Modified Roman usable even when the embedded fonts are partially exploded.
- Dropped the first version of `tools/run_regression_suite.py`; it regenerates DXFs for the legacy samples and writes comparison reports into `analysis/regressions/` so we can track entity deltas without hand-cut commands.
- Captured the new-style control deltas in `notes/new_style_control_summary.md`—they match the old-style losses, which confirms the parser (not the save path) is at fault.
