## Old-style `.mcd` Conversion Game Plan

Legacy drawings in `old_style_mcd_files/` were saved after at least one explode pass 15+ years ago. They include partially exploded lettering, raw component geometry, and Monu-CAD DXF references (both partially and fully exploded). The goal is to rebuild a conversion workflow that faithfully reproduces what Monu-CAD exports today, without manual intervention or pack-data cleanups.

### Objectives
1. **Baseline accuracy** – quantify how far our current `mcd_to_dxf` + `render_component_png` outputs drift from Monu-CAD’s DXF (line counts, layers, missing lettering, etc.).
2. **Robust parsing** – tolerate duplicated or stale records without aborting; gracefully skip garbage created by historical undo/erase artifacts.
3. **Text recovery** – decode partially exploded lettering so editable strings (MAIN / other fonts) still render as geometry even after the first explode pass.
4. **Regression safety net** – bake comparison harnesses so future tweaks can be validated against the preserved Monu-CAD DXF snapshots.

### Task List
- [x] T1: Build a DXF diff tool (`tools/compare_dxf_entities.py`) that counts per-entity deltas (LINE/ARC/CIRCLE/TEXT) and reports layer/name anomalies.
- [x] T2: Run the diff tool across all seven drawings (`Codex DXF` vs matching `MonuCAD *_partially_exploded.dxf` when available, otherwise vs fully exploded export) and capture reports under `notes/diff_reports/`.
- [x] T3: For drawings lacking partial Monu-CAD exports (TW84, TW87, WI36, WI37), inspect fully-exploded DXFs to catalog glyph/component names still referenced so we know which textual assets disappeared. *(See `tools/catalog_dxf_inserts.py` and `notes/component_catalogs/fully_exploded_insert_catalog.txt`.)*
- [x] T4: Reverse the “duplicate record name” failure path by instrumenting `mcd_to_dxf.parse_entities()` to log offending record IDs without throwing; confirm the parser keeps going even if helper data repeats. *(Covered by `--duplicate-log` plus the unified dedupe path.)*
- [x] T5: Add heuristics to `_iter_text_entities()` for legacy payloads where the metrics + font name block was partially exploded (expect orphaned floats followed by glyph labels). *(Glyph-label decoder now turns TLV placement labels (FSKM*/MAINO) back into `TextEntity`s when the metrics look sane.)*
- [x] T6: Teach `FontManager` to fall back on glyph JSON even when the `.fnt` payload can’t be parsed cleanly (guard against partially exploded fonts embedded in the drawing). *(Falls back to `FONTS/components_*` slices when JSON/.fnt parsing fails.)*
- [x] T7: Introduce a “reference DXF” regression suite: one command regenerates Codex DXFs for all samples and compares against the stored Monu-CAD exports with tolerances. *(`tools/run_regression_suite.py` + reports in `analysis/regressions/`.)*
- [x] T8: Update `MCD_Format_Tech_Manual.md` with the old-style workflow, pack-data context, and troubleshooting guidance.
- [x] T9: Extend the dev diary with progress entries so hand-offs remain frictionless.
- [x] T10: Leverage `new_style_mcd_files/` (re-saved Monu-CAD v9 format, un-exploded) as a control group—diff their decoded geometry against both the original “old style” payloads and the Monu-CAD DXFs to pinpoint which layout quirks are unique to the legacy saves. *(Summarized in `notes/new_style_control_summary.md`.)*
- [x] T11: Blend TLV-derived segments with the standard type=2/3 record parsing so arcs/circles survive when both encodings appear in the same payload.

### Immediate Next Steps
1. Implement T1 and capture diffs for at least TW85 + WI35 (have both partially and fully exploded references).
2. Summarize discrepancies to pinpoint which parser gaps hurt the fidelity most (text vs raw geometry vs helper records).
3. Iterate on parser fixes (T4–T6), re-run diffs, and feed insights back into the manual/diary (T8–T9).

### Diff Highlights (current parser vs Monu-CAD)
- TW84: lost **8,221 arcs**, **5 circles**, and shifted all 21k lines off their native layers into `MCD_Layer_0`.
- TW85 (partial reference): Monu-CAD kept 553 arcs + 39 INSERTs; our export flattens everything into 728 lines.
- TW87: missing **1,267 arcs** and **2.6k lines** after flattening; no component/text INSERTs remain.
- WI35: 69 INSERTs plus 3 arcs drop entirely; Codex synthesizes 417 layer-0 lines vs Monu-CAD’s 24 layered lines.
- WI36/WI37: every arc/circle disappears; thousands of lines either go missing or migrate to layer 0.
- YIKES: even the “not exploded” DXF relies on INSERT blocks; Codex turns all geometry into 738 plain lines and loses 36 circles + 838 arcs once the drawing is exploded once.

### Remediation Plan Draft
1. **Blend TLV + record streams** – `_extract_short_component_lines()` currently nulls out `collect_candidate_records()`, which wipes every ARC/CIRCLE helper from these legacy payloads. Instead, treat TLV-derived segments as *additional* linework layered on top of type=2/3 records, gated by how many TLV chunks we see. (Impacts TW84/WI36/WI37/YIKES.)
2. **Legacy arc helpers** – the arc parser assumes helper records trail type=3 entries, but the old files often interleave them with duplicate layer IDs and stale erase records. Need a fuzzy search window that can hop across “duplicate name” padding and keep the arc when at least one helper pair matches the radius. (Targets TW85/TW87.)
3. **Component INSERT preservation** – Monu-CAD DXFs still contain INSERTs referencing lettering/component blocks even after one explode pass. We either need to evaluate the component geometry (preferably via the TLV slices already decoded) or emit lightweight INSERT placeholders so downstream CAD can rebind. (YIKES/WI35.)
4. **Layer naming strategy** – `write_dxf()` invents `MCD_Layer_<id>` which hides the original numeric layer (0/1/2). We should map layer IDs back to their literal strings (when we can recover them) or at least expose both so parity checks are meaningful.
5. **Duplicate record hygiene** – add logging + filters so repeated helper entries do not cascade into bad geometry, mirroring Monu-CAD’s Pack Data cleanup without forcing the user to run it.
6. **Text metrics fallbacks** – partially exploded lettering leaves behind floats + glyph labels but no clean length-pref strings. Extend `_iter_text_entities()` to 1) sniff TLV text blocks, 2) exploit the new glyph JSON caches (MAIN/VERMARCO) so that as long as we recover baseline + width, the text can be redrawn.
7. **Verification harness** – script runner that regenerates DXF/PNG for every old-style file, runs `compare_dxf_entities.py`, and fails CI if counts drift beyond an allowed tolerance. This keeps future fixes honest once parity is closer.

### Latest findings (2025‑11‑12 afternoon)
- TLV blending fix (T11) now keeps the legacy record stream alive even when TLV chunks exist. Old-style DXFs gained hundreds of arcs/circles (e.g., TW85 now reports 176 arcs vs 0 previously; WI35 shows 335 arcs vs 0). Counts still trail Monu-CAD by ~70% because we don’t yet rebuild INSERTs/layer assignments, but we’re no longer throwing away the entire arc helper graph.
- New-style control files (un-exploded re-saves) still expose the same gaps as old-style ones, confirming the missing geometry stems from parser limits rather than just the legacy save process. Diff reports live next to the originals: `notes/diff_reports/new_style_*_vs_monucad.txt`.
- Next coding targets: tackle duplicate record hygiene + INSERT preservation (T4/T3) so we can close the remaining deltas before moving on to text recovery.
- Duplicate-record instrumentation shows just how noisy these payloads are (TW84 new-style logs >200k duplicate geometries; legacy YIKES logs ~8.6k). The warnings now print the first few offending offsets/etype IDs so we can correlate them with Pack Data cleanup behavior when we build smarter filters.
- Built `tools/dump_component_blocks.py` and dumped both the legacy (`notes/component_dumps/TW85_old_style_components.json`) and re-saved (`notes/component_dumps/TW85_new_style_components.json`) component metadata. Key observations:
  * **Old-style TW85** – component id `0x6E694C43` (literally “CLine”) has *no* placement trailers; its geometry only exists as TLV segments, which explains why the partially exploded drawing ends up with ~1.6k segments regardless of how many letters appear in the scene.
  * **New-style TW85** – component id `0x00003802` carries 835 `0x4803` sub-blocks plus a placement trailer named `CRS2`. The placement blob embeds a transformation matrix (still undecoded) that we must apply to stamp the component’s geometry into world space. Without honoring these placements, our DXF only shows the base component sitting at the origin, hence the “close but missing everything” look the user noticed.
  * Going forward we can use the JSON dumps to reverse-engineer the 0x4803/0x0538 payloads and the placement trailers without re-running component_inspector manually.
