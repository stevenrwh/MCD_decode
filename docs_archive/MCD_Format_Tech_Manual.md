# Monu-CAD v9 `.mcd` Technical Notes

This document summarizes everything we currently know about the Monu-CAD v9 binary drawing format, the deliberate obfuscation/compression tricks it uses, how we peeled those back, and how the `mcd_to_dxf.py` extractor turns the recovered geometry into DXF entities.

---

## 1. High-Level File Layout (v9 samples)

| Offset (hex) | Bytes / Value                            | Notes                                                                                     |
|--------------|-------------------------------------------|-------------------------------------------------------------------------------------------|
| `00`         | `1F 8B 08 00 00 00 00 00 00 0B`           | Looks like a gzip header (magic + method + zeroed fields) but the rest is **not** valid.  |
| `0A`         | ASCII `"MCD2"`                            | Format tag.                                                                               |
| `10–1F`      | Doubles/floats (`A0 C0 00 00 ...`)        | Likely drawing extents and header metadata.                                               |
| `20–40`      | ASCII `"Created with MONU-CAD Pro v9.2.10"` | Plain-text banner; proves we’re inside the right container.                               |
| `45`         | `1F 4C 35 79 ...`                         | Earliest location where a **valid raw deflate stream** can be found.                      |
| `45+`        | Deflate payload (~500–700 decompressed bytes) | Contains ASCII configuration text followed by packed binary geometry.                     |

Observations:

1. The `.mcd` file masquerades as a gzip file but is intentionally corrupted. Attempting to run `gzip -d` or `zlib.decompress` on the entire blob fails with `invalid block type`.
2. The real drawing data is a standalone deflate stream that starts inside the file (offset `0x45` in the sample). Because the gzip header is bogus, the canonical decompressor never reaches it.

---

## 2. Obfuscation & Compression Strategy

1. **Fake GZIP Envelope**  
   - The first 10 bytes mimic a gzip header to mislead casual inspection tools.  
   - Immediately after the header the file switches to junk bytes; there is no valid gzip footer.

2. **Hidden Raw Deflate Stream**  
   - A legitimate DEFLATE block exists later in the file but is not referenced by any header pointer.  
   - The only way to reach it is to *brute-force* every offset with `zlib.decompress(..., wbits=-15)` until one succeeds.

3. **Lightweight Data Mixing**  
   - The decompressed payload begins with human-readable settings (`PolishTile = TRUE`, etc.), possibly to hide the binary tables deeper in the stream.  
   - Binary sections mix 8-bit, 16-bit, 32-bit, and 64-bit little-endian values without explicit block headers; the meaning must be inferred from patterns.

No additional encryption was detected once the deflate stream is located; the contents are plain.

---

## 3. Structure of the Decompressed Payload (v9)

The payload from `10x10_rect.mcd` (after deflate) splits into two logical segments:

1. **INI-like Configuration Text** (offset `0x00–0xF0`)  
   Example:
   ```
   PolishTile = TRUE
   SteeledTile = TRUE
   RockPitchTile = FALSE
   BluedTile = TRUE
   SingleLineThickness = 0.1000
   FrostedLineThickness = 0.2500
   ```
   This section is terminated by `\r\n` pairs followed by two null bytes.

2. **Binary Geometry Tables** (offset `0xF0+`)  
   The remainder consists of repeated fixed-length records.

### 3.1 Geometry Record Layout (Lines)

Sliding a 40-byte window across the binary portion revealed consistent hits that decode cleanly to little-endian doubles with values `{ -5, 0, 5, 10 }`. Aligning on those hits exposes the record template:

```
struct LineRecord {
    uint32 layer_id;      // e.g. 1
    uint32 entity_type;   // 2 == straight line
    double x1;
    double y1;
    double x2;
    double y2;
};
```

*Example @ offset `0x106` (decimal 262) from the sample file:*

```
layer_id     = 0x00000001
entity_type  = 0x00000002
(x1, y1)     = (-5.0, 0.0)
(x2, y2)     = ( 5.0, 0.0)
```

This describes the bottom edge of the 10×10 rectangle. Three more consecutive records describe the remaining vertical/horizontal edges. All coordinates are double-precision floats (IEEE754, little-endian).

### 3.2 Other Blocks

Before each cluster of line records there are 32-byte zones filled with values such as `0x00000238`, `0x000014C0`, `0x00002440`. These decode to either small integers or doubles equals to `±5`/`10`. They may encode:

- Drawing extents (min/max X/Y), repeated for each entity block.
- Layout- or tile-related parameters carried over from the textual section.

The overall pattern suggests that **other entity types** (arcs, polylines) likely reuse the same `layer_id + entity_type` preamble with different payload lengths. At present we have only confirmed type `2` lines.

### 3.5 Automated Security Sweep

- `tools/mcpro9_security_sweep.py` drives MCPro9 through the canonical “LI” macro, then saves the same geometry once per File Security option (`line_all.mcd`, `line_specific.mcd`, `line_only.mcd`, `line_masters.mcd`). The dialog exposes push-buttons (not radio buttons), so the script explicitly clicks `&All Your Satellites`, `&Specific Satellite`, `&Only You`, or `&Masters Only` before accepting it.
- Always save into MCPro9’s default Drawing Save directory to avoid the zero-byte stubs we observed when typing arbitrary absolute paths. After the sweep completes, copy the generated `.mcd` files out of the Monu-CAD workspace and run `python mcd_section_parser.py <file> --json analysis/<mode>_header.json` to diff the 7×9 table for each security mode.
- Use the new dialog screenshots under `analysis/security_sweep/` to confirm the UI sequence (Save As → Drawing Save → File Security Options). These captures proved that Frida does not interfere with the UI when we let MCPro9 manage the save path.
- When you need realtime visibility into the UI state, pass `--screen-feed` to the sweep script. This records every dialog rectangle to `analysis/security_sweep/with_frida/feed/…` and, when combined with `--screen-preview`, opens an OpenCV window that mirrors the active dialog so you can verify cursor focus or button presses live.

---

## 4. Reverse Engineering Approach

1. **Hex Inspection**  
   - `Format-Hex 10x10_rect.mcd` shows the fake gzip header, ASCII banner, and heavy noise.

2. **Brute-force Deflate**  
   - Ran Python loop calling `zlib.decompress(data[i:], -zlib.MAX_WBITS)` for each offset.  
   - First successful payload was at offset 69 (`0x45`) with decompressed length 669 bytes.

3. **Payload Dumping**  
   - Saved the deflated bytes to `10x10_rect.decompressed`; validated INI portion and binary tail.

4. **Type Guessing**  
   - Interpreted the binary tail as sequences of 16/32-bit ints and 64-bit doubles until meaningful values (±5.0, 10.0) emerged.
   - Confirmed that a sliding 40-byte window with the `<II dddd>` pattern yields four unique line segments forming the known rectangle.

5. **DXF Validation**  
   - Transcribed the recovered lines into a minimal DXF via a script, opened it in external CAD to confirm geometry fidelity.

---

## 5. Decoder Script (`mcd_to_dxf.py`)

Key stages inside the script:

1. **Load & Scan** (`brute_force_deflate`)  
   - Reads entire file into memory.  
   - Sequentially tries every byte offset with raw zlib (`wbits=-15`).  
   - Accepts the first decompression that produces `>=128` bytes (configurable).

2. **Geometry Extraction** (`extract_linework`)  
   - Slides through the payload looking for `<uint32 layer, uint32 type>` where `type==2`.  
   - Parses the next 32 bytes as four doubles.  
   - Filters out non-finite values and degenerate zero-length lines.  
   - Deduplicates segments using `(layer, start, end)` as the key.

3. **DXF Writer** (`write_dxf`)  
   - Emits a minimal DXF `SECTION ENTITIES` block.  
   - Each extracted line becomes a DXF `LINE` with `layer` named `MCD_Layer_<id>` and Z forced to `0`.  
   - ARC records (etype `3`) grab the next helper record to recover center/end coordinates and are emitted as DXF `ARC` entities with computed start/end angles.  
   - Coordinates/angles are formatted at 1e-6 precision to avoid DXF parser complaints.

4. **CLI** (`main`)  
   - Reports which offset contained valid data and how many entities were found.  
   - Default output path is `<input>.dxf`; override with `-o`.  
   - Optional flags `--start-offset`, `--stop-offset`, `--min-payload` let you tune scanning on larger jobs.


### 5.1 Quick Start (No prior context required)

1. **Open a terminal inside `C:\Dev\MCD_decode`.**
   Every script referenced below already lives there.
2. **Ensure Python 3.11+ is installed** and that the `python` command works from that terminal.
3. **Convert any `.mcd` file to DXF:**
   ```powershell
   python mcd_to_dxf.py path\to\drawing.mcd
   ```
   - Output defaults to `path\to\drawing.dxf`; supply `-o new.dxf` to override.
   - The script prints the deflate offset and the number of LINE/ARC entities recovered.
4. **Inspect legacy payloads (optional):**
   ```powershell
   python analyze_v6_payload.py
   ```
   This highlights which byte offsets change across the `line_0,0_to_*` samples.
5. **Validate the DXF** in any CAD viewer (AutoCAD, LibreCAD, etc.).

Because the script focuses on entity type `2`, expanding it to handle additional primitives should only require teaching `extract_linework` more parsing functions keyed on `entity_type`.

- **Why line-only drawings are easier:** as long as a file only stores entity type `2` segments, every geometry record is homogeneous and translates directly to a DXF `LINE`. Once fonts, text, or reusable blocks enter the mix, the payload interleaves additional entity types (most likely splines, polylines, attribute tables, and block definition dictionaries). Those structures will require bespoke parsers and DXF writers before we can guarantee fidelity.

---

## 6. Older "v6" Files Behave Differently

The corpus now includes many `.mcd` files created by *Monu-CAD Pro v6.3.3*. These files share the same fake gzip front matter, but the payloads differ dramatically:

- The ASCII banner reads `"Created with MONU-CAD Pro v6.3.3 Windows 95 v1.00.12"`.
- The deflate stream starts much later (e.g., offset `0x183` for `10x10_try2_rect.mcd`) and inflates to ~900 bytes.
- Only two `<layer, type>` headers are present: `(9,1)` and `(1,1)`. **No `(1,2)` line records exist**, which is why the current decoder only sees the outer rectangle when run on v6 files: the inner linework is encoded in a different structure.
- Immediately after the textual section lies a repeating block full of `0x00000238`, `0x000014C0`, `0x00002440`, `0x00FFFFFF`, etc. This appears to be a packed table that mixes 16-bit counters, 24-bit colors, and fixed-point coordinates. It never occurs in the v9 payload.

**Implication:** there are at least two `.mcd` dialects. After removing the `layer==0` filter we learned that the legacy files *do* carry the familiar `(type=2)` line records—they simply place them on `layer 0`, which is why the first pass missed them. Arc-only drawings, on the other hand, encode their geometry as `(type=3)` records followed by a few `(type=0)` helpers containing the center/end coordinates. The current decoder now handles both cases, but other primitives (text, blocks, etc.) may still rely on additional helper records hidden in that first 0x100-byte block.

## 7. Temp Files: Capture Attempts

We tried to piggyback on MCPro9's temp artifacts (Process Monitor shows it creating `McdXXXX.tmp`/`.CFG` under `%LOCALAPPDATA%\Temp` while loading drawings). A PowerShell watcher script now:

1. Spawns MCPro9 in a suspended state via `CreateProcess(..., CREATE_SUSPENDED)`.
2. Resumes it for a configurable window, then suspends it again while the watcher copies any `Mcd*.tmp/.cfg` into `captured_temp\`.

Despite automating the launch/suspend cadence, we were not fast enough to catch the expected temp files when opened by double-clicking an `.mcd`. For now we are shelving the temp-file angle and focusing on reversing the on-disk payload directly.

## 8. Outstanding Questions & Next Steps (Actionable)

1. **Collect More Entity Samples**
   - Target drawings containing text, blocks, splines, dimensions, or anything beyond straight lines/arcs.
   - For each sample, archive the original `.mcd`, the exported DXF, and (if possible) the intended geometry description.

2. **Decode Additional Entity Types**
   - Re-run `analyze_v6_payload.py` on the new samples to spot which bytes change.
   - Extend `parse_entities` with new handlers once patterns emerge (e.g., `etype == 4` for text, etc.).

3. **Document Header Semantics**
   - The 0x100-byte “mystery block” likely holds extents, layers, and defaults. Map each field by diffing the minimalist samples.
   - Update this manual once fields are named so future work doesn’t repeat the same exploratory steps.

4. **Support Multiple Deflate Regions**
   - Some large `.mcd` files may contain more than one embedded deflate stream. Enhance `brute_force_deflate` to collect *all* viable payloads and parse them in sequence.

5. **Investigate Integrity / Round-tripping**
   - Look for CRCs or checksums in the header that might be required to rebuild `.mcd` files.
   - Determine whether the INI-style configuration block influences geometry or is purely cosmetic; this is required if we ever want to write `.mcd` back out.

---

## 9. Summary

- `.mcd` files spoof a gzip header but hide their real content as a raw deflate stream deeper in the file.  
- The decompressed payload mixes ASCII configuration with binary entity records.  
- Line entities follow a clear `<layer, type, 4xdouble>` pattern, and `(type=3)` arcs come with helper records that encode the center/end points—both are now decoded for DXF conversion.  
- The `mcd_to_dxf.py` script automates the brute-force decompression and DXF emission, so both legacy (layer 0) and modern `.mcd` files can be cracked and converted reproducibly.

## 10. `.mcc` Component Files

`M9_RGB_COMPONENT.mcc` uses the same fake `MCD2` header and hidden deflate stream (payload offset ~0x183) but the payload is entirely different:

- Inflated size ≈22 KB, no INI text or `<layer,type>` structures.
- Only notable ASCII strings are `ComponentDefinition` (near the start) and `M9_RGB_COMPONENT` (near the end); everything else is opaque binary.
- No nested deflate streams or obvious raster headers (`BM`, `PNG`, etc.). This indicates `.mcc` is a separate Monu-CAD resource format (likely color/bitmap data). Converting it to PNG/DXF will require fresh reverse-engineering; the current `.mcd` decoder does not apply.

## 11. Arc & Circle Diagnostics (v9)

Modern v9 files surround every `(type=3)` arc with a dense cluster of helper records (layers `0/64/72`, etypes ranging from `0` to 30-bit integers). To study those helpers without writing ad hoc scripts each time:

1. **In-script logging.** `mcd_to_dxf.py` now supports `--dump-arc-helpers dump.txt` (optionally `--arc-helper-window 20`). This writes each arc's payload coordinates plus the next *N* raw records (offset, layer, etype, both point pairs). It's the fastest way to see the exact helper stack Monu-CAD emits before/after every arc.
2. **DXF comparison.** `diagnose_arc_helpers.py` cross-references a `.mcd` file with a trusted DXF export of the same drawing, matches each type-3 record to the corresponding DXF `ARC`, and prints a report that includes the DXF start/end coordinates and which helper records might reference those points. The tail of the report highlights helper records that seem to describe true circles (e.g., the 6" outline centered at (0,0)).

Example workflow:

```powershell
python mcd_to_dxf.py drawing.mcd --dump-arc-helpers arc_dump.txt
python diagnose_arc_helpers.py drawing.mcd drawing_exported_from_monucad.dxf -o arc_helper_report.txt
```

Together these tools shrink the guess/test loop for arc decoding: capture the helper dump, align it with the DXF numbers, then teach `parse_entities` which helper combination stores the missing end points or circle metadata.

---

## 12. `.mcc` Component Files

`M9_RGB_COMPONENT.mcc` and similar archives reuse the fake `MCD2` header but hold opaque TLV blobs rather than the `<layer,type>` records found in `.mcd`. Things we know so far:

- The payload starts with a `CComponentDefinition` TLV whose children are nested `CLine`/`CArc` entries. Every line/arc is encoded as a TLV whose payload is a run of doubles.
- No `<layer,type>` records exist, so any tooling that assumes the `.mcd` layout will emit spikes/noise.
- Some component packs (e.g., `piano keys.mcc`, `RK39.mcd`) only appear via these TLVs, so the TLV walker is mandatory before we can decode the geometry reliably.

Action items:

1. Detect TLV-only payloads (`ComponentDefinition` string immediately followed by nested TLVs). 
2. Walk each TLV, decode the numeric payload, and emit `LineEntity`/`ArcEntity` objects. 
3. Feed those primitives back into `mcd_to_dxf.py` / `render_component_png.py` before falling back to exported DXFs.

Files affected so far: `piano keys.mcc`, `RK39.mcd`, and most stock component packs. Until the TLV walker is feature-complete, these assets will continue to rely on exported DXFs.

---

## 13. Fonts & Text Entities (WIP)

Our lettering fixtures (`TEST_WITH_FONT.mcd`, `TEST_VERMARCO_mcalf020.mcd`) forced us to inventory the font ecosystem. Key findings:

### 13.1 Font lookup files
- `FONTS/mcfonts.lst` acts as an INI registry: sections like `[MAIN]` point to `dtafile = m92`, `fontfile = mcalf092`, etc. This is how Monu-CAD maps a user-visible font name to the `.fnt`/`.dta` pair sitting next to the drawing.
- `.dta` files are 95×95 grids of `uint16` values. Entry `0` is always `1` (format flag). Most entries read `1000` (neutral advance) or `0`, but clusters carry unique values that match the kerning pairs observed in Monu-CAD. `analyze_m92_dta.py` reshapes the grid and emits JSON for scripting.

### 13.2 `.fnt` archives
- Fonts such as `FONTS/Mcalf092.fnt` and `FONTS/Mcalf020.fnt` reuse the fake `.mcd` shell; the hidden deflate stream inflates into TLV soup containing `CComponentDefinition`, `CLine`, `MAINa`, `M92A`, etc.
- `glyph_tlv_parser.py` now walks every `CComponentDefinition` chunk, decodes the record arrays, and produces structured glyph payloads (label, bbox, baseline, advance, grouped line segments, header ints).
- `generate_main_font_assets.py` consumes those components and builds `FONTS/MAIN_glyphs.json` / `FONTS/MAIN_glyphs.dxf`, which FontManager uses when rendering MAIN text. If JSON is missing we fall back to the TLV/DXF heuristics, but the default toolchain now matches Monu-CAD for MAIN.

### 13.3 Vm / Vermarco specifics
- We captured `FONTS/Mcalf020.fnt`, `FONTS/Vm.dta`, and matching DXF exports (`FONTS/Mcalf020_exported_from_monucad.dxf`). The DXF exposes the authoritative BLOCK geometry while we reverse the `.fnt` blobs.
- `tail_vm_offsets.json` enumerates the tail tuple → offset ranges discovered so far. Most glyphs rely on `(0, 256, 0, 768)`; punctuation pulls from `(0, 256, 0, 1792)` or rarer tuples like `(-21504, -7304, 189, 5824)`.
- `export_vm_strokes.py`, `generate_vm_font_assets.py`, and FontManager currently fall back to the exported DXF when the `.fnt` parser can’t isolate the proper strokes. The Vermarco flow is therefore a *temporary crutch* until we decode the metadata described below.

### 13.4 TLV helper scripts
- `catalog_font_labels.py` – finds label offsets in the `.fnt` payload so we can cut reliable slices.
- `extract_font_components.py` – writes each glyph blob to `FONTS/components/###_LABEL.bin`.
- `font_components.py`, `summarize_component_headers.py`, `summarize_font_components.py` – parse the sliced blobs, report per-glyph stats, and highlight header patterns.
- `export_vm_strokes.py` – dumps each tail tuple’s raw stroke geometry to `FONTS/VM_tail_records.json`.
- `map_tail_offsets.py` – correlates component tails with the raw stroke offsets, producing `tail_vm_offsets.json`.

### 13.5 Vm diagnostics (new)
- `analyze_vm_component_chunks.py` slices every Vm component record into its 12-int chunks, records the metadata + raw/transformed coordinates, and writes `vm_chunk_dump.json`.
- `summarize_vm_meta.py` aggregates those metadata quartets and emits `vm_meta_summary.json`, showing which tails/glyphs reuse the same chunk metadata.
- `match_vm_chunk_tail_candidates.py` cross-references the chunk dump with `Mcalf020.fnt.decompressed`, scores the nearest tail offsets (top N per segment), and saves the result as `vm_chunk_match_candidates.json`. Each row captures the metadata quartet, raw/transformed bounding boxes, candidate offset span, and squared distance so we can finally see how the component metadata lines up with the real tail strokes.
- These dumps prove that each chunk carries two pairs of local coordinates (8 ints) plus four metadata ints, but the transformed coordinates still live in tail space. The new matcher shows which tail offsets are *closest* to each chunk (1460 chunks total), but the distances are still large, so the next step is to collapse those spans into deterministic metadata → tail index ranges.

### 13.6 Text payloads
- `inspect_text_entity.py` decodes the editable-text blob inside `TEST_WITH_FONT` / `TEST_VERMARCO`. Each entity stores the string, eight floats (height, width scale, baseline spacing, italic flag, horizontal scale, tracking, insert X/Y), and a font ID string.
- Once the glyph pipeline becomes 100% `.fnt`-driven, the text walker will instantiate glyphs directly from those metrics without exploding the text inside Monu-CAD.

### 13.7 Kerning + metrics
- `analyze_m92_dta.py`, `analyze_vm_dta.py`, etc., convert the `.dta` grids into sparse JSON kerning tables so FontManager can apply real spacing.

### 13.8 Outstanding work
- Map every metadata quartet to an explicit stroke subset in `VM_tail_records.json`.
- Replace the external `Vm_tail_transform_stats.csv` calibration with formulas derived from the component headers.
- Remove the DXF fallback once the mapping + formulas exist.

---

## 14. Handover / What To Do Next

The open work now falls into three tracks. Tackle them roughly in this order:

### 14.1 Component decoding & DXF parity
1. **Decode the `0x0538` sub-blocks.** Use `component_inspector.py` (and the JSON snapshot we saved for FACE) to map each double run to real geometry. The goal is to recover the two missing 0.5-unit circles without relying on Monu-CAD.
2. **Hook the decoder into `mcd_to_dxf.py`.** After you can emit `CircleEntity` objects from a `CComponentDefinition`, teach the converter to (a) detect embedded component blocks, (b) parse them once per payload, and (c) merge their primitives whenever the matching placement trailer appears.
3. **Add regression fixtures.** Once FACE renders correctly, capture the Monu-CAD DXF as `reference/MCD_CONTAINS_FACE_COMPONENT.dxf` and add it to the smoke-test list so future edits cannot reintroduce the circle gap.

### 14.2 Fonts & text pipeline
1. **TLV component decoder.** ✅ `mcd_to_dxf` now uses `iter_label_chunks()` + `font_components.parse_component_bytes()` to decode the `CLine`/`CArc` blobs embedded inside TLV-only drawings, so RK-class assets finally emit real linework without depending on an exported DXF. Still open: curve primitives (`CArc`/`CCircle`), nested placements, and piping the decoded geometry into the PNG renderer.
2. **Map Vm metadata to tail strokes.** Leverage `vm_chunk_dump.json`, `vm_meta_summary.json`, `vm_chunk_match_candidates.json`, and `VM_tail_records.json` to correlate every 12-int chunk (metadata + coords) with the actual stroke sequence it references. Use the matcher output to cluster the offset spans, tighten the scoring, and then promote the proven ranges into code so we end up with a hard mapping (metadata quartet → tail index range).
3. **Vm glyph rendering from `.fnt`.** After that mapping exists, update `generate_vm_font_assets.py` so it selects the proper tail strokes per chunk, applies the recovered transforms, and emits LINE/ARC entities directly from the `.fnt` data. This removes the dependency on `FONTS/Mcalf020_exported_from_monucad.dxf`.
4. **Header decoding (MAIN + Vm).** Correlate header slots `h16..h26` against the affine transforms captured in `Vm_tail_transform_stats.csv` (and the MAIN component stats). Document the formulas and bake them into the generator so the CSV calibration step goes away.
5. **Text TLV automation.** Generalize the font-aware parser into a TLV walker that can locate the text record, pull the font id, and decode the float metrics without hard-coded offsets (`TEST_WITH_FONT` / `TEST_VERMARCO` are the templates).
6. **Kerning + metrics bridge.** Load the proper `.dta` grid for each font referenced in `mcfonts.lst`, convert the 95×95 matrix into advance/kerning tables, and feed that into the renderer so the spacing matches Monu-CAD.
7. **Deduplicate labels.** Introduce stable glyph identifiers (e.g., `001:VM0`) in JSON/CSV exports so repeated labels no longer collide downstream.
8. **Regression fixtures.** After `.fnt`-driven glyph reconstruction works, capture clean reference DXFs and add them to the smoke test suite next to the MAIN fixtures (still keep the Monu-CAD DXFs around solely for verification). Include the JSON dumps (tail offsets, chunk summaries, metadata maps) so we can diff Vm layout changes over time.

### 14.3 Tooling & workflow
1. ~~**Document component offsets.** Extend `component_inspector.py` so it optionally dumps the per-chunk byte ranges.~~ (Done - the CLI printout and JSON now include `off=0x... len=...` for each component.)
2. **Expand helper diagnostics.** Automate the `mcd_to_dxf --dump-arc-helpers` + `diagnose_arc_helpers.py` combo so every regression run captures helper stats for FACE (components) and the font samples (glyph strokes).
3. **Smoke-test wiring.** Add the new inspector JSON and arc-helper logs to the regression checklist in Section 9 so we notice immediately when an upstream change alters the TLV layout.
4. **Format fingerprinting.** Teach the tooling to flag TLV-only payloads (no `etype=2/3` records, repeated `CLine` TLVs) vs. the legacy record layout so we can pick the correct decoder automatically and surface actionable warnings instead of silently emitting garbage DXFs.

### 14.5 Daily engineering diary ("ritual")
To keep hand-offs smooth and prevent context loss:
1. Maintain `notes/dev_diary.md` (or an equivalent markdown log). Every working session gets an entry with date, short summary of what was attempted, samples touched, outstanding blockers, and explicit next steps.
2. When a breakthrough happens (new entity decoded, regression fixed, parser insight, etc.), record it in the diary **and** link to the relevant section of this manual so the knowledge is captured twice.
3. Before stopping for the day, re-read the previous two entries and confirm the "next steps" list is still accurate. If priorities changed, update both the diary and Section 14 so future agents don't repeat obsolete experiments.
4. When handing over to someone else, point them at the latest diary entry plus any open checklist items so they can resume without rereading the full history.

### 14.4 Rendering pipeline
1. **Nested components / fonts.** `render_component_png.py` can only draw whatever `parse_entities()` understands today. As soon as we decode nested component placements or font glyph references we need to feed that back into the renderer so complex `.mcc` files (component packs, lettering assets) no longer appear as partial outlines.
2. **Batch + metadata.** Add a helper that walks a directory of `.mcd/.mcc` files and emits both preview + hi-res PNGs with consistent naming, DPI metadata (300 dpi for print), and optional background colors.
3. **Styling hints.** Once we decode layer styles/line weights from the payload, extend the renderer to honor stroke thickness or color (helpful for granite texture mockups). For now everything is a uniform black stroke.

### 14.6 Old-style legacy conversion testbed
We now own a dedicated sandbox under `old_style_mcd_files/`. Every drawing inside that folder was saved 15–16 years ago **after at least one explode pass**, so editable lettering has already been broken into component INSERTs and some components have been reduced to raw geometry. Monu-CAD also left stale, undeleted records in the payload—`Pack Data` cleans them up in the app, but we must tolerate them in our parser.

Checklist for working with these files:
1. **Inventory first.** Keep `notes/old_style_inventory.md` in sync whenever new samples/exports are added (Monu-CAD DXFs, Codex DXFs, PNG previews). This is the canonical manifest for the testbed.
2. **Use Monu-CAD as ground truth.** Each drawing has at least one DXF exported from Monu-CAD (some have both “partially exploded” and “fully exploded” captures). Treat those as the reference geometry until the parser matches them within tolerance.
3. **Run the DXF diff harness.** `python tools/compare_dxf_entities.py REF.dxf CANDIDATE.dxf --report notes/diff_reports/<name>.txt` produces per-entity and per-layer counts. Wire this into every regression pass so we know exactly how many arcs/INSERTs/circles were lost.
4. **Remember Pack Data semantics.** The “Duplicate record name” exporter error was caused by decades of erased-but-not-packed entities. Our converter must (a) skip duplicates instead of aborting and (b) optionally surface a warning that Pack Data would have removed the extra helpers.
5. **Document remediation steps.** Use `notes/old_style_conversion_plan.md` to capture the active task list (TLV/record blending, INSERT preservation, text heuristics, etc.) and link each milestone back to this section plus the dev diary.

Success criteria for this track: Codex DXFs/PNGs for TW84, TW85, TW87, WI35, WI36, WI37, and YIKES match the Monu-CAD exports (entity counts + layer names) without manual cleanup or Pack Data pivots, and the regression harness enforces that going forward.

### 14.7 New-style component placements (Monu-CAD v9 re-saves)
Recent `.mcd` saves (see `new_style_mcd_files/`) no longer explode lettering into raw LINE entities. Instead, each payload contains:

1. **One giant `CComponentDefinition` block** – hundreds of `0x4803` sub-blocks that reuse the MAIN glyph encoding (28 int16 header slots followed by stroke records delimited by sentinel `0x8003`). You can parse them with `font_components.parse_component_bytes()` and feed the segments into `_segments_from_short_chunk()`.
2. **A single placement trailer** – names like `CRS2`, `KC551B`, `MAIN6` (the last one is just a MAIN-font component). Each trailer begins with `0x075BCD15` (magic), instance id, component id, then ~256 bytes of TLV-style data. Every numeric field is stored as a 16.16 fixed-point value (divide the 32-bit integer by 2^32 to get the unit-range float). Repeating tags (`0x4803`, `0x3000`, `0xE0000000`) mark matrix rows and auxiliary metadata.

Recommended workflow:
- Run `tools/dump_component_blocks.py new_style_mcd_files/<file>.mcd --json notes/component_dumps/<file>_new_style_components.json` to capture both the sub-block summary and the raw placement bytes (already done for TW84, TW85, YIKES).
- Normalize the placement values (word ÷ 2^32) and correlate them with the Monu-CAD DXF INSERT data (e.g., CRS2 sits at `(0, 12.9375)` with scale 1.0). This exposes the affine transform matrix encoded in the trailer.
- Feed that matrix back into `mcd_to_dxf`: parse each `0x4803` block, apply the placement transform, and emit the resulting geometry before writing the DXF. Once this works for the new-style files we can adapt the same logic to the legacy dialect (which retains the TLV strokes but lost the placement trailers).

Latest poking shows every trailer begins with a tag=0x0000,size=192 TLV that stores 48 little-endian 16.16 slots. Grouping them 4-at-a-time looks like a 3x4 affine matrix (basis vectors plus translation). The scalar TLVs (tag=0xE000,*) still mirror the normalized fractions we saw earlier, but the INSERT payload we care about lives in the glyph table described below.

The second large block (~8 KiB in the CRS2 capture) is a fixed-length table of glyph placements. Each record contains:

1. Five little-endian float64 values ordered as (10, 20, 50, 41, 42) - insert X, insert Y, rotation, scale X, scale Y exactly as DXF stores them.
2. Twelve bytes of padding (currently zero).
3. A length-prefixed ASCII glyph label (FSKMY, SKMPERID, ...).
4. Five little-endian 32-bit integers: the MAIN component id (0x4207 so far), two flag slots (both 0x00000100 in the CRS2 trailer), an unknown counter, and a data-offset back into the 0x4803 block.

`tools/extract_placement_debug.py` decodes both the 3×4 matrices and these per-glyph records into `notes/placement_matrix_snapshots.json` so new agents can diff the doubles directly against the reference DXF INSERTs without re-parsing the binary.

With the placement pipeline in place we’ll finally match Monu-CAD’s block inserts without exporting from the CAD app.

### 14.11 Vm chunk matcher backlog
1. **Tighten the candidate scoring.** The first matcher pass (`match_vm_chunk_tail_candidates.py`) keeps the five closest offsets per chunk segment but the squared distances are still large. Cluster by metadata quartet, bucket the recurring spans, and record the confidence so we know which quartets already produce stable ranges.
2. **Promote spans to a lookup table.** Once a quartet → span mapping survives the clustering, bake it into a JSON/py module and teach FontManager/generate_vm_font_assets.py to pull the proper tail subset without falling back to the DXF export.
3. **Add regression hooks.** Capture a gold sample of the candidate JSON and add a smoke check (e.g., ensure each quartet still resolves to the same offset span width). This keeps future matcher tweaks from scrambling the metadata IDs before the renderer consumes them.

---

## 15. PNG preview renderer (`render_component_png.py`)

`render_component_png.py` is a headless rasterizer that feeds the geometry parsed by `mcd_to_dxf` into Pillow and emits square PNGs. It works for both `.mcd` drawings and standalone `.mcc` components because it reuses the same `brute_force_deflate()` + `parse_entities()` pipeline (component circle injection included). Usage:

```powershell
python render_component_png.py FACE.mcc `
    --preview FACE_thumb.png --preview-size 256 `
    --hires FACE_fullres.png --hires-size 4096
```

- `--preview` / `--hires` pick any combination; at least one is required. `--hires-size` can be pushed to **4096+** if you need more detail—the script scales the line weight automatically so double lines stay distinct instead of bleeding together.
- Sizes represent the side length in pixels. The renderer samples every arc into polylines, trims obvious outliers (helper centers that sit `1e5` units away), keeps aspect ratio, and adds ~5 % padding before scaling the drawing to the square canvas.
- Backgrounds are fully transparent and the DPI metadata defaults to 72 (Photoshop’s dialog). We can inject a different `dpi` tag later if print workflows demand it, but the real fidelity comes from the raw pixel count.
- Works equally well for `.mcd` files (e.g., `117149301-SPICER.mcd`) so long as `parse_entities()` can describe the scene with lines/arcs/circles. Nested components/fonts still appear exploded only when the parser already knows how to interpret them—see Section 14 for that roadmap.

This closes the loop for simple component cataloguing: convert `.mcc` → DXF (if needed) or straight to PNG for UI previews / QA without round-tripping through Monu-CAD. The renderer intentionally shares all geometry heuristics with `mcd_to_dxf`, so every parser improvement automatically flows into the PNG pipeline.

### 14.12 Automation + regression harness (2025‑03‑02)
- **Component/name cataloguing** – `tools/catalog_dxf_inserts.py` now walks the BLOCK/INSERT sections in any DXF. Running it against the only references we have for TW84/TW87/WI36/WI37 produced `notes/component_catalogs/fully_exploded_insert_catalog.txt` (all empty) plus a contrasting TW85 partial dump. This cements T3: the fully-exploded exports really do wipe every glyph reference, so the only surviving vocab is whatever we can recover from TLV/component payloads.
- **Duplicate hygiene telemetry** – `mcd_to_dxf.py` accepts `--duplicate-log <path>`, logging every offset/layer/etype pair filtered by the parser. This replaces the brittle “duplicate record name” failure path with Pack Data-style warnings and a file we can diff between runs.
- **Glyph label heuristics** – `_iter_text_entities()` now inspects TLV glyph placement records and decodes labels such as `FSKMY`/`MAINO` back into characters using the dtafile prefixes. Metrics are sanity-checked (finite, ≤1e5 units, scale ≥1e‑3) so partially exploded garbage is ignored instead of poisoning the font cache.
- **Font fallbacks** – if a `.fnt` archive refuses to parse, `FontManager` can build glyphs straight from the sliced component bins (`FONTS/components_mcalf010/…`). MAIN/VERMARCO continue to prefer the curated JSON, but every other font now has a last-resort extractor that keeps text rendering alive even when the drawing embedded a partially exploded font.
- **Regression harness** – `tools/run_regression_suite.py` regenerates DXFs for the legacy samples and instantly diffs them against the preserved Monu-CAD exports (reports land under `analysis/regressions/`). Pass `--sample <name>` to focus on one drawing or `--skip-convert` to reuse cached DXFs when tuning the parser.
- **New-style control notes** – `notes/new_style_control_summary.md` captures how the re-saved v9 payloads still miss the same INSERT-heavy geometry, proving the gaps stem from our parser rather than the ancient save format. Pair this with the regression suite to determine whether a fix benefits both dialects.

### 15.1 Server wrappers (`mcc_server_rendering/`)

To keep the existing PHP tooling untouched while we phase in `.mcc` previews, we added a tiny server-facing package:

- `mcc_server_rendering/mcc_to_png_preview.py` – CLI wrapper that calls the logic above and emits a square PNG at whatever size the caller requests (default `400`). It mirrors the old MTN renderer’s interface, prints errors to `stderr`, and returns Unix-style exit codes so the PHP bridge can convert the PNG to base64 just like before.
- `mcc_server_rendering/mcc_to_dxf.py` – CLI wrapper that uses `mcd_to_dxf`’s `brute_force_deflate`/`parse_entities`/`write_dxf` pipeline and spits out a standards-compliant DXF for in-browser viewers.

Both wrappers simply extend `sys.path` to import the real modules (`render_component_png.py`, `mcd_to_dxf.py`, plus helpers such as `component_parser.py` and `tlv_dump.py`). Every parser fix therefore flows through to the server tools automatically.

> **Deployment reminder:** the server cannot see the workstation’s checkout. Any time we touch `mcd_to_dxf.py`, `render_component_png.py`, their helper modules, or the reference DXF/font assets (`FONTS/Mcalf092...`, `FONTS/Mcalf020...`), we must manually re-upload the updated files next to the wrappers before the new behavior is visible online.

### 15.2 Renderer backlog
1. **Consume glyph JSON directly.** As soon as the Vm metadata → tail mapping lands, teach `render_component_png.py` to load the `.json` glyph tables so Vm/MAIN text renders without exploding Monu-CAD into DXF first.
2. **Text/TLV smoke tests.** Add fixtures that rasterize the text-heavy samples (`TEST_WITH_FONT`, `TEST_VERMARCO`) and diff the PNGs/line counts so we notice immediately when glyph spacing or kerning regresses.
3. **Batch ergonomics.** Expose a `--batch` flag that walks a directory of `.mcd/.mcc` files, stamps both preview + hi-res PNGs, and writes a manifest (label, bbox, source) so QA can spot-check the renders without digging through logs.

### 16.7 Font archive previews (`fnt_to_dxf.py`)

- `tools/fnt_to_dxf.py` renders standalone `.fnt/.fnt.decompressed` archives into DXF by driving the same `FontDefinition.render()` path that `.mcd` text uses. Glyph ordering comes from `_derive_font_mapping()` (with built-in maps for MAIN/VERMARCO), and the command line exposes nominal text height/tracking.
- When the `FONTS/` tree is present we hand `FontManager` that directory so MAIN and VERMARCO `.fnt` exports reuse the real outline glyphs captured in `VERMARCO_glyphs.json` / `Mcalf020_exported_from_monucad.dxf`. Unknown fonts still fall back to the raw centerline skeletons because we have not decoded their outline-expansion TLVs yet, so treat those DXFs as layout previews.
- Pass `--disable-font-manager` to force the raw `.fnt` decoding path even if `mcfonts.lst` is available. This is handy when experimenting with the Vermarco stroke-expansion TLVs because it ensures we exercise the new chunk/tail loaders instead of the cached outline JSON. The script currently reverts to the centerline glyph whenever the recovered outline list is shorter than the original skeleton, so improved metadata mapping is still on the roadmap.
- In the raw path we now read the component tails from `components_Mcalf020/*.bin`, look them up inside `VM_tail_records.json`, and rebuild the outline segments directly from the `.fnt` payload. Vermarco therefore renders with the proper outlines even without the FontManager shortcut, while other fonts continue to export their centerline skeletons until we decode their helper TLVs.

## 16. Legacy proto `.dwg` containers (`old_monucad_format_dwg/`)

The `old_monucad_format_dwg` folder holds pre-v9 Monu-CAD exports. Despite their `.DWG` extension they are not AutoCAD files—they are compact binary containers that predate the modern `.mcd` layout. We can reuse our deflate scanners, but the surrounding structure is different.

### 16.1 Fixed 0x30-byte header

Every sample begins with the same 48-byte header block:

| Offset | Size | Example (`#19857RG.DWG`) | Notes |
|--------|------|-------------------------|-------|
| `0x00` | u16  | `0x0000`                | Reserved (always zero). |
| `0x02` | u16  | `0x0014` or `0x001E`    | Format/version code. |
| `0x04` | u32  | `0x007B0F27`            | Packed timestamp (varies per file; matches file dates). |
| `0x08` | u32  | `0x00010003`            | Magic constant. |
| `0x0C` | u32  | `0x00000000`            | Reserved. |
| `0x10` | f32  | `3.410954`              | Layout height in inches. |
| `0x14` | u8 + ASCII | `0x05 + "KC89A"` | Length-prefixed component label (up to five chars). |
| `0x1A` | u16  | `0x0000`                | Padding. |
| `0x1C` | u32  | `0x00200000`            | Constant `2.0` stored as 16.16 (purpose TBD). |
| `0x20` | u32  | `0x00000000`            | Reserved. |
| `0x24` | u32  | `0x00000000`            | Reserved. |
| `0x28` | u32  | `0x00020000`, `0x000B0000`, … | 16.16 fixed-point span (2.0″, 11.0″, 2818.0″, ... depending on the job). |
| `0x2C` | u32  | `0x00000D00`, `0x00000F00`, … | Second 16.16 slot (~0.05–0.06″). |

Offsets `0x18–0x1B` carry a 32-bit component id that lines up with the ASCII label (`0x00004139` for KC89A, `0x00005446` for TF). Together they tell us which glyph archive the proto DWG expects before we even touch the compressed payloads.

### 16.2 Embedded deflate blobs

Running `collect_deflate_streams()` on the raw bytes reveals anywhere from two to ~200 raw deflate segments per file:

| File | Deflate chunks | Total decompressed bytes |
|------|----------------|--------------------------|
| `#19857RG.DWG` | 19 | 2.6 KB |
| `#19927RG.DWG` | 85 | 14 KB |
| `#23305.DWG`   | 192 | 29 KB |

Most payloads decompress into single-byte fills (0x51, 0x75, 0xEB, …), so these blobs look more like palette/lookup tables than geometry. They behave just like the hidden streams inside `.mcd`, so `collect_deflate_streams()` is the right tool when we start building a dedicated extractor.

### 16.3 Trailer metadata records

The final ~0x80 bytes contain *pairs* of 0x20-byte blocks. The first block holds the numeric descriptor, the second block starts with zeroes and then embeds a `[len][ASCII]` label:

```
20 00 11 0F 00 00 05 00 01 00 C5 E8 17 41 CF BC B1 40 ...
00 00 00 00 00 00 04 48 4E 44 32 00 00 00 00 00 00 ...
```

- `size = 0x20`
- `flags = 0x0F11` or `0x0F16`
- `count = 5`, `chunk_id = 1`
- Two floats that match the layout width/height
- Two 16-bit fields (purpose TBD)
- Two more floats that usually read as `1.0`
- Label block stores `[len][ASCII]` at offsets 6/7 (e.g., `04 'H''N''D''2'`)

Every proto DWG ends with at least two of these pairs: one for the primary glyph (`KC89A`, `M335`, `FSKCK`, `OEAPOST`, `MAIN5`, …) and one for `SL`. They give us a clean mapping back to the MAIN/Vm component archives without touching the deflate payloads.

### 16.4 Geometry command stream (WIP)

Between the header and trailer sits a dense 16-bit command stream. We currently see hundreds of thousands of records per file. The low opcodes (`0x0000`…`0x000F`) carry the interesting bits:

- `0x0002` – Usually followed by two IEEE-754 floats (x, y). Some params (e.g., `0x0D00`) act as control markers and are not followed by floats.
- `0x0001` – Same payload layout as opcode 2 (two floats). Appears to denote a different pen mode.
- `0x0003` – Also followed by two floats. Frequently occurs immediately after opcode 0x0000 control blocks.
- `0x0000`, `0x000B`, `0x000F` – Control opcodes (no float payload) that seem to delineate strokes or reference trailer entries.

Heuristic: treat any opcode 1/2/3 whose next two 32-bit words look like floats (exponent between ~80 and 150) as a coordinate record. This already recovers 700–3000 points per file.

### 16.5 Prototype parser / segment dump

`tools/proto_dwg_inspector.py` wires the observations above into a practical workflow:

- Summaries for every DWG live under `notes/proto_dwg_notes/*.json` (header fields, opcode histograms, trailer labels, deflate offsets).
- Passing `--points` writes `{x, y, opcode, param}` tuples extracted with the float heuristic.
- Passing `--segments` connects successive points into naive line segments so we can preview the outline without opening Monu-CAD.

The deflate chunks still look like palette data (no trailer labels appear inside), but the opcode/point dump already produces recognizable geometry for KC89A/M335/FSKCK/MAIN*, so once we decode the remaining control opcodes we can drop these proto DWGs straight into the DXF pipeline.
