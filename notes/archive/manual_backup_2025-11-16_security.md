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
   This section is terminated by `\r
` pairs followed by two null bytes.

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

### 3.3 Section Locator Header

Modern `.mcd` payloads ship with a fixed-size header *before* the INI text. Thanks to the `mcd_section_parser.py` updates we can now dump it directly from each sample:

- `config_offset = payload.find(b"PolishTile = ")` (typically `0x112`) and the table occupies the preceding `config_offset - 0x12` bytes. Two big-endian copies of the INI length live at `config_offset-2` and `config_offset-6`.
- Immediately before the text sit three little-endian integers: `section_count` (always 9 so far), and two reserved slots (`field1`, `field2`, both `1` in our specimens).
- The table itself looks like **seven arrays of nine 32-bit values** (`7 * 9 * 4 = 0xFC` bytes) followed by a 4-byte tail (`14404E7A` in `117149301-SPICER.mcd`). We have not labelled the arrays yet, but they are consistent per drawing and almost certainly encode section IDs, offsets, lengths, and CRCs referenced by the “Section-locator records / CRC does not match” error strings inside MCPro9.

Running `python mcd_section_parser.py 117149301-SPICER.mcd --json analysis\117149301-SPICER_summary.json` yields:

```
[header] sections=9 field1=1 field2=1 config_len=141 config_off=0x112 arrays=7 table_bytes=0x100 remainder=4
"section_header": {
  "arrays": [
    [0xC9B10033, 0x4C48CB69, ...],   # Array 0
    [0x5822B9DF, 0xEBB3B853, ...],   # Array 1
    ...
    [0x9FFF8AF1, 0x8FEF1430, ...]    # Array 6
  ],
  "table_remainder": "14404E7A"
}
```

This gives us raw numbers we can now line up with actual chunk boundaries while we chase down the CRC routine inside MCPro9.

### 3.4 File Security Flags

Monu-CAD’s “File Security” dialog (Masters Only vs. Satellites, etc.) leaves a footprint in the section header. Comparing `unlocked.mcd` and `locked.mcd` (identical geometry/config text) shows:

- The INI text, geometry records, and component payloads are byte-for-byte identical after the header.
- **Only the section-locator table (first `0x100` bytes) changes**: all seven per-section arrays take on completely different 32-bit values when the file is saved as “locked”.

Example (`analysis/unlocked_summary.json` vs. `analysis/locked_summary.json`):

```
unlocked sections[0] = [0x56601d6e, 0x27196123, 0xee05d0c1, 0xd6f04081, 0x1de52eff, 0xe3b11a2a, 0x64661c3f]
locked   sections[0] = [0xb5df4356, 0x32ac3022, 0x9aa33c25, 0x7f52ee48, 0xf5efab41, 0xdfb4df00, 0xb4280028]
```

The dramatic change (with everything else constant) strongly suggests these arrays embed the security flags-possibly as encrypted/hashed keys rather than plain booleans. This narrows our search inside MCPro9: whatever routine populates the seven arrays must read the dialog setting, derive new values, and rewrite the header before deflating. When we trace those functions (next section) we should see the satellite/master options flow directly into the section table.

- The updated `mcd_section_parser.py` exposes that 0x100-byte block directly: the first 63 DWORDs are treated as the per-section arrays, the four bytes that follow are preserved as the tail, and anything before the INI/config marker is logged as metadata. The new header summary (running `python mcd_section_parser.py locked.mcd --json analysis/locked_summary.json`) can be diffed against the unlocked JSON to prove that only those slots change when the security dialog is toggled. Because the parser now serializes the same block back into bytes, we can reapply the salted values produced by `0x0051C1BA` once we understand how those values are derived.
- The helper also autodetects `.mcd`, `.mcc`, and `.fnt` payloads (`--type` overrides the mode, `--glyph-limit` controls font previews), so the same command line now describes drawings, component packs, and fonts without bringing MCPro9 into the loop. Component definitions / placement trailers are listed whenever they exist, and font archives additionally print glyph metrics so we can see how the table changes across different save states.
- When it is time to push edits back into the fake `MCD2` container, `monucad_pack.py build --section-header analysis/locked_header.json` (or any JSON that captured `section_header`) now rewrites the initial 0x100 bytes/tail before emitting the new `.mcd`, so we can replay the salted values once they are known.
- For live instrumentation, `tools/frida_monucad_header.js` + `tools/capture_section_header.py` spawn MCPro9 under Frida, hook the writer at `0x0051C1BA`, and log every `(row, column, value)` tuple when the save flow runs. The capture script currently opens a target `.mcd`, dismisses the overwrite confirmation dialog, and waits for hook events; once the "File Security" menu automation is wired up we’ll finally have both locked/unlocked tables captured straight from MCPro9 and can feed them back through the SectionHeader serializer.

- placeholder
#### Dialog instrumentation (IDs 2030/2031)

- The static `AFX_DIALOGINFO` table that drives all of the file-security UI lives at **VA `0x011225B0` (RVA `0x0D225B0`, file offset `0x00C4A5B0`)**. Entries are `struct { WORD dlg_id; WORD pad; DWORD info_ptr; }`. The block begins with IDs `0x7ED`..`0x7FC`; the two we care about are `0x7EE` (File Security Options) and `0x7EF` (Satellite selector). Their `info_ptr` values are `0x800028D8` and `0x80002900` respectively (high bit is always set on these MFC dialog descriptors).
- Two tiny wrappers call `CDialog::DoModal`/`DialogBoxParam` for those resources:
  - `0x004A4360` (`this` in `ECX`) pushes `0x7EE` and ends with `vtable = 0x00E18BD0`. That class hosts the main File Security radio buttons.
  - `0x00588990` (`vtable = 0x00E2C3C8`) pushes `0x7EF` and sets up the "Specific Satellite" child dialog.
- **Call sites.** Each wrapper is referenced seven times (per the relocation table). The paired call instructions sit at `0x004555B7/0x00455620`, `0x00455AE5/0x00455B47`, `0x0048BB9C/0x0048BD0D`, `0x0048CCBF/0x0048CE01`, `0x0051C5DC/0x0051C62C`, `0x00576F0B/0x005770A2`, and `0x005A041F/0x005A049F`. These call sites line up with the various save flows (drawing save command, component save, etc.), so setting a breakpoint on any of the `call 0x4A4360` / `call 0x588990` pairs will trigger whenever MCPro9 needs to prompt for security.
- **OnOK handlers.**
  - Dialog 2031 (satellite picker) lands in `0x00588B60`. That handler calls `[esi+0x68]->DoDataExchange` to pull the radio / list-box state, then funnels the result through `0xC405E6`, `0xC40C6A`, and finally `[0xE06864]` to stash the chosen satellite IDs into the caller’s data model. Instrumenting this routine lets you confirm which structures hold the "Master/Satellite/Specific" flags before any CRC math runs.
  - Dialog 2030’s message handlers are clustered right after the constructor: `0x004A4400` toggles the UI state and pushes the choice to `0x468EF0`/`0x4689A0`, while `0x004A44A0` copies the decision back into the owning `CDrawingSaveCommand`. Break here if you need to log the radio buttons earlier than the save command does.
- **Hand-off to the header writer.** Once `DialogBoxParam` returns (the `call 0xC4070C` at `0x00455634`), the save routine grabs the dialog context from `[esp+0x13C]`, normalizes it via `0xC404EA/0x4D9B50`, and then jumps into `0x0051C090`. That large helper iterates over the seven section arrays (`edi` is the save command, `[esp+0x20]` points at the per-section table) and repeatedly calls `0x519A20`/`0xC409C4` before writing out each 32-bit slot via `[0xE06694]`. This is the spot to watch for writes to the first `0x100` bytes of the deflated buffer:
  - Break `0x0051C090` and log its first parameter (`ECX`/`EDI`) so you can identify the owning `CDrawingSaveCommand`.
  - Set a data breakpoint on `deflate_buffer+0x0` once you know which member inside that object points at the payload (look at the pointer passed into `0xC404EA` just before each write).
  - If scripting, instrument the `mov ecx, dword ptr [esp + 0x18] / mov edx, dword ptr [esp + 0x10] / call [0xE06694]` sequence inside `0x0051C190`—that pair of pointers is exactly what feeds the seven-by-nine table.

With those hook points you can watch the Master/Satellite dialog feed straight into the section-header writer, grab the intermediate values, and eventually mirror the 32-bit transforms (likely salted CRCs) in our Python tooling.

#### Header writer anatomy (`0x0051C090`)

- `CDrawingSaveCommand` stores the section-descriptor array at `this+0x108` (pointer) and the section count at `this+0x10C`. Function `0x519A20` walks that table: it dereferences each descriptor, calls the vtable slot at `+0x18` to materialize a 32-bit token, and immediately writes that value using the function pointer at `[0xE06694]` (resolves to code at `0x00BF7784`, effectively the writer for the section-locator buffer).
- After the file-security dialog returns, the save pipeline calls `0x51C090`. The prologue builds several temporary CArray/COleStream objects (`locals @ [esp+0x10],[0x18],[0x20],[0x24]`). Then:
  1. `0x519A20` is invoked once to get the active section descriptor (`esi` in this routine).
  2. The outer loop (`ebp` stepping by 4, `edi` running from 1..section_count) walks each of the seven arrays. For every row it retrieves the corresponding pointer by calling `0xC404EA` with `([esp+0x20] + ebp)`, then prepares the per-column scratch buffers through `0xC40790`.
  3. The inner loop (`esi` is the column index) repeatedly calls `0xC404EA` to grab the destination slot, then hits the crucial block at `0x0051C1BA`:
     ```
     mov ecx, [esp+0x18]   ; source value (already salted with the dialog choice)
     mov edx, [esp+0x10]   ; destination pointer inside the 0x100-byte header
     push ecx
     push edx
     call [0xE06694]       ; writes 4 bytes, returns non-zero on success
     ```
     Hook here (or patch the call target at `0x00BF7784`) to dump each `(row, column, value)` tuple as it’s emitted.
  4. On failure the code calls `0xC4117A` with `(1, column_index)` which surfaces “Section-locator” warnings—handy for verifying you’ve latched the right bytes.
- After all rows are filled, the routine calls `0x51CFA0` to stream the freshly-built section table (plus the trailing `14404E7A` marker) into the deflation buffer. If you need to observe the final `0x100` block in memory, break `0x51CFA0` after the call to `0x51E060`—the buffer pointer lives in `[esp+0x11C0]` at that point.

**Instrumentation recipe**
1. Breakpoint `0x0051C190`. Let it hit once and dump:
   - `EDI` → `CDrawingSaveCommand*`
   - `[EDI+0x108]` (array base), `[EDI+0x10C]` (count)
   - `[esp+0x10]` (destination slot) and `[esp+0x18]` (computed value) every time the call executes.
2. Optionally patch `[0xE06694]` with a logging stub (or set a breakpoint on `0x00BF7784`). The first argument (`EDX`) is the destination pointer inside the deflated payload; the second (`ECX`) points at the 4-byte value you want to mirror in Python.
3. Cross-check the dumps against `analysis/unlocked_summary.json` and `locked_summary.json`. The only difference between dialog choices should be the word values sent through this loop.

Once we clone the computation that feeds `[esp+0x18]` we can reproduce the same 7×9 tables in `mcd_section_parser.py` / `monucad_pack.py` and flip the security flag programmatically.

##### Row/column bookkeeping at `0x0051C1BA`

- The outer loop stores the current row offset in `EBP`. Because it increments by four each time, you can derive `row_index = EBP >> 2` directly in your debugger script.
- Immediately before entering the inner loop, `0xC404EA`/`0xC40790` copy the row container into the scratch slot at `[ESP+0x10]`. Treat that pointer as the base of the 9-entry array for the active row—`dest_ptr = [ESP+0x10]`.
- Each pass through the inner loop calls `0xC404EA` again with `([ESP+0x20] + column*4)` and stores the scalar at `[ESP+0x18]`. By the time execution reaches `0x0051C1BA`, the registers contain:
  - `ECX = *(DWORD*)[ESP+0x18]` → 32-bit value about to be emitted
  - `EDX = [ESP+0x10]` → pointer to the row buffer (`section_table[row_index]`)
  - `ESI` → column counter (starts at 0 and increments after every call)
- To log the exact cell that is being filled, compute `column_index = (poi([esp+0x10])_cursor - dest_ptr) / 4`. In x32dbg you can express this inside the breakpoint action:
  ```
  bp 0x51C1BA
  printf "row=%d col=%d value=%08X dest=%p
", @ebp>>2, (poi(@esp+0x10+4)-poi(@esp+0x10))/4, poi(@esp+0x18), poi(@esp+0x10)
  ```
  (Adjust the expression to match your debugger’s syntax; the key is that `[esp+0x10]` always points at the current row’s base.)
- `[0xE06694]` currently points at `0x00BF7784`, a `SectionLocatorWriter::Push` stub that simply appends the value into the row container and returns success. Patching or breakpointing that address gives you a single hook for every 32-bit slot without retooling the caller.

Collecting the `(row_index, column_index, value)` triples for both dialog states gives you the raw material needed to reverse the salt/CRC routine per section.

##### Where the descriptor vector comes from

- Constructors around `0x00500650` (`CDrawingSaveCommand::Init?`) fetch the descriptor array via `0x0046BA20` and immediately stash it at `this+0x108`, with the count stored at `this+0x10C`. That getter is just `mov eax, [ecx+0x7C0]; ret`, so the real storage sits at offset `0x7C0` inside the save-command object (very likely an embedded `CArray`).
- Companion setters (`0x0046BA30`, `0x0046BA60`, …) write doubles at `this+0x7C8`, `this+0x7D0`, etc. This block looks like a struct of 64-bit fields plus the descriptor pointer at `0x7C0`, which explains why so many save-related classes copy it verbatim (`mov [esi+0x108], eax` shows up in dozens of constructors).
- Routine `0x0051B430` walks the same descriptor vector and pushes each descriptor's contribution into a caller-supplied `CArray` via `CArray::Append` (`0xC406E2`). This is the write path used when the save command exports metadata (e.g., section names) before the security rewrite kicks in.
- Translation: if you can dump `[this+0x7C0]` just before hitting `0x0051C090`, you'll get a raw pointer to the seven descriptor objects. Each one has a vtable with at least the `Serialize()` slot at `+0x18`-those are the per-row calculators feeding `[esp+0x18]`. Static addresses for the vtables still need to be mapped, but now we know exactly where they live in memory once the save command is built.

#### Bezier-to-Arc shim (`CQBezierToArc.dll`)

- The UI command that converts Beziers to arcs loads `MONU-CAD Pro\CQBezierToArc.dll` (three exports: `CQBezierToArc`, `CQBezierToArcM`, `DEL`). The first export simply gathers three caller parameters, pushes two baked-in doubles, then calls `CQBezierToArcM`.
- We can drop a shim in place: rename the original to `CQBezierToArc_real.dll`, deploy our own DLL with the same exports, and have it `LoadLibraryW` the real one. Because Monu-CAD loads the plugin from its working directory, no additional registry tricks are required.
- Shim source + build script live in `tools/bezier_shim/`. `build_shim.bat` compiles `CQBezierToArc_shim.cpp` into `CQBezierToArc.dll`, which logs every invocation to `CQBezierShim.log`, captures a full-memory minidump (`CQBezierDump_*.dmp`) the first time it runs, **and now hooks `kernel32!ReadFile/WriteFile` via the shim’s IAT**. Every `.mcd` read/write gets mirrored into the log (first 0x200 bytes per call), so we can see the exact header bytes flowing through MCPro9 without attaching a debugger. This gives us an in-process Trojan horse we can extend further: dump save-command structures, poke `CDrawingSaveCommand`, or call the section-header writer directly-after all, our code runs inside Monu-CAD without triggering its anti-debugger.

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

### 4.1 MCPro9 Loader Hooks

Static analysis of `analysis/MCPro9_unpacked.exe` gives us reproducible instrumentation points:

- **RuntimeClass map.** Strings such as `CDrawingLoadBaseCommand`, `CDrawingLoadCommand`, `CDrawingSaveCommand`, `CDrawingSaveV7Command`, and `CComponentSaveCommand` sit in contiguous CRuntimeClass entries around `0x00E15D00`. Each entry exposes the factory pointer, so dropping a debugger breakpoint there captures every load/save attempt.
- **Section-locator validation.** The loader reports “Section-locator records”, “ObjectMap section-locator”, “Header section-locator”, and “ObjFreeSpace data does not match section-locator ObjFreeSpace size” when the table disagrees with the actual data (`analysis/MCPro9_unpacked_strings.txt:8523-8547`). Hook these strings to observe the exact struct offsets and CRC enforcement.
- **CRC choke points.** Every section is accompanied by “CRC does not match in %ls” error text, which all routes into `CMCDException`. This makes it trivial to log/fuzz the checksum routine before implementing our own writer.
- **Lettering guards.** Immediately after the section checks, the loader compares lettering metadata (“Number of lettering groups loaded…” @ string 7184). That tells us the INI prelude + lettering tables must agree before any geometry is accepted.
- **Exception funnel.** All fatal parsing errors throw `CMCDException` (“The file is not a valid MONU-CAD drawing / component file”). Hooking the constructor reveals which validation stage failed and with what parameters.

Use these hooks whenever you validate new structural guesses: reproduce the exception, capture the offending offsets, then update this manual so the next agent doesn’t need to rediscover them.

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

`M9_RGB_COMPONENT.mcc` (and every sample under `MONU-CAD_Win95_copy\Mcc`) uses the same fake `MCD2` header + hidden deflate stream, but all payload bytes belong to embedded `CComponentDefinition` blocks rather than `<layer,type>` geometry. Our `component_parser.py` helper formalizes the structure:

| Region | Size / Type | Notes |
|--------|-------------|-------|
| Marker | ASCII `CComponentDefinition` | Search anchor for each block. |
| BBox   | 16 bytes (4 floats) | Component extents, often all zeros. |
| Header ints | 28 bytes (7 × `uint32`) | includes component id, flags, payload sizes. |
| Body   | TLV stream or tagged sub-blocks | Stores the actual geometry/control data. |

### 10.1 TLV mode
- TLV header `<HHHH>` (tag, dtype, count, length) followed by payload bytes.
- `dtype == 0xFFFF` labels mark ASCII class names (`CLine`, `CArc`, `CCircle`); follow-on TLVs contain doubles describing control points.
- `_iter_tlv_fields` + `iter_label_chunks` lets us rip every class/payload without the MCPro9 loader.

### 10.2 Tagged sub-block mode
- Newer files emit inline blocks with tags such as `0x3805` (`ComponentSubBlock.tag`). Each block stores `<tag><dtype><count><declared_size>` followed by a float array.
- Circle blocks encode `(cx, cy, px, py)` quadruplets; `extract_circle_primitives` recovers centers/radii so we can rewrite or render them freely.

### 10.3 Placement trailers
- FACE-style trailers look like `[len][ASCII name][0x075BCD15 magic][instance_id][component_id]`.
- `iter_component_placements` scans the payload to extract every placement (use `component_inspector.py --instances` for JSON output).

### 10.4 Tooling recap
- `component_inspector.py blob.mcc --json summary.json --instances` emits a machine-readable description of every definition and placement trailer.
- `mcd_section_parser.py` (see §12) now lists embedded components alongside the drawing geometry so `.mcd` and `.mcc` analysis stays in sync.

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

## 12. Section / Component Summaries via `mcd_section_parser.py`

To keep interoperability work reproducible we added `mcd_section_parser.py`, a CLI wrapper that ties the hidden deflate payload, geometry parser, and component inspector together:

```
python mcd_section_parser.py MONU-CAD_Win95_copy\Mcd\Demo_DD01.mcd `
    --instances `
    --json analysis\Demo_DD01_summary.json
```

What it does:

- Locates + inflates the hidden deflate stream (or accepts `.decompressed` payloads directly).
- Splits the payload into the textual prelude vs. binary tail (handy when tracing the loader’s INI parsing).
- Reuses `parse_entities` to count/measure line, arc, and circle entities and compute a best-effort bounding box.
- Lists every embedded `CComponentDefinition` block and optional FACE placement trailer using the same helpers as `component_inspector.py`.
- Optionally writes a JSON summary; stash these in `analysis\` so future agents have a canonical description of each sample without opening MCPro9.

This bridges the gap between our manual notes and the binary: every `.mcd` or `.mcc` we care about can now be summarized, versioned, and diffed entirely within the reverse-engineering workspace. When it is time to push edits back into a Monu-CAD-friendly container, run `python monucad_pack.py unpack <file>` to grab the deflate payload, tweak it (geometry, component TLVs, fonts), then `python monucad_pack.py pack edited_payload.decompressed -o new_file.mcd` to emit a fresh file that keeps the fake-gzip header/offsets MCPro expects. For greenfield drawings/components you can now feed a JSON spec directly into `monucad_pack.py build spec.json --payload-out new_payload.decompressed -o new_file.mcd`; the spec supports `config_text/config_lines`, `lines`, and generic `records` so we can synthesize simple `.mcd` payloads without hand-editing hex.

---

## 13. Font Payload Summaries (`fnt_section_parser.py`)

Fonts (`*.fnt`, `*.dta`) use the same fake-gzip wrapper as `.mcd` files, but their payloads consist entirely of TLV-encoded component blocks. The `fnt_section_parser.py` helper mirrors the drawing/component inspector:

```
python fnt_section_parser.py MONU-CAD_Win95_copy\Fonts\Mcalf020.fnt `
    --limit 5 `
    --json analysis\Mcalf020_summary.json
```

This prints the deflate offset, payload size, and a preview of each glyph (label, line count, baseline, advance, bounding box). When `--json` is provided the full glyph metadata (as emitted by `glyph_tlv_parser`) is written to disk so we can diff fonts or feed them into downstream tooling. Combine this with `monucad_pack.py unpack/pack` to round-trip `.fnt` or `.dta` archives exactly like we now do for `.mcd/.mcc`.

## 3.5 Automated Security Sweep

- `tools/mcpro9_security_sweep.py` drives MCPro9 through the canonical "LI" macro, then saves the same geometry once per File Security option (`line_all.mcd`, `line_specific.mcd`, `line_only.mcd`, `line_masters.mcd`). The dialog exposes push-buttons (not radio buttons), so the script explicitly clicks `&All Your Satellites`, `&Specific Satellite`, `&Only You`, or `&Masters Only` before accepting it.
- Always save into MCPro9's default Drawing Save directory to avoid the zero-byte stubs we observed when typing arbitrary absolute paths. After the sweep completes, copy the generated `.mcd` files out of the MonuCAD workspace and run `python mcd_section_parser.py <file> --json analysis/<mode>_header.json` to diff the 7×9 table for each security mode.
- Use the new dialog screenshots under `analysis/security_sweep/` to confirm the UI sequence (Save As → Drawing Save → File Security Options). These captures proved that Frida does not interfere with the UI when we let MCPro9 manage the save path.
