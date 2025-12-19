# Legacy DXF INSERT Catalogs

`tools/catalog_dxf_inserts.py` now prints the BLOCK definitions and INSERT usage
inside any DXF so we can quickly see which glyph/component names survive a given
export pass.  Running it against the only references we have for TW84/TW87/WI36/WI37
shows why their text vanished from our Codex exports:

```
python tools/catalog_dxf_inserts.py \
  old_style_mcd_files/TW84_monucad_export_fully_exploded.dxf \
  old_style_mcd_files/TW87_monucad_export_fully_exploded.dxf \
  old_style_mcd_files/WI36_monucad_export_fully_exploded.dxf \
  old_style_mcd_files/WI37_monucad_export_fully_exploded.dxf
```

All four reports contain only the default `*Model_Space`/`*Paper_Space*` blocks
and **zero INSERTs**.  The fully exploded Monu-CAD exports nuked every block,
so there are no surviving glyph IDs to diff against.  This confirms the data
gap called out in T3 of the conversion plan: these drawings need either a
partial Monu-CAD export (not available) or a control sample from the new-style
save path to recover the text references.

For comparison, running the same script on `TW85_monucad_export_partially_exploded.dxf`
lists dozens of block names (FSKMA/FSKMB/… SKMPERID) with INSERT counts, giving
us exactly the glyph vocabulary that disappeared once Monu-CAD performed the
final explode.  The catalog output lives next to this note under
`fully_exploded_insert_catalog.txt` so the next agent can revisit the numbers
without rerunning the tool.
