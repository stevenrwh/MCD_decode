# New-style Control Summary

We re-ran the DXF diff tool against the re-saved v9 payloads in
`new_style_mcd_files/` to confirm that the geometry gaps we see are not unique
to the 15‑year-old legacy saves. The numbers below come straight from the
`notes/diff_reports/new_style_*_vs_monucad.txt` snapshots.

| Sample | ARC Δ (Codex ‑ MonuCAD) | CIRCLE Δ | LINE Δ |
|--------|-------------------------|----------|--------|
| TW84   | −8,208                  | −2       | −19,954 |
| TW85   | −547                    | +5       | −324    |
| TW87   | −1,260                  | −1       | −2,536  |
| WI35   | +7                      | +1       | −404    |
| WI36   | −361                    | +1       | −2,573  |
| WI37   | −1,698                  | 0        | −6,611  |
| YIKES  | −837                    | −6       | −1      |

Takeaways:

- **Same missing INSERTs** – TW84/TW87/WI37 still shed thousands of arcs vs the
  Monu-CAD exports even when the source `.mcd` is the clean v9 save. That rules
  out “legacy explode” quirks and confirms the parser still needs to honor the
  placement trailers + helper records described in §14.7.
- **Circle helpers survive in control files** – WI35/WI36 show tiny positive
  circle deltas (+1) because the control DXFs still reference component helpers
  that our parser currently treats as standalone CIRCLEs. Once we stamp the
  components properly those deltas should disappear.
- **YIKES mirrors TW84** – even with the un-exploded export we lose ~800 arcs,
  so flattening the drawing (or not) is irrelevant: without INSERT support the
  lettering evaporates in both dialects.
- **Regression harness hook** – `tools/run_regression_suite.py` already covers
  the legacy inputs; point it at the `new_style_mcd_files/*.mcd` set (using the
  same Monu-CAD references above) to keep this control group green as the
  parser evolves.

This fulfills T10 from `notes/old_style_conversion_plan.md`: the remaining
delta is squarely a parser problem, not a quirk of the old save path.
