# Repo Status and GitHub Push Plan

## Current state
- Git repo initialized on branch `main`.
- Commits:
  1. **Initial snapshot before modularization** – full tree (all tools/assets included).
  2. **Extract deflate helpers and add format guide** – adds `monucad/deflate_io.py`, `docs/format_guide.md`, and rewires `mcd_to_dxf.py` to import deflate helpers.
- Remote configured: `origin` -> `git@github.com:stevenrwh/MCD_decode.git`.

## Push attempts
- `git push -u origin main` timed out twice (120s and 300s). The repo contains many large binaries (e.g., `tools/radare2-6.0.4-w64`, `tools/upx-*`, `tools/strings`, numerous font archives, sample DXFs/PNGs), so the initial push is very heavy.

## Recommended cleanup before retrying
- Prune bulky tool bundles and other non-source artifacts from history, then push:
  - Candidates to drop: `tools/radare2-6.0.4-w64/`, `tools/upx-4.2.2-win64*`, `tools/strings/`, `tools/*zip`, large reference binaries that aren’t needed to build/parse (keep small sample .mcd/.mcc/.fnt/.dxf as needed).
  - Use `git filter-repo` (preferred) to remove paths from all commits, then force-push:
    ```bash
    git filter-repo --invert-paths --path tools/radare2-6.0.4-w64 --path tools/upx-4.2.2-win64 --path tools/strings
    git remote set-url origin git@github.com:stevenrwh/MCD_decode.git
    git push -f -u origin main
    ```
  - If `git filter-repo` is unavailable, fall back to `git filter-branch` or create a fresh repo containing only the curated subset (source, scripts, small samples, docs).
- If retaining large assets is required, consider Git LFS for the big binaries/zips before pushing.

## Next coding steps (modularization)
- Continue moving logic out of `mcd_to_dxf.py` into `monucad/`:
  - `component.py` (component defs/sub-blocks), `catalog.py` (label/index tables), `placement.py` (trailers/block tables), `geometry.py` (entity helpers), `converter.py` (orchestrator).
  - Keep `mcd_to_dxf.py` as a thin CLI that wires the modules together.
- Expand `docs/format_guide.md` with concrete counts/offsets as placement tables are decoded (old/new MCD, MCC, FNT).
