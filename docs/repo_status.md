# Repo Status and GitHub Push Plan

## Current State ✅
- **GitHub repo live**: https://github.com/stevenrwh/MCD_decode
- Git repo on branch `main`, pushed successfully on Dec 19, 2025.
- Remote: `origin` -> `https://github.com/stevenrwh/MCD_decode.git`

## Cleanup Completed (Dec 19, 2025)
Successfully reduced repo from **439 MB → 66 MB** using `git filter-repo`.

### Files Removed from History
| Size | Path | Reason |
|------|------|--------|
| ~680 MB | `MONU-CAD_Win95_copy/` | Entire folder now gitignored (installation copy) |
| 52 MB | `Monucad9_Operations_Logfile.CSV` | Large log file |
| ~70 MB | `FONTS/*.TTF` | System fonts (ARIALUNI, BATANG, SIMSUN, etc.) |
| ~15 MB | `tools/` | Radare2, UPX, strings binaries |
| 13 MB | `analysis/MCPro9_unpacked.exe` | Unpacked executable |
| 7 MB | `Monucad9_opening_10x10_try2_rect.CSV` | Large log |
| ~4 MB | `vm_chunk*.json` | Regenerable JSON dumps |

### Current .gitignore
```gitignore
__pycache__/, *.pyc, *.pyo, *.swp, .DS_Store, .venv/
MONU-CAD_Win95_copy/    # Full installation folder
tools/                   # Binary tools
analysis/*.exe           # Unpacked executables
*.dmp                    # Crash dumps
Monucad9_*.CSV           # Large logs
FONTS/*.TTF, *.ttf       # System fonts
vm_chunk*.json           # Regenerable JSON
tmp_*, *.tmp             # Temporary files
```

## Repository Contents
- **Source code**: `mcd_to_dxf.py`, `monucad/` package, `*.py` scripts
- **Documentation**: `KNOWLEDGE.md`, `docs/`, `notes/`
- **Sample files**: Small `.mcd`, `.mcc`, `.dxf`, `.fnt` test files
- **Diagnostics**: `diagnostics/` folder with analysis scripts

## Next Steps (Modularization)
- Continue moving logic out of `mcd_to_dxf.py` into `monucad/`:
  - `component.py`, `catalog.py`, `placement.py`, `geometry.py`, `converter.py`
- Keep `mcd_to_dxf.py` as a thin CLI that wires the modules together.
- Expand `docs/format_guide.md` with concrete counts/offsets.
