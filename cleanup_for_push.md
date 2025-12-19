# Git Repository Cleanup for GitHub Push

## ✅ COMPLETED - Dec 19, 2025

**Repository successfully pushed to GitHub**: https://github.com/stevenrwh/MCD_decode

### Results
| Metric | Before | After |
|--------|--------|-------|
| Repo size | 439 MB | 66 MB |
| Push time | Timeout (300s+) | ~13 seconds |

## What Was Removed from History

Used `git filter-repo --invert-paths` to remove:

| Size | Path |
|------|------|
| ~680 MB | `MONU-CAD_Win95_copy/` (entire folder) |
| 52 MB | `Monucad9_Operations_Logfile.CSV` |
| 23 MB | `FONTS/ARIALUNI.TTF` |
| 15 MB | `FONTS/BATANG.TTF` |
| 13 MB | `analysis/MCPro9_unpacked.exe` |
| 11 MB | `tools/radare2-6.0.4-w64/` |
| 10 MB | `FONTS/SIMSUN.TTF`, `MSMINCHO.TTF`, `PMINGLIU.TTF` |
| 8 MB | Large DXF exports |
| 6 MB | `MONU-CAD_Win95_copy/help/*.chm` |
| 3-4 MB | `MONU-CAD_Win95_copy/MONU-CAD Pro/MCPro*.exe` |

## Commands Used

```powershell
# 1. Create backup branch
git branch backup-before-cleanup

# 2. Remove large files from history
git filter-repo --invert-paths `
    --path-glob "MONU-CAD_Win95_copy/MONU-CAD Pro/*.dmp" `
    --path-glob "MONU-CAD_Win95_copy/MONU-CAD Pro/*.exe" `
    --path "MONU-CAD_Win95_copy/help/" `
    --path "tools/" `
    --path "analysis/MCPro9_unpacked.exe" `
    --path-glob "FONTS/*.TTF" `
    --path-glob "FONTS/*.ttf" `
    --path "Monucad9_Operations_Logfile.CSV" `
    --path "Monucad9_opening_10x10_try2_rect.CSV" `
    --path-glob "vm_chunk*.json" `
    --force

# 3. Re-add remote and push
git remote add origin https://github.com/stevenrwh/MCD_decode.git
git push -u origin main

# 4. Remove entire MONU-CAD_Win95_copy from tracking
git rm -r --cached "MONU-CAD_Win95_copy/"
git commit -m "Ignore entire MONU-CAD_Win95_copy folder"
git push
```

## Current .gitignore

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.swp
.DS_Store
.venv/

# Large binary tools (get fresh from releases)
tools/
analysis/*.exe

# Crash dumps and logs
*.dmp
*.CSV
Monucad9_*.CSV

# Large MONU-CAD installation (document how to obtain separately)
MONU-CAD_Win95_copy/MONU-CAD Pro/*.exe
MONU-CAD_Win95_copy/MONU-CAD Pro/*.dmp
MONU-CAD_Win95_copy/help/

# Large fonts (system fonts, not needed)
FONTS/*.TTF
FONTS/*.ttf

# Large JSON dumps
vm_chunk*.json

# Temporary files
tmp_*
*.tmp
```

## After Cleanup

1. Verify repo size: `git count-objects -vH`
2. Expected size: ~30-50 MB (mostly source code, small samples, docs)
3. Push should complete in < 60 seconds

## What to Keep
- All `.py` source files
- `KNOWLEDGE.md`, `format_guide.md`, docs/
- Small sample `.mcd`, `.mcc`, `.dxf` files (< 1MB each)
- `monucad/` package
- `deflate_io.py` and other modules
