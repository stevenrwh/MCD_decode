# Git Repository Cleanup for GitHub Push

## Problem
The repository is ~439 MB with large binary files in history, causing push timeouts.

## Largest Files in History
| Size | Path |
|------|------|
| 135 MB | `MONU-CAD_Win95_copy/MONU-CAD Pro/*.dmp` (crash dumps) |
| 52 MB | `Monucad9_Operations_Logfile.CSV` |
| 23 MB | `FONTS/ARIALUNI.TTF` |
| 15 MB | `FONTS/BATANG.TTF` |
| 13 MB | `analysis/MCPro9_unpacked.exe` |
| 11 MB | `tools/radare2-6.0.4-w64/` |
| 10 MB | `FONTS/SIMSUN.TTF`, `MSMINCHO.TTF`, `PMINGLIU.TTF` |
| 8 MB | Large DXF exports |
| 6 MB | `MONU-CAD_Win95_copy/help/*.chm` |
| 3-4 MB | `MONU-CAD_Win95_copy/MONU-CAD Pro/MCPro*.exe` |

## Recommended Cleanup

### Option 1: Remove Large Binaries from History (Recommended)

Run this PowerShell script to remove large/unnecessary files from Git history:

```powershell
cd c:\Dev\MCD_decode

# Create backup branch first
git branch backup-before-cleanup

# Remove large files from history using git filter-repo
git filter-repo --invert-paths `
    --path-glob 'MONU-CAD_Win95_copy/MONU-CAD Pro/*.dmp' `
    --path-glob 'MONU-CAD_Win95_copy/MONU-CAD Pro/*.exe' `
    --path 'MONU-CAD_Win95_copy/help/' `
    --path 'tools/' `
    --path 'analysis/MCPro9_unpacked.exe' `
    --path-glob 'FONTS/*.TTF' `
    --path-glob 'FONTS/*.ttf' `
    --path 'Monucad9_Operations_Logfile.CSV' `
    --path 'Monucad9_opening_10x10_try2_rect.CSV' `
    --path-glob 'vm_chunk*.json' `
    --force

# Re-add remote (filter-repo removes it)
git remote add origin git@github.com:stevenrwh/MCD_decode.git

# Force push to GitHub
git push -u origin main --force
```

### Option 2: Fresh Start (if filter-repo causes issues)

```powershell
# Rename current repo
Rename-Item .git .git_backup

# Create fresh git history
git init
git add .
git commit -m "Initial commit - clean history"
git remote add origin git@github.com:stevenrwh/MCD_decode.git
git branch -M main
git push -u origin main --force
```

## Update .gitignore First

Before cleanup, update `.gitignore` to prevent re-adding large files:

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
