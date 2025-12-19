# MonuCad Reverse Engineering Playbook

This note is meant for whichever agent lands in `C:\Dev\MCD_DECODE`. It is *intentionally generic* so it applies to **any** MonuCad binary or artifact and does **not** rely on RockCAD-specific knowledge beyond the tooling we already staged in the ROA repo.

---

## 1. Environment Snapshot

All reversing should happen from `C:\Dev\MCD_DECODE`. Keep your own notes (diary, case writeups, etc.) under this folder so they live beside the MonuCad assets.

Key neighboring resources:

| Path | Purpose |
|------|---------|
| `C:\Dev\ROA_DECODE\tools\` | Ready-to-use reversing utilities (radare2, Semi VB Decompiler, strings dumps, PowerShell helpers). |
| `C:\Dev\MonuCad_Installer` + `Monucad_Thumbnails_*` | Vendor binaries and sample output worth referencing when validating your RE results. |
| `C:\Dev\PythonMCDReverseEng` | Python experiments for `.mcd/.mcc` parsing; good reference when scripting. |

Copy whatever you need into `MCD_DECODE` so the new agent does not have to keep context-switching into the ROA repo.

---

## 2. Core Toolbelt

| Tool | Where to copy from | Why it matters |
|------|-------------------|----------------|
| **radare2 6.0.4** | `C:\Dev\ROA_DECODE\tools\r2\radare2-6.0.4-w64\` | Lightweight disassembler/debugger that works well on PE32, PE32+, and position-independent binaries. Copy the entire folder (or symlink it) so `.\tools\r2\radare2.exe` is local. |
| **Ghidra** *(optional)* | Install separately (portable ZIP or scoop) | Best for building data-flow graphs, struct definitions, and exportable scripts. Keep the project under `MCD_DECODE\analysis\ghidra_projects`. |
| **x64dbg / x32dbg** | Install from official release | Essential for single-step debugging, API tracing, and patching. Configure the `Symbols` path to point at `%ProgramFiles(x86)%\Windows Kits\10\Debuggers`. |
| **ProcMon / ProcExp** | SysInternals | Watch filesystem/registry/network usage when running MonuCad. Capture logs under `analysis\procmon_runs`. |
| **strings & floss** | Use `Sysinternals\strings.exe` and `FLOSS` from FireEye | Quick way to surface embedded file paths, help text, and function names even when stripped. |
| **Python 3.11+** | Already on the box (`python` in PATH) | For ad-hoc binary parsers. Set up a `venv` inside `MCD_DECODE\tools\pyenv` if you need extra modules (construct, capstone, unicorn, etc.). |
| **Capstone + Keystone** | `pip install capstone keystone-engine` inside your venv | Speeds up scripting disassembly/assembly for patching or instruction matching. |
| **IDA Free / Hex-Rays** *(if licensed)* | Optional | Great for cross-reference heavy work. Drop your IDB files into `analysis\ida`. |

> ✅ **Action:** Create `MCD_DECODE\tools\` and mirror the `radare2`, `strings`, and `Semi-VB` folders from `ROA_DECODE` the first time you set up the workspace. Even though MonuCad isn’t VB, the hex editors and dependency walkthroughs that ship with semi VB are still useful for resource scouting.

---

## 3. Workflow Templates

1. **Triaging a new binary**
   - `sigcheck -nobanner <file>` to capture hashes and compiler timestamps.
   - `diec64.exe <file>` (Detect-It-Easy) if you install it; otherwise rely on PEiD or radare2’s `iI`.
   - `strings -n 6 <file>` and `floss <file>` to dump plaintext (redirect into `analysis/<binary>_strings.txt`).
   - `radare2 -AA <file>` ➜ `pdf @ entry0`, `izz`, `afl` to map functions. Export call graphs via `agf entry0`.

2. **Importing into Ghidra**
   - Use the latest JDK (17+) and set project root at `analysis\ghidra_projects`.
   - Mark imported functions using the `analysis/convert_proc_map.md` template from ROA as inspiration; document start RVAs in a sibling `analysis/<binary>_map.md`.

3. **Dynamic instrumentation**
   - If you suspect encryption/decompression, attach x32dbg, set breakpoints on `VirtualProtect`, `LoadLibraryA/W`, `GetProcAddress`, `DeviceIoControl`, etc.
   - Use ProcMon filters: `Process Name = <binary>` and log to `analysis\procmon_<binary>_<timestamp>.PML`.
   - For self-modifying code, run radare2 in debug mode: `radare2 -d <binary>` and leverage `db sym.imp.KERNEL32.VirtualAlloc`.

4. **Filesystem watchers**
   - Reuse `C:\Dev\ROA_DECODE\tools\check_windows.ps1` to catch pop-up dialogs while you focus on the debugger. Copy it to `MCD_DECODE\tools\check_windows.ps1`.

---

## 4. Suggested Folder Layout

```
MCD_DECODE\
├── analysis\
│   ├── binary_maps\
│   ├── procmon_runs\
│   ├── scripts\
│   └── ghidra_projects\
├── tools\
│   ├── radare2\
│   ├── check_windows.ps1
│   ├── strings\
│   └── python_venv\
├── samples\   (drop interesting .mcd/.mcc/.dll/.exe here)
└── MCD_REVERSE_ENGINEERING_PLAYBOOK.md (this file)
```

Adjust as needed, but keep everything self-contained so another agent can take over instantly.

---

## 5. General Tactics for MonuCad Binaries

MonuCad tools are historically native Win32 applications (Delphi, C/C++, or custom). Assume:

- **Custom file formats** (`.mcd`, `.mcc`, `.mcp`) that merge geometry + metadata. Build small Python readers to dump sections before time-boxing bigger RE.
- **USB dongle / license checks**: Monitor for `HASP`, `Sentinel`, or custom device drivers. Snapshot registry keys and `SetupAPI.dev.log`.
- **DLL side-car helpers**: Search for `LoadLibrary` on vendor DLLs. Keep copies of any `.dll` or `.ocx` modules next to specimens so you can analyze them offline.
- **Compression/encryption**: If you hit a blob that decompresses at runtime, set breakpoints on `RtlDecompressBuffer`, `LZCopy`, `CryptAcquireContext`, etc.
- **Scripting**: When facing repetitive UIs, lean on PowerShell + AutoHotkey to drive them while capturing API traces.

---

## 6. Checklists Before You Start

- [ ] Copy `radare2-6.0.4-w64`, `strings`, and `check_windows.ps1` from `ROA_DECODE\tools\` into `MCD_DECODE\tools\`.
- [ ] Install (or confirm availability of) SysInternals suite, Ghidra, x64dbg.
- [ ] Create `analysis\binary_maps\README.md` explaining naming conventions for future agents.
- [ ] Establish a `venv` if you plan to use Python tooling beyond the standard library.
- [ ] Document *every* binary you touch in an `analysis/<binary>_notes.md` file (hash, compiler info, interesting strings, sections, suspicion).

---

## 7. Useful Commands (copy/paste ready)

```pwsh
# Quick hash + signer info
Get-FileHash .\MonuCad.exe -Algorithm SHA256
sigcheck64.exe -nobanner -q .\MonuCad.exe

# Strings dump
strings.exe -n 6 .\MonuCad.exe > analysis\MonuCad_strings.txt

# Radare2 headless function map
.\tools\radare2\radare2.exe -q -c "aa; afl" .\MonuCad.exe > analysis\MonuCad_afl.txt

# ProcMon capture (run separately, stop when repro complete)
Procmon64.exe /Quiet /Minimized /BackingFile analysis\procmon_MonuCad_$(Get-Date -Format yyyyMMdd_HHmm).pml

# Monitor GUIs
pwsh .\tools\check_windows.ps1 -Watch -Match "Error|Dongle|License"
```

---

## 8. Handoff Expectations

When you are done (or pausing):

1. Summarize new findings in `analysis/<binary>_notes.md` and mention the tools/build numbers used.
2. If you modified or patched binaries, keep originals untouched; place patched copies under `analysis\patched`.
3. Update this playbook whenever you introduce a new tool or workflow so future agents stay aligned.

Good luck, have fun, and keep your disassemblies tidy. 💾

