param(
    [string]$McdFile,
    [string]$McProExe = "C:\MONU-CAD_Win'95\MONU-CAD Pro\MCPro9.exe",
    [string]$CaptureDir = ".\captured_temp",
    [int]$SuspendDelayMs = 400,
    [switch]$AutoSuspend,
    [switch]$LaunchSuspended
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$CaptureDir = (Resolve-Path -Path (New-Item -ItemType Directory -Path $CaptureDir -Force)).ProviderPath
$tempDir = Join-Path -Path $env:LOCALAPPDATA -ChildPath "Temp"

Write-Host "Capturing Mcd temp files from $tempDir into $CaptureDir" -ForegroundColor Cyan

if ($LaunchSuspended) {
    Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class SuspendedLauncher {
    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct STARTUPINFO {
        public int cb;
        public string lpReserved;
        public string lpDesktop;
        public string lpTitle;
        public int dwX;
        public int dwY;
        public int dwXSize;
        public int dwYSize;
        public int dwXCountChars;
        public int dwYCountChars;
        public int dwFillAttribute;
        public int dwFlags;
        public short wShowWindow;
        public short cbReserved2;
        public IntPtr lpReserved2;
        public IntPtr hStdInput;
        public IntPtr hStdOutput;
        public IntPtr hStdError;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct PROCESS_INFORMATION {
        public IntPtr hProcess;
        public IntPtr hThread;
        public int dwProcessId;
        public int dwThreadId;
    }

    [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    public static extern bool CreateProcess(
        string lpApplicationName,
        string lpCommandLine,
        IntPtr lpProcessAttributes,
        IntPtr lpThreadAttributes,
        bool bInheritHandles,
        uint dwCreationFlags,
        IntPtr lpEnvironment,
        string lpCurrentDirectory,
        ref STARTUPINFO lpStartupInfo,
        out PROCESS_INFORMATION lpProcessInformation);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool CloseHandle(IntPtr hObject);
}
"@
}

function Resume-McdProcess {
    param([int]$ProcessId)
    $process = Get-Process -Id $ProcessId -ErrorAction Stop
    foreach ($thread in $process.Threads) {
        try {
            $thread.Resume()
        } catch {
            # Ignore threads we cannot manipulate (e.g., already resumed)
        }
    }
}

function Suspend-McdProcess {
    param([int]$ProcessId)
    $process = Get-Process -Id $ProcessId -ErrorAction Stop
    foreach ($thread in $process.Threads) {
        try {
            $thread.Suspend()
        } catch {
            # Ignore threads we cannot touch (e.g., already suspended)
        }
    }
}

function Register-McdWatcher {
    param(
        [string]$Filter
    )

    $watcher = New-Object System.IO.FileSystemWatcher $tempDir, $Filter
    $watcher.IncludeSubdirectories = $false
    $watcher.NotifyFilter = [IO.NotifyFilters]'FileName, LastWrite'
    $watcher.EnableRaisingEvents = $true

    Register-ObjectEvent -InputObject $watcher -EventName Created -SourceIdentifier "MCD_$Filter" -Action {
        $src = $Event.SourceEventArgs.FullPath
        $timestamp = (Get-Date).ToString("yyyyMMdd_HHmmssfff")
        $name = [IO.Path]::GetFileNameWithoutExtension($src)
        $ext = [IO.Path]::GetExtension($src)
        $dest = Join-Path -Path $using:CaptureDir -ChildPath ("{0}_{1}{2}" -f $name, $timestamp, $ext)

        for ($i = 0; $i -lt 40; $i++) {
            try {
                Copy-Item -LiteralPath $src -Destination $dest -ErrorAction Stop
                Write-Host "Captured $src -> $dest" -ForegroundColor Green
                break
            } catch {
                Start-Sleep -Milliseconds 50
            }
        }
    }
}

$eventSubs = @()
foreach ($filter in @("Mcd*.tmp", "Mcd*.cfg")) {
    $eventSubs += Register-McdWatcher -Filter $filter
}

if ($PSBoundParameters.ContainsKey("McdFile")) {
    if (-not (Test-Path -LiteralPath $McProExe)) {
        throw "MCPro9 executable not found at '$McProExe'. Edit -McProExe to point to your installation."
    }

    $proc = $null
    $initiallySuspended = $false

    if ($LaunchSuspended) {
        # When lpApplicationName is supplied, lpCommandLine should contain only the arguments.
        # MCPro9 understands being launched with the drawing path as its single argument.
        $cmdLine = '"{0}"' -f $McdFile
        $workingDir = Split-Path -Path $McProExe
        $si = New-Object SuspendedLauncher+STARTUPINFO
        $si.cb = [Runtime.InteropServices.Marshal]::SizeOf([type][SuspendedLauncher+STARTUPINFO])
        $pi = New-Object SuspendedLauncher+PROCESS_INFORMATION
        $CREATE_SUSPENDED = 0x00000004

        $ok = [SuspendedLauncher]::CreateProcess($McProExe, $cmdLine, [IntPtr]::Zero, [IntPtr]::Zero, $false, $CREATE_SUSPENDED, [IntPtr]::Zero, $workingDir, [ref]$si, [ref]$pi)
        if (-not $ok) {
            throw "CreateProcess failed with Win32 error $([Runtime.InteropServices.Marshal]::GetLastWin32Error())."
        }

        $proc = Get-Process -Id $pi.dwProcessId
        [SuspendedLauncher]::CloseHandle($pi.hThread) | Out-Null
        [SuspendedLauncher]::CloseHandle($pi.hProcess) | Out-Null

        $initiallySuspended = $true
        Write-Host "MCPro9 created in a suspended state (PID $($proc.Id))." -ForegroundColor Yellow
    } else {
        $proc = Start-Process -FilePath $McProExe -ArgumentList "`"$McdFile`"" -PassThru
        Write-Host "Started MCPro9 (PID $($proc.Id)) for '$McdFile'."
    }

    if ($AutoSuspend) {
        if ($initiallySuspended) {
            Write-Host "Process is already suspended; launch resume whenever you are ready." -ForegroundColor Yellow
        } else {
            Start-Sleep -Milliseconds $SuspendDelayMs
            try {
                Suspend-McdProcess -ProcessId $proc.Id
                Write-Host "Suspended MCPro9 (PID $($proc.Id)). Copying temp files..." -ForegroundColor Yellow
            } catch {
                Write-Warning "Suspend failed: $_"
            }
        }
    }

    if ($initiallySuspended) {
        Write-Host "Press Enter when you want MCPro9 to start running. The watcher will keep copying temp files." -ForegroundColor Yellow
        [void][Console]::ReadLine()
        Resume-McdProcess -ProcessId $proc.Id
        if ($AutoSuspend) {
            Start-Sleep -Milliseconds $SuspendDelayMs
            try {
                Suspend-McdProcess -ProcessId $proc.Id
                Write-Host "Re-suspended MCPro9 after $SuspendDelayMs ms. Press Enter again to continue." -ForegroundColor Yellow
                [void][Console]::ReadLine()
                Resume-McdProcess -ProcessId $proc.Id
            } catch {
                Write-Warning "Secondary suspend failed: $_"
            }
        }
    } else {
        Write-Host "Press Enter to resume MCPro9 (or close this window to leave it suspended)." -ForegroundColor Yellow
        [void][Console]::ReadLine()
        try {
            Resume-McdProcess -ProcessId $proc.Id
            Write-Host "Resumed MCPro9." -ForegroundColor Cyan
        } catch {
            Write-Warning "Resume failed: $_"
        }
    }

    Write-Host "Watcher stays active. Press Ctrl+C to stop once you're done." -ForegroundColor Cyan
    Wait-Process -Id $proc.Id
} else {
    Write-Host "Watcher armed. Launch MCPro9 normally; any Mcd*.tmp/.cfg files will be copied immediately. Press Ctrl+C to quit." -ForegroundColor Cyan
    while ($true) {
        Start-Sleep -Seconds 1
    }
}
