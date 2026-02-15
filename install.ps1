# CASTLE One-Line Installer — Windows PowerShell
# Usage:
#   irm https://castle-ai.github.io/install.ps1 | iex
#   powershell -ExecutionPolicy ByPass -c "irm https://castle-ai.github.io/install.ps1 | iex"
#
# Options (set as environment variables before piping):
#   $env:CASTLE_CPU_ONLY       = "1"   # Force CPU-only PyTorch
#   $env:CASTLE_NO_CHECKPOINTS = "1"   # Skip model download (~2 GB)
#   $env:CASTLE_VERSION        = "0.0.18"  # Specific version
#   $env:CASTLE_UNINSTALL      = "1"   # Remove CASTLE
#
# Or run the script directly with parameters:
#   .\install.ps1 -CpuOnly -NoCheckpoints -Version "0.0.18"
#   .\install.ps1 -Uninstall
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    [switch]$CpuOnly,
    [switch]$NoCheckpoints,
    [string]$Version = "",
    [switch]$Uninstall,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'   # faster Invoke-WebRequest

# ── Read env-var overrides (for piped usage) ─────────────────────────────────
if ($env:CASTLE_CPU_ONLY       -eq "1") { $CpuOnly       = $true }
if ($env:CASTLE_NO_CHECKPOINTS -eq "1") { $NoCheckpoints  = $true }
if ($env:CASTLE_UNINSTALL      -eq "1") { $Uninstall      = $true }
if ($env:CASTLE_VERSION -and -not $Version) { $Version = $env:CASTLE_VERSION }

$CASTLE_HOME   = if ($env:CASTLE_HOME) { $env:CASTLE_HOME } else { "$env:USERPROFILE\.castle" }
$CASTLE_BIN    = "$CASTLE_HOME\bin"
$CASTLE_CKPT   = "$CASTLE_HOME\ckpt"
$CASTLE_VENV   = "$CASTLE_HOME\venv"
$PYTHON_VER    = "3.10"

# ── Checkpoint URLs ──────────────────────────────────────────────────────────
$SAM_URL       = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
$SAM_FILE      = "sam_vit_b_01ec64.pth"

$DEAOT_GDRIVE  = "1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ"
$DEAOT_FILE    = "R50_DeAOTL_PRE_YTB_DAV.pth"

$DINOV2_URL    = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
$DINOV2_FILE   = "dinov2_vitb14_reg4_pretrain.pth"

# ── Helpers ──────────────────────────────────────────────────────────────────
function Write-Info  { param([string]$Msg) Write-Host "ℹ  $Msg" -ForegroundColor Blue }
function Write-Ok    { param([string]$Msg) Write-Host "✓  $Msg" -ForegroundColor Green }
function Write-Warn  { param([string]$Msg) Write-Host "⚠  $Msg" -ForegroundColor Yellow }
function Write-Err   { param([string]$Msg) Write-Host "✗  $Msg" -ForegroundColor Red }
function Exit-Fatal  { param([string]$Msg) Write-Err $Msg; exit 1 }

function Show-Banner {
    Write-Host ""
    Write-Host "  ╔═══════════════════════════════════════╗" -ForegroundColor Cyan
    Write-Host "  ║          CASTLE Installer             ║" -ForegroundColor Cyan
    Write-Host "  ║  Combined Approach for Segmentation   ║" -ForegroundColor Cyan
    Write-Host "  ║  and Tracking with Latent Extraction  ║" -ForegroundColor Cyan
    Write-Host "  ╚═══════════════════════════════════════╝" -ForegroundColor Cyan
    Write-Host ""
}

# ── Help ─────────────────────────────────────────────────────────────────────
if ($Help) {
    Show-Banner
    Write-Host @"
  Options (as parameters):
    -CpuOnly          Force CPU-only PyTorch
    -NoCheckpoints    Skip model download (~2 GB)
    -Version VER      Install a specific version
    -Uninstall        Remove CASTLE
    -Help             Show this message

  Options (as env vars, for piped usage):
    `$env:CASTLE_CPU_ONLY       = "1"
    `$env:CASTLE_NO_CHECKPOINTS = "1"
    `$env:CASTLE_VERSION        = "0.0.18"
    `$env:CASTLE_UNINSTALL      = "1"
"@
    exit 0
}

# ── Uninstall ────────────────────────────────────────────────────────────────
if ($Uninstall) {
    Show-Banner
    Write-Info "Uninstalling CASTLE …"
    if (Test-Path $CASTLE_HOME) { Remove-Item -Recurse -Force $CASTLE_HOME }

    # Remove from user PATH
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -and $userPath.Contains($CASTLE_BIN)) {
        $newPath = ($userPath -split ";" | Where-Object { $_ -ne $CASTLE_BIN }) -join ";"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        Write-Info "Removed $CASTLE_BIN from user PATH"
    }

    Write-Ok "CASTLE uninstalled."
    exit 0
}

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
Show-Banner

# ── Platform check ───────────────────────────────────────────────────────────
if ($env:OS -ne "Windows_NT") {
    Exit-Fatal "This script is for Windows. Use install.sh on Linux/macOS."
}
$arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
Write-Info "Platform: Windows $arch"

# ── Install uv ───────────────────────────────────────────────────────────────
function Install-Uv {
    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($uvCmd) {
        Write-Ok "uv already installed: $(& uv --version)"
        return
    }
    Write-Info "Installing uv …"
    Invoke-Expression "& { $(Invoke-RestMethod https://astral.sh/uv/install.ps1) }"

    # Refresh PATH
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "Machine")

    $uvCmd = Get-Command uv -ErrorAction SilentlyContinue
    if (-not $uvCmd) { Exit-Fatal "uv installation failed — 'uv' not found on PATH" }
    Write-Ok "uv installed: $(& uv --version)"
}

# ── CUDA detection ───────────────────────────────────────────────────────────
function Get-CudaBackend {
    if ($CpuOnly) { return "cpu" }

    # Try nvidia-smi
    $nvSmi = $null
    foreach ($candidate in @("nvidia-smi", "C:\Windows\System32\nvidia-smi.exe")) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) { $nvSmi = $cmd.Source; break }
    }
    if (-not $nvSmi) { return "cpu" }

    try {
        $output = & $nvSmi 2>$null
        $match  = $output | Select-String "CUDA Version:\s+([\d.]+)"
        if (-not $match) { return "cpu" }
        $cudaVer = $match.Matches[0].Groups[1].Value
        $parts   = $cudaVer -split "\."
        $major   = [int]$parts[0]
        $minor   = [int]$parts[1]

        if ($major -ge 12) {
            if ($minor -ge 6) { return "cu126" }
            elseif ($minor -ge 4) { return "cu124" }
            else { return "cu121" }
        }
        elseif ($major -eq 11 -and $minor -ge 8) { return "cu118" }
        else {
            Write-Warn "CUDA $cudaVer is too old for current PyTorch — falling back to CPU"
            return "cpu"
        }
    }
    catch { return "cpu" }
}

# ── GPU display ──────────────────────────────────────────────────────────────
function Get-GpuDisplay {
    param([string]$Cuda)
    if ($Cuda -eq "cpu") { return "CPU only" }
    try {
        $name    = (& nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Select-Object -First 1).Trim()
        $output  = & nvidia-smi 2>$null
        $match   = $output | Select-String "CUDA Version:\s+([\d.]+)"
        $cudaVer = if ($match) { $match.Matches[0].Groups[1].Value } else { $Cuda }
        if ($name) { return "CUDA $cudaVer ($name)" }
        return "CUDA ($Cuda)"
    }
    catch { return "CUDA ($Cuda)" }
}

# ── Install CASTLE ───────────────────────────────────────────────────────────
function Install-Castle {
    param([string]$Cuda)

    $pkg = "castle-ai"
    if ($Version) { $pkg = "castle-ai==$Version" }
    $py  = "$CASTLE_VENV\Scripts\python.exe"

    # Create / reuse venv
    if (Test-Path $CASTLE_VENV) {
        Write-Info "Reusing existing venv at $CASTLE_VENV"
    } else {
        Write-Info "Creating Python $PYTHON_VER venv …"
        & uv venv $CASTLE_VENV --python $PYTHON_VER
    }

    # PyTorch
    Write-Info "Installing PyTorch (backend: $Cuda) …"
    if ($Cuda -eq "cpu") {
        & uv pip install --python $py torch torchvision --index-url https://download.pytorch.org/whl/cpu
    } else {
        & uv pip install --python $py torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
    }

    # CASTLE
    Write-Info "Installing $pkg …"
    try {
        & uv pip install --python $py $pkg
    }
    catch {
        Write-Warn "Full install failed — trying without GPU-specific extras …"
        & uv pip install --python $py $pkg --no-deps 2>$null
        & uv pip install --python $py `
            torchmetrics numpy scipy scikit-learn h5py matplotlib plotly `
            av opencv-python-headless Pillow umap-learn gradio `
            typer rich tqdm natsort termcolor gdown 2>$null
    }
    Write-Ok "CASTLE installed"

    # Symlink ckpt/ into package for DEFAULT_CKPT_DIR resolution
    try {
        $configPy = & $py -c "import castle.core.config as c; print(c.__file__)" 2>$null
        if ($configPy) {
            $pkgBase = (Resolve-Path (Join-Path (Split-Path $configPy) "..\.." )).Path
            $ckptTarget = Join-Path $pkgBase "ckpt"
            if (-not (Test-Path $CASTLE_CKPT)) { New-Item -ItemType Directory -Path $CASTLE_CKPT -Force | Out-Null }
            if (-not (Test-Path $ckptTarget)) {
                # Use junction (works without admin on Windows)
                cmd /c mklink /J "$ckptTarget" "$CASTLE_CKPT" 2>$null | Out-Null
                Write-Info "Linked $ckptTarget → $CASTLE_CKPT"
            }
        }
    }
    catch { Write-Warn "Could not auto-link checkpoint directory — you may need to set CASTLE_CKPT manually" }
}

# ── Download checkpoints ────────────────────────────────────────────────────
function Download-File {
    param([string]$Url, [string]$Dest)
    if (Test-Path $Dest) {
        Write-Ok "Already exists: $(Split-Path $Dest -Leaf)"
        return
    }
    Write-Info "Downloading $(Split-Path $Dest -Leaf) …"
    $ProgressPreference = 'Continue'
    Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
    $ProgressPreference = 'SilentlyContinue'
    Write-Ok "Downloaded $(Split-Path $Dest -Leaf)"
}

function Download-GDrive {
    param([string]$Id, [string]$Dest)
    if (Test-Path $Dest) {
        Write-Ok "Already exists: $(Split-Path $Dest -Leaf)"
        return
    }
    Write-Info "Downloading $(Split-Path $Dest -Leaf) from Google Drive …"
    $py = "$CASTLE_VENV\Scripts\python.exe"
    try {
        & $py -m gdown $Id -O $Dest 2>&1
    }
    catch {
        Write-Warn "gdown failed — trying direct download …"
        $url = "https://drive.google.com/uc?export=download&id=$Id&confirm=t"
        Invoke-WebRequest -Uri $url -OutFile $Dest -UseBasicParsing
    }
    if (-not (Test-Path $Dest)) { Exit-Fatal "Failed to download $(Split-Path $Dest -Leaf)" }
    Write-Ok "Downloaded $(Split-Path $Dest -Leaf)"
}

function Download-Checkpoints {
    if ($NoCheckpoints) {
        Write-Warn "Skipping checkpoint download (--no-checkpoints)"
        return
    }
    if (-not (Test-Path $CASTLE_CKPT)) { New-Item -ItemType Directory -Path $CASTLE_CKPT -Force | Out-Null }
    Write-Info "Downloading model checkpoints to $CASTLE_CKPT …"

    Download-File   -Url $SAM_URL       -Dest "$CASTLE_CKPT\$SAM_FILE"
    Download-GDrive -Id  $DEAOT_GDRIVE  -Dest "$CASTLE_CKPT\$DEAOT_FILE"
    Download-File   -Url $DINOV2_URL    -Dest "$CASTLE_CKPT\$DINOV2_FILE"

    Write-Ok "All checkpoints ready"
}

# ── Global command ───────────────────────────────────────────────────────────
function Setup-Command {
    if (-not (Test-Path $CASTLE_BIN)) { New-Item -ItemType Directory -Path $CASTLE_BIN -Force | Out-Null }

    # Create castle.cmd wrapper
    $cmd = @"
@echo off
set "CASTLE_HOME=%USERPROFILE%\.castle"
"%CASTLE_HOME%\venv\Scripts\castle.exe" %*
"@
    Set-Content -Path "$CASTLE_BIN\castle.cmd" -Value $cmd -Encoding ASCII

    # Add to user PATH if not already there
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not $userPath -or -not $userPath.Contains($CASTLE_BIN)) {
        $newPath = if ($userPath) { "$CASTLE_BIN;$userPath" } else { $CASTLE_BIN }
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        $env:Path = "$CASTLE_BIN;$env:Path"
        Write-Info "Added $CASTLE_BIN to user PATH"
        Write-Warn "Restart your terminal for PATH changes to take effect."
    }

    Write-Ok "Global 'castle' command installed"
}

# ── Version marker ───────────────────────────────────────────────────────────
function Write-VersionMarker {
    $py = "$CASTLE_VENV\Scripts\python.exe"
    try {
        $ver = & $py -c "from importlib.metadata import version; print(version('castle-ai'))" 2>$null
    } catch { $ver = $null }
    if (-not $ver) { $ver = if ($Version) { $Version } else { "unknown" } }
    Set-Content -Path "$CASTLE_HOME\version" -Value $ver
    return $ver
}

# ── Summary ──────────────────────────────────────────────────────────────────
function Show-Summary {
    param([string]$Ver, [string]$Cuda)
    $gpuInfo = Get-GpuDisplay $Cuda

    $samOk   = if (Test-Path "$CASTLE_CKPT\$SAM_FILE")   { "✓" } else { "✗" }
    $deaotOk = if (Test-Path "$CASTLE_CKPT\$DEAOT_FILE") { "✓" } else { "✗" }
    $dinoOk  = if (Test-Path "$CASTLE_CKPT\$DINOV2_FILE") { "✓" } else { "✗" }

    Write-Host ""
    Write-Host "  ✅ CASTLE installed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Version:  $Ver"
    Write-Host "  Location: $CASTLE_HOME"
    Write-Host "  GPU:      $gpuInfo"
    Write-Host "  Models:   SAM $samOk  DeAOT $deaotOk  DINOv2 $dinoOk"
    Write-Host ""
    Write-Host "  Run " -NoNewline; Write-Host "castle --help" -ForegroundColor Cyan -NoNewline; Write-Host " to get started."
    Write-Host "  Run " -NoNewline; Write-Host "castle gui" -ForegroundColor Cyan -NoNewline; Write-Host "    to launch the desktop GUI."
    Write-Host ""
    Write-Host "  To uninstall:"
    Write-Host '    powershell -c "$env:CASTLE_UNINSTALL=1; irm https://castle-ai.github.io/install.ps1 | iex"'
    Write-Host ""
}

# ── Run ──────────────────────────────────────────────────────────────────────
Install-Uv

$cuda = Get-CudaBackend
Write-Info "Compute backend: $cuda"

Install-Castle -Cuda $cuda
Download-Checkpoints
Setup-Command

$ver = Write-VersionMarker
Show-Summary -Ver $ver -Cuda $cuda
