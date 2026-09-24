# CASTLE installer - Windows PowerShell
#
# Installs the current CASTLE (the `dev` branch of CASTLE-ai/castle-ai) into one
# folder: source code, a Python 3.10 environment (.venv) and the model
# checkpoints (ckpt). Picks CPU or CUDA PyTorch automatically.
#
# Usage (PowerShell):
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/CASTLE-ai/castle-ai/dev/install.ps1 | iex"
#
# Options as environment variables (for the piped one-liner) or parameters:
#   $env:CASTLE_INSTALL_DIR    = "C:\castle"  # -InstallDir  (default: $env:USERPROFILE\castle)
#   $env:CASTLE_CPU_ONLY       = "1"          # -CpuOnly     force CPU PyTorch
#   $env:CASTLE_NO_CHECKPOINTS = "1"          # -NoCheckpoints
#   $env:CASTLE_UNINSTALL      = "1"          # -Uninstall   (keeps the projects folder)
#
# Exit codes: 0 = installed, 1 = failed, 2 = stopped before installing because
# the install folder path contains non-ASCII characters (see message).
# The last line printed is always CASTLE_INSTALL_RESULT=OK or =FAIL / =STOPPED.
#
# Kept ASCII-only: Windows PowerShell 5.1 reads BOM-less scripts in the system
# code page (cp950 on Traditional Chinese Windows).
# ---------------------------------------------------------------------------

[CmdletBinding()]
param(
    [string]$InstallDir = "",
    [switch]$CpuOnly,
    [switch]$NoCheckpoints,
    [switch]$Uninstall,
    [switch]$Help
)

$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'   # faster Invoke-WebRequest

if ($env:OS -ne "Windows_NT") {
    Write-Host "[XX] This script is for Windows. Use install.sh on Linux/macOS." -ForegroundColor Red
    Write-Host "CASTLE_INSTALL_RESULT=FAIL"
    exit 1
}

# -- Read env-var overrides (for piped usage) --------------------------------
if ($env:CASTLE_CPU_ONLY       -eq "1") { $CpuOnly       = $true }
if ($env:CASTLE_NO_CHECKPOINTS -eq "1") { $NoCheckpoints = $true }
if ($env:CASTLE_UNINSTALL      -eq "1") { $Uninstall     = $true }
$ExplicitDir = $false
if ($InstallDir) { $ExplicitDir = $true }
elseif ($env:CASTLE_INSTALL_DIR) { $InstallDir = $env:CASTLE_INSTALL_DIR; $ExplicitDir = $true }
else { $InstallDir = "$env:USERPROFILE\castle" }

$REPO_ZIP   = "https://github.com/CASTLE-ai/castle-ai/archive/refs/heads/dev.zip"
$PYTHON_VER = "3.10"
$Venv       = Join-Path $InstallDir ".venv"
$Py         = Join-Path $Venv "Scripts\python.exe"
$Ckpt       = Join-Path $InstallDir "ckpt"

# -- Checkpoints: file name, source, SHA-256 ---------------------------------
$CHECKPOINTS = @(
    @{ File = "sam_vit_b_01ec64.pth"
       Url  = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
       Sha  = "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912" },
    @{ File = "R50_DeAOTL_PRE_YTB_DAV.pth"
       Gdrive = "1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ"
       Sha  = "7e8a8d83310739bac02817f6bf48b6bbe2bbd7d5325722f1084088eb3aee1e06" },
    @{ File = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
       Gdrive = "18doehnHWWnz9zBtOdgYZ3XMTpgPYbYZ6"
       Sha  = "73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c" }
)

# -- Helpers -----------------------------------------------------------------
function Write-Info { param([string]$Msg) Write-Host "[..] $Msg" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Msg) Write-Host "[OK] $Msg" -ForegroundColor Green }
function Write-Warn { param([string]$Msg) Write-Host "[!!] $Msg" -ForegroundColor Yellow }
function Write-Err  { param([string]$Msg) Write-Host "[XX] $Msg" -ForegroundColor Red }
function Exit-Fail {
    param([string]$Msg)
    Write-Err $Msg
    Write-Host "CASTLE_INSTALL_RESULT=FAIL"
    exit 1
}

if ($Help) {
    Write-Host "Usage: .\install.ps1 [-InstallDir DIR] [-CpuOnly] [-NoCheckpoints] [-Uninstall]"
    Write-Host "Same options as env vars for the piped one-liner:"
    Write-Host "  CASTLE_INSTALL_DIR, CASTLE_CPU_ONLY=1, CASTLE_NO_CHECKPOINTS=1, CASTLE_UNINSTALL=1"
    exit 0
}


# -- Uninstall (keeps projects\) ---------------------------------------------
if ($Uninstall) {
    if (-not (Test-Path $InstallDir)) { Write-Ok "Nothing to remove at $InstallDir"; exit 0 }
    Get-ChildItem -LiteralPath $InstallDir -Force | Where-Object { $_.Name -ne "projects" } |
        Remove-Item -Recurse -Force
    Write-Ok "CASTLE removed from $InstallDir (the projects folder, if any, was kept)."
    exit 0
}

Write-Host ""
Write-Host "  CASTLE installer (Windows)" -ForegroundColor Cyan
Write-Host "  Install folder: $InstallDir"
Write-Host ""

# -- 1. Install folder must be decided by the user if it is non-ASCII --------
# Some video/image libraries fail on paths with e.g. Chinese characters, which a
# Chinese Windows user name puts into the default location.
if ((-not $ExplicitDir) -and ($InstallDir -match '[^\x00-\x7F]')) {
    Write-Warn "The install folder contains non-English characters:"
    Write-Warn "    $InstallDir"
    Write-Warn "Some video tools can fail on such paths. Choose one and run the installer again:"
    Write-Warn '  a) an English-only folder (recommended), e.g.  $env:CASTLE_INSTALL_DIR = "C:\castle"'
    Write-Warn "  b) keep this folder anyway:                     `$env:CASTLE_INSTALL_DIR = `"$InstallDir`""
    Write-Host "CASTLE_INSTALL_RESULT=STOPPED"
    exit 2
}

# -- 2. uv (brings its own Python; no system Python needed) ------------------
function Install-Uv {
    if (Get-Command uv -ErrorAction SilentlyContinue) { Write-Ok "uv found: $(& uv --version)"; return }
    Write-Info "Installing uv ..."
    Invoke-Expression "& { $(Invoke-RestMethod https://astral.sh/uv/install.ps1) }"
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "User") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "Machine")
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Exit-Fail "uv installation failed - 'uv' not found on PATH" }
    Write-Ok "uv installed: $(& uv --version)"
}

# -- 3. Source code (dev branch zip; no git needed) --------------------------
function Get-Source {
    Write-Info "Downloading CASTLE (dev branch) ..."
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ("castle-" + [guid]::NewGuid())
    New-Item -ItemType Directory -Path $tmp -Force | Out-Null
    try {
        $zip = Join-Path $tmp "castle.zip"
        Invoke-WebRequest -Uri $REPO_ZIP -OutFile $zip -UseBasicParsing
        Expand-Archive -LiteralPath $zip -DestinationPath $tmp -Force
        $src = Get-ChildItem -LiteralPath $tmp -Directory | Select-Object -First 1
        New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
        # Overwrite code in place; projects\ and downloaded ckpt files are not in the zip.
        Copy-Item -Path (Join-Path $src.FullName "*") -Destination $InstallDir -Recurse -Force
    }
    catch { Exit-Fail "Could not download or unpack CASTLE: $_" }
    finally { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue }
    Write-Ok "Source code in $InstallDir"
}

# -- 4. PyTorch backend -------------------------------------------------------
function Get-CudaBackend {
    if ($CpuOnly) { return "cpu" }
    $nvSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvSmi) { return "cpu" }
    try {
        $match = (& $nvSmi.Source 2>$null) | Select-String "CUDA Version:\s+([\d.]+)"
        if (-not $match) { return "cpu" }
        $parts = $match.Matches[0].Groups[1].Value -split "\."
        $major = [int]$parts[0]; $minor = [int]$parts[1]
        if ($major -ge 12) {
            if ($minor -ge 6) { return "cu126" } elseif ($minor -ge 4) { return "cu124" } else { return "cu121" }
        }
        elseif ($major -eq 11 -and $minor -ge 8) { return "cu118" }
        Write-Warn "CUDA $($match.Matches[0].Groups[1].Value) is too old for current PyTorch - using CPU"
        return "cpu"
    }
    catch { return "cpu" }
}

# -- 5. Python environment ----------------------------------------------------
function Install-Env {
    param([string]$Cuda)
    if (-not (Test-Path $Py)) {
        Write-Info "Creating Python $PYTHON_VER environment ..."
        & uv venv $Venv --python $PYTHON_VER
        if ($LASTEXITCODE -ne 0) { Exit-Fail "uv venv failed" }
    }
    Write-Info "Installing PyTorch ($Cuda) ..."
    & uv pip install --python $Py torch torchvision --index-url "https://download.pytorch.org/whl/$Cuda"
    if ($LASTEXITCODE -ne 0) { Exit-Fail "PyTorch install failed" }
    # Core only: the GPU accelerators (cuML, xformers) have no Windows builds.
    Write-Info "Installing CASTLE ..."
    & uv pip install --python $Py -e $InstallDir
    if ($LASTEXITCODE -ne 0) { Exit-Fail "CASTLE install failed" }
    Write-Ok "Python environment ready"
}

# -- 6. Checkpoints (verified by SHA-256) ------------------------------------
function Get-Checkpoints {
    New-Item -ItemType Directory -Path $Ckpt -Force | Out-Null
    $failed = @()
    foreach ($c in $CHECKPOINTS) {
        $dest = Join-Path $Ckpt $c.File
        if ((Test-Path $dest) -and ((Get-FileHash $dest -Algorithm SHA256).Hash -eq $c.Sha)) {
            Write-Ok "Already present: $($c.File)"; continue
        }
        Write-Info "Downloading $($c.File) ..."
        try {
            if ($c.Url) { Invoke-WebRequest -Uri $c.Url -OutFile $dest -UseBasicParsing }
            else { & $Py -m gdown $c.Gdrive -O $dest }
        }
        catch { }
        if ((Test-Path $dest) -and ((Get-FileHash $dest -Algorithm SHA256).Hash -eq $c.Sha)) {
            Write-Ok "Downloaded $($c.File)"
        }
        else {
            Remove-Item -LiteralPath $dest -Force -ErrorAction SilentlyContinue
            $failed += $c
        }
    }
    if ($failed.Count -gt 0) {
        Write-Err "These model files could not be downloaded (Google Drive may be rate-limiting):"
        foreach ($c in $failed) {
            $src = if ($c.Url) { $c.Url } else { "https://drive.google.com/file/d/$($c.Gdrive)/view" }
            Write-Err "    $($c.File)   <-  $src"
        }
        Write-Err "Download them in a browser, put them in $Ckpt, then run the installer again."
        Exit-Fail "Checkpoints missing"
    }
}

# -- 7. Self-check ------------------------------------------------------------
function Test-Install {
    & $Py -c "import torch, gradio, av, cv2, castle.core.models, castle.service.clip_service; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
    if ($LASTEXITCODE -ne 0) { Exit-Fail "Import check failed" }
}

# -- Run ----------------------------------------------------------------------
Install-Uv
Get-Source
$cuda = Get-CudaBackend
Write-Info "PyTorch backend: $cuda"
Install-Env -Cuda $cuda
if ($NoCheckpoints) { Write-Warn "Skipping checkpoint download (CASTLE_NO_CHECKPOINTS)" } else { Get-Checkpoints }
Test-Install

Write-Host ""
Write-Ok "CASTLE is installed in $InstallDir"
Write-Host "  Start CASTLE (keep the window open while you use it):"
Write-Host "      cd `"$InstallDir`"" -ForegroundColor Cyan
Write-Host "      .\.venv\Scripts\python.exe app.py" -ForegroundColor Cyan
Write-Host "  Then open http://127.0.0.1:7860 in your browser."
Write-Host ""
Write-Host "CASTLE_INSTALL_RESULT=OK"
exit 0
