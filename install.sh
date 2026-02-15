#!/usr/bin/env bash
# CASTLE One-Line Installer — Linux / macOS
# Usage:
#   curl -fsSL https://castle-ai.github.io/install.sh | bash
#   curl -fsSL https://castle-ai.github.io/install.sh | bash -s -- --cpu-only
#   curl -fsSL https://castle-ai.github.io/install.sh | bash -s -- --uninstall
#
# Options:
#   --cpu-only          Force CPU-only PyTorch (skip CUDA detection)
#   --no-checkpoints    Skip model checkpoint download (~2 GB)
#   --version VER       Install a specific castle-ai version (default: latest)
#   --uninstall         Remove the CASTLE installation entirely
# ---------------------------------------------------------------------------
set -euo pipefail

CASTLE_HOME="${CASTLE_HOME:-$HOME/.castle}"
CASTLE_BIN="$CASTLE_HOME/bin"
CASTLE_CKPT="$CASTLE_HOME/ckpt"
CASTLE_VENV="$CASTLE_HOME/venv"
CASTLE_VERSION=""          # empty → latest
PYTHON_VERSION="3.10"
CPU_ONLY=false
NO_CHECKPOINTS=false
UNINSTALL=false

# ── Checkpoint URLs ──────────────────────────────────────────────────────────
SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_FILE="sam_vit_b_01ec64.pth"

DEAOT_GDRIVE_ID="1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ"
DEAOT_FILE="R50_DeAOTL_PRE_YTB_DAV.pth"

DINOV2_URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
DINOV2_FILE="dinov2_vitb14_reg4_pretrain.pth"

# ── Colours ──────────────────────────────────────────────────────────────────
if [[ -t 1 ]] && [[ "${TERM:-}" != "dumb" ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; BOLD=''; NC=''
fi

info()  { printf "${BLUE}ℹ${NC}  %s\n" "$*"; }
ok()    { printf "${GREEN}✓${NC}  %s\n" "$*"; }
warn()  { printf "${YELLOW}⚠${NC}  %s\n" "$*" >&2; }
err()   { printf "${RED}✗${NC}  %s\n" "$*" >&2; }
die()   { err "$*"; exit 1; }

banner() {
    printf "${CYAN}"
    cat <<'EOF'

  ╔═══════════════════════════════════════╗
  ║          CASTLE Installer             ║
  ║  Combined Approach for Segmentation   ║
  ║  and Tracking with Latent Extraction  ║
  ╚═══════════════════════════════════════╝

EOF
    printf "${NC}"
}

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu-only)       CPU_ONLY=true;       shift ;;
        --no-checkpoints) NO_CHECKPOINTS=true; shift ;;
        --uninstall)      UNINSTALL=true;      shift ;;
        --version)
            [[ -n "${2:-}" ]] || die "--version requires a value"
            CASTLE_VERSION="$2"; shift 2 ;;
        -h|--help)
            banner
            cat <<'HELP'
  Options:
    --cpu-only          Force CPU-only PyTorch
    --no-checkpoints    Skip model download (~2 GB)
    --version VER       Install a specific version
    --uninstall         Remove CASTLE
    -h, --help          Show this message
HELP
            exit 0 ;;
        *) die "Unknown option: $1" ;;
    esac
done

# ── Uninstall ────────────────────────────────────────────────────────────────
if [[ "$UNINSTALL" == true ]]; then
    banner
    info "Uninstalling CASTLE …"
    rm -rf "$CASTLE_HOME"
    rm -f "$HOME/.local/bin/castle"
    ok "CASTLE uninstalled."
    exit 0
fi

# ── Banner ───────────────────────────────────────────────────────────────────
banner

# ── Platform detection ───────────────────────────────────────────────────────
detect_platform() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Linux)  ;;
        Darwin) ;;
        *)      die "Unsupported OS: $OS  (only Linux and macOS are supported)" ;;
    esac

    case "$ARCH" in
        x86_64|amd64) ARCH="x86_64" ;;
        aarch64|arm64) ARCH="arm64" ;;
        *) die "Unsupported architecture: $ARCH  (need x86_64 or arm64)" ;;
    esac

    info "Platform: $OS $ARCH"
}

# ── uv ───────────────────────────────────────────────────────────────────────
install_uv() {
    if command -v uv &>/dev/null; then
        ok "uv already installed: $(uv --version)"
        return
    fi
    info "Installing uv …"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Ensure uv is on PATH for the rest of the script
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv &>/dev/null || die "uv installation failed — 'uv' not found on PATH"
    ok "uv installed: $(uv --version)"
}

# ── CUDA detection ───────────────────────────────────────────────────────────
detect_cuda() {
    if [[ "$CPU_ONLY" == true ]]; then
        echo "cpu"; return
    fi

    # macOS → no NVIDIA CUDA
    if [[ "$OS" == "Darwin" ]]; then
        echo "cpu"; return
    fi

    if ! command -v nvidia-smi &>/dev/null; then
        echo "cpu"; return
    fi

    local cuda_ver
    cuda_ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)
    if [[ -z "$cuda_ver" ]]; then
        echo "cpu"; return
    fi

    local major minor
    major=$(echo "$cuda_ver" | cut -d. -f1)
    minor=$(echo "$cuda_ver" | cut -d. -f2)

    if (( major >= 12 )); then
        if (( minor >= 6 )); then echo "cu126"
        elif (( minor >= 4 )); then echo "cu124"
        else echo "cu121"; fi
    elif (( major == 11 && minor >= 8 )); then
        echo "cu118"
    else
        warn "CUDA $cuda_ver is too old for current PyTorch — falling back to CPU"
        echo "cpu"
    fi
}

# ── GPU display name (for summary) ──────────────────────────────────────────
gpu_display() {
    local cuda="$1"
    if [[ "$cuda" == "cpu" ]]; then
        echo "CPU only"
        return
    fi
    local gpu_name cuda_ver
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    cuda_ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || true)
    if [[ -n "$gpu_name" ]]; then
        echo "CUDA $cuda_ver ($gpu_name)"
    else
        echo "CUDA ($cuda)"
    fi
}

# ── Create venv & install packages ───────────────────────────────────────────
install_castle() {
    local cuda="$1"
    local pkg="castle-ai"
    [[ -n "$CASTLE_VERSION" ]] && pkg="castle-ai==$CASTLE_VERSION"

    # Create (or reuse) venv
    if [[ -d "$CASTLE_VENV" ]]; then
        info "Reusing existing venv at $CASTLE_VENV"
    else
        info "Creating Python $PYTHON_VERSION venv …"
        uv venv "$CASTLE_VENV" --python "$PYTHON_VERSION"
    fi

    local py="$CASTLE_VENV/bin/python"

    # Install PyTorch + torchvision with the right index
    info "Installing PyTorch (backend: $cuda) …"
    if [[ "$cuda" == "cpu" ]]; then
        uv pip install --python "$py" \
            torch torchvision --index-url https://download.pytorch.org/whl/cpu
    else
        uv pip install --python "$py" \
            torch torchvision --index-url "https://download.pytorch.org/whl/$cuda"
    fi

    # Install CASTLE itself (skip xformers & cuml-cu12 on CPU / macOS to avoid build errors)
    info "Installing $pkg …"
    if [[ "$cuda" == "cpu" ]] || [[ "$OS" == "Darwin" ]]; then
        # Install without GPU-specific extras that would fail on CPU/macOS
        uv pip install --python "$py" "$pkg" \
            --no-deps 2>/dev/null || true
        # Install deps manually, skipping problematic ones
        uv pip install --python "$py" \
            torchmetrics numpy scipy scikit-learn h5py matplotlib plotly \
            av opencv-python-headless Pillow umap-learn gradio \
            typer rich tqdm natsort termcolor gdown 2>/dev/null || true
    else
        uv pip install --python "$py" "$pkg"
    fi

    ok "CASTLE installed"

    # Symlink ckpt/ into the package so DEFAULT_CKPT_DIR resolves correctly
    # config.py uses: Path(__file__).resolve().parent.parent.parent / 'ckpt'
    # For pip install → site-packages/castle/core/config.py → 3 up = site-packages/
    local config_py
    config_py=$("$py" -c "import castle.core.config as c; print(c.__file__)" 2>/dev/null || true)
    if [[ -n "$config_py" ]]; then
        local pkg_base
        pkg_base=$(cd "$(dirname "$config_py")/../.." && pwd)
        mkdir -p "$CASTLE_CKPT"
        if [[ ! -e "$pkg_base/ckpt" ]]; then
            ln -sf "$CASTLE_CKPT" "$pkg_base/ckpt"
            info "Linked $pkg_base/ckpt → $CASTLE_CKPT"
        fi
    fi
}

# ── Download checkpoints ────────────────────────────────────────────────────
download_file() {
    local url="$1" dest="$2"
    if [[ -f "$dest" ]]; then
        ok "Already exists: $(basename "$dest")"
        return
    fi
    info "Downloading $(basename "$dest") …"
    curl -fL --progress-bar -o "$dest" "$url"
    ok "Downloaded $(basename "$dest")"
}

download_gdrive() {
    local id="$1" dest="$2"
    if [[ -f "$dest" ]]; then
        ok "Already exists: $(basename "$dest")"
        return
    fi
    info "Downloading $(basename "$dest") from Google Drive …"
    # Use gdown from the castle venv (it's a dependency)
    local py="$CASTLE_VENV/bin/python"
    "$py" -m gdown "$id" -O "$dest" 2>&1 || {
        # Fallback: try direct URL
        local fallback="https://drive.google.com/uc?export=download&id=$id&confirm=t"
        warn "gdown failed — trying direct download …"
        curl -fL --progress-bar -o "$dest" "$fallback" || die "Failed to download $(basename "$dest")"
    }
    ok "Downloaded $(basename "$dest")"
}

download_checkpoints() {
    if [[ "$NO_CHECKPOINTS" == true ]]; then
        warn "Skipping checkpoint download (--no-checkpoints)"
        return
    fi

    mkdir -p "$CASTLE_CKPT"
    info "Downloading model checkpoints to $CASTLE_CKPT …"

    download_file   "$SAM_URL"   "$CASTLE_CKPT/$SAM_FILE"
    download_gdrive "$DEAOT_GDRIVE_ID" "$CASTLE_CKPT/$DEAOT_FILE"
    download_file   "$DINOV2_URL" "$CASTLE_CKPT/$DINOV2_FILE"

    ok "All checkpoints ready"
}

# ── Global command wrapper ───────────────────────────────────────────────────
setup_command() {
    mkdir -p "$CASTLE_BIN"

    cat > "$CASTLE_BIN/castle" <<'WRAPPER'
#!/usr/bin/env bash
CASTLE_HOME="${CASTLE_HOME:-$HOME/.castle}"
exec "$CASTLE_HOME/venv/bin/castle" "$@"
WRAPPER
    chmod +x "$CASTLE_BIN/castle"

    # Symlink into ~/.local/bin
    mkdir -p "$HOME/.local/bin"
    ln -sf "$CASTLE_BIN/castle" "$HOME/.local/bin/castle"

    # Check if ~/.local/bin is on PATH
    if ! echo "$PATH" | tr ':' '\n' | grep -qx "$HOME/.local/bin"; then
        warn "\$HOME/.local/bin is not in your PATH."
        warn "Add it by appending to your shell config:"
        warn "  echo 'export PATH=\"\$HOME/.local/bin:\$PATH\"' >> ~/.bashrc"
    fi

    ok "Global 'castle' command installed"
}

# ── Version marker ───────────────────────────────────────────────────────────
write_version() {
    local py="$CASTLE_VENV/bin/python"
    local ver
    ver=$("$py" -c "import castle; print(castle.__version__)" 2>/dev/null || \
          "$py" -c "from importlib.metadata import version; print(version('castle-ai'))" 2>/dev/null || \
          echo "${CASTLE_VERSION:-unknown}")
    echo "$ver" > "$CASTLE_HOME/version"
    echo "$ver"
}

# ── Summary ──────────────────────────────────────────────────────────────────
print_summary() {
    local ver="$1" cuda="$2"
    local gpu_info
    gpu_info=$(gpu_display "$cuda")

    local sam_ok="✗" deaot_ok="✗" dinov2_ok="✗"
    [[ -f "$CASTLE_CKPT/$SAM_FILE" ]]   && sam_ok="✓"
    [[ -f "$CASTLE_CKPT/$DEAOT_FILE" ]] && deaot_ok="✓"
    [[ -f "$CASTLE_CKPT/$DINOV2_FILE" ]] && dinov2_ok="✓"

    printf "\n"
    printf "  ${GREEN}✅ CASTLE installed successfully!${NC}\n"
    printf "\n"
    printf "  ${BOLD}Version:${NC}  %s\n" "$ver"
    printf "  ${BOLD}Location:${NC} %s\n" "$CASTLE_HOME"
    printf "  ${BOLD}GPU:${NC}      %s\n" "$gpu_info"
    printf "  ${BOLD}Models:${NC}   SAM %s  DeAOT %s  DINOv2 %s\n" "$sam_ok" "$deaot_ok" "$dinov2_ok"
    printf "\n"
    printf "  Run ${CYAN}castle --help${NC} to get started.\n"
    printf "  Run ${CYAN}castle gui${NC}    to launch the desktop GUI.\n"
    printf "\n"
    printf "  To uninstall:\n"
    printf "    curl -fsSL https://castle-ai.github.io/install.sh | bash -s -- --uninstall\n"
    printf "\n"
}

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
main() {
    detect_platform
    install_uv

    local cuda
    cuda=$(detect_cuda)
    info "Compute backend: $cuda"

    install_castle "$cuda"
    download_checkpoints
    setup_command

    local ver
    ver=$(write_version)
    print_summary "$ver" "$cuda"
}

main
