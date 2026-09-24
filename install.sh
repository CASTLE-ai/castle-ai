#!/usr/bin/env bash
# CASTLE installer — Linux / macOS
#
# Installs the current CASTLE (the `dev` branch of CASTLE-ai/castle-ai) into one
# folder: source code, a Python 3.10 environment (.venv) and the model
# checkpoints (ckpt). Picks CPU, CUDA or Apple-silicon PyTorch automatically.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/CASTLE-ai/castle-ai/dev/install.sh | bash
#   curl -fsSL https://raw.githubusercontent.com/CASTLE-ai/castle-ai/dev/install.sh | bash -s -- --install-dir ~/castle
#
# Options:
#   --install-dir DIR   Install folder (default: ~/castle; env CASTLE_INSTALL_DIR)
#   --cpu-only          Force CPU-only PyTorch (skip NVIDIA detection)
#   --no-checkpoints    Skip model checkpoint download
#   --uninstall         Remove CASTLE from the install folder (keeps projects/)
#
# Exit codes: 0 = installed, 1 = failed, 2 = stopped before installing because
# the install folder path contains non-ASCII characters (see message).
# The last line printed is always CASTLE_INSTALL_RESULT=OK or =FAIL / =STOPPED.
# ---------------------------------------------------------------------------
set -euo pipefail

INSTALL_DIR="${CASTLE_INSTALL_DIR:-}"
EXPLICIT_DIR=false
[[ -n "$INSTALL_DIR" ]] && EXPLICIT_DIR=true
PYTHON_VERSION="3.10"
CPU_ONLY=false
NO_CHECKPOINTS=false
UNINSTALL=false
REPO_TARBALL="https://github.com/CASTLE-ai/castle-ai/archive/refs/heads/dev.tar.gz"

# ── Checkpoints: file | source (URL or Google Drive id) | SHA-256 ───────────
CHECKPOINTS=(
    "sam_vit_b_01ec64.pth|https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth|ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"
    "R50_DeAOTL_PRE_YTB_DAV.pth|gdrive:1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ|7e8a8d83310739bac02817f6bf48b6bbe2bbd7d5325722f1084088eb3aee1e06"
    "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth|gdrive:18doehnHWWnz9zBtOdgYZ3XMTpgPYbYZ6|73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c"
)

# ── Colours ──────────────────────────────────────────────────────────────────
if [[ -t 1 ]] && [[ "${TERM:-}" != "dumb" ]]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; NC=''
fi

info()  { printf "${BLUE}[..]${NC} %s\n" "$*"; }
ok()    { printf "${GREEN}[OK]${NC} %s\n" "$*"; }
warn()  { printf "${YELLOW}[!!]${NC} %s\n" "$*" >&2; }
err()   { printf "${RED}[XX]${NC} %s\n" "$*" >&2; }
die()   { err "$*"; echo "CASTLE_INSTALL_RESULT=FAIL"; exit 1; }

usage() {
    cat <<'USAGE'
Usage: install.sh [--install-dir DIR] [--cpu-only] [--no-checkpoints] [--uninstall]
  --install-dir DIR   Install folder (default: ~/castle; env CASTLE_INSTALL_DIR)
  --cpu-only          Force CPU-only PyTorch
  --no-checkpoints    Skip model checkpoint download
  --uninstall         Remove CASTLE from the install folder (keeps projects/)
USAGE
}

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-dir)    INSTALL_DIR="$2"; EXPLICIT_DIR=true; shift 2 ;;
        --cpu-only)       CPU_ONLY=true; shift ;;
        --no-checkpoints) NO_CHECKPOINTS=true; shift ;;
        --uninstall)      UNINSTALL=true; shift ;;
        -h|--help)        usage; exit 0 ;;
        *) die "Unknown option: $1 (see --help)" ;;
    esac
done
[[ -z "$INSTALL_DIR" ]] && INSTALL_DIR="$HOME/castle"
VENV="$INSTALL_DIR/.venv"
PY="$VENV/bin/python"
CKPT="$INSTALL_DIR/ckpt"

OS="$(uname -s)"
case "$OS" in
    Linux|Darwin) ;;
    *) die "Unsupported OS: $OS (use install.ps1 on Windows)" ;;
esac

# ── Uninstall (keeps projects/) ──────────────────────────────────────────────
if $UNINSTALL; then
    if [[ ! -d "$INSTALL_DIR" ]]; then ok "Nothing to remove at $INSTALL_DIR"; exit 0; fi
    find "$INSTALL_DIR" -mindepth 1 -maxdepth 1 ! -name projects -exec rm -rf {} +
    ok "CASTLE removed from $INSTALL_DIR (the projects folder, if any, was kept)."
    exit 0
fi

printf "\n  ${CYAN}CASTLE installer (%s)${NC}\n  Install folder: %s\n\n" "$OS" "$INSTALL_DIR"

# ── 1. Install folder must be decided by the user if it is non-ASCII ─────────
if ! $EXPLICIT_DIR && LC_ALL=C grep -q '[^ -~]' <<<"$INSTALL_DIR"; then
    warn "The install folder contains non-English characters:"
    warn "    $INSTALL_DIR"
    warn "Some video tools can fail on such paths. Choose one and run the installer again:"
    warn "  a) an English-only folder (recommended):  ... | bash -s -- --install-dir /opt/castle"
    warn "  b) keep this folder anyway:               ... | bash -s -- --install-dir \"$INSTALL_DIR\""
    echo "CASTLE_INSTALL_RESULT=STOPPED"
    exit 2
fi

# ── 2. uv (brings its own Python; no system Python needed) ───────────────────
install_uv() {
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    if command -v uv &>/dev/null; then ok "uv found: $(uv --version)"; return; fi
    info "Installing uv ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh || die "uv installation failed"
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv &>/dev/null || die "uv installation failed — 'uv' not found on PATH"
    ok "uv installed: $(uv --version)"
}

# ── 3. Source code (dev branch tarball; no git needed) ───────────────────────
get_source() {
    info "Downloading CASTLE (dev branch) ..."
    local tmp; tmp="$(mktemp -d)"
    curl -fsSL "$REPO_TARBALL" | tar -xz -C "$tmp" || { rm -rf "$tmp"; die "Could not download or unpack CASTLE"; }
    mkdir -p "$INSTALL_DIR"
    # Overwrite code in place; projects/ and downloaded ckpt files are not in the tarball.
    cp -R "$tmp"/*/. "$INSTALL_DIR"/
    rm -rf "$tmp"
    ok "Source code in $INSTALL_DIR"
}

# ── 4. PyTorch backend ───────────────────────────────────────────────────────
detect_backend() {
    if [[ "$OS" == "Darwin" ]]; then echo "macos"; return; fi
    if $CPU_ONLY; then echo "cpu"; return; fi
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then echo "cuda"; else echo "cpu"; fi
}

# ── 5. Python environment ────────────────────────────────────────────────────
install_env() {
    local backend="$1"
    if [[ ! -x "$PY" ]]; then
        info "Creating Python $PYTHON_VERSION environment ..."
        uv venv "$VENV" --python "$PYTHON_VERSION" || die "uv venv failed"
    fi
    info "Installing PyTorch ($backend) ..."
    case "$backend" in
        cpu)   uv pip install --python "$PY" torch torchvision --index-url https://download.pytorch.org/whl/cpu ;;
        *)     uv pip install --python "$PY" torch torchvision ;;  # PyPI: CUDA on Linux, MPS on macOS
    esac || die "PyTorch install failed"
    info "Installing CASTLE ..."
    if [[ "$backend" == "cuda" ]]; then
        # Linux + NVIDIA: add the GPU accelerators (cuML, xformers).
        uv pip install --python "$PY" -e "$INSTALL_DIR[gpu]" \
            --extra-index-url https://pypi.nvidia.com || die "CASTLE install failed"
    else
        uv pip install --python "$PY" -e "$INSTALL_DIR" || die "CASTLE install failed"
    fi
    ok "Python environment ready"
}

# ── 6. Checkpoints (verified by SHA-256) ─────────────────────────────────────
sha256_of() {
    if command -v sha256sum &>/dev/null; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

get_checkpoints() {
    mkdir -p "$CKPT"
    local failed=() entry file src sha dest
    for entry in "${CHECKPOINTS[@]}"; do
        IFS='|' read -r file src sha <<<"$entry"
        dest="$CKPT/$file"
        if [[ -f "$dest" && "$(sha256_of "$dest")" == "$sha" ]]; then ok "Already present: $file"; continue; fi
        info "Downloading $file ..."
        if [[ "$src" == gdrive:* ]]; then
            "$PY" -m gdown "${src#gdrive:}" -O "$dest" || true
        else
            curl -fL --progress-bar -o "$dest" "$src" || true
        fi
        if [[ -f "$dest" && "$(sha256_of "$dest")" == "$sha" ]]; then
            ok "Downloaded $file"
        else
            rm -f "$dest"
            failed+=("$entry")
        fi
    done
    if [[ ${#failed[@]} -gt 0 ]]; then
        err "These model files could not be downloaded (Google Drive may be rate-limiting):"
        for entry in "${failed[@]}"; do
            IFS='|' read -r file src sha <<<"$entry"
            [[ "$src" == gdrive:* ]] && src="https://drive.google.com/file/d/${src#gdrive:}/view"
            err "    $file   <-  $src"
        done
        err "Download them in a browser, put them in $CKPT, then run the installer again."
        die "Checkpoints missing"
    fi
}

# ── 7. Self-check ────────────────────────────────────────────────────────────
self_check() {
    "$PY" -c "import torch, gradio, av, cv2, castle.core.models, castle.service.clip_service; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" \
        || die "Import check failed"
}

# ── Run ──────────────────────────────────────────────────────────────────────
install_uv
get_source
BACKEND="$(detect_backend)"
info "PyTorch backend: $BACKEND"
install_env "$BACKEND"
if $NO_CHECKPOINTS; then warn "Skipping checkpoint download (--no-checkpoints)"; else get_checkpoints; fi
self_check

printf "\n"
ok "CASTLE is installed in $INSTALL_DIR"
printf "  Start CASTLE (keep the terminal open while you use it):\n"
printf "      ${CYAN}cd \"%s\" && ./.venv/bin/python app.py${NC}\n" "$INSTALL_DIR"
printf "  Then open http://127.0.0.1:7860 in your browser.\n\n"
echo "CASTLE_INSTALL_RESULT=OK"
exit 0
