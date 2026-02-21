#!/usr/bin/env bash
# =============================================================================
# CASTLE AI — Docker Entrypoint
# =============================================================================
# Responsibilities:
#   1. Detect GPU availability and log device info
#   2. Download any missing model checkpoints to /models/
#   3. Apply sane runtime defaults
#   4. Launch the Gradio server on 0.0.0.0:7860
# =============================================================================
set -euo pipefail

# ── Colours (disabled if not a TTY) ─────────────────────────────────────────
if [[ -t 1 ]]; then
    GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'
    BLUE='\033[0;34m'; RED='\033[0;31m'; NC='\033[0m'
else
    GREEN=''; YELLOW=''; CYAN=''; BLUE=''; RED=''; NC=''
fi

info()  { printf "${BLUE}[castle]${NC} %s\n"           "$*"; }
ok()    { printf "${GREEN}[castle] ✓${NC} %s\n"        "$*"; }
warn()  { printf "${YELLOW}[castle] ⚠${NC} %s\n" "$*" >&2; }

# ── Banner ───────────────────────────────────────────────────────────────────
printf "${CYAN}"
cat <<'EOF'
  ╔══════════════════════════════════════╗
  ║         CASTLE AI  🏰                ║
  ║   Behavior Analysis from Video       ║
  ╚══════════════════════════════════════╝
EOF
printf "${NC}"

# ── 1. GPU detection ─────────────────────────────────────────────────────────
detect_device() {
    if [[ "${CASTLE_DEVICE:-auto}" == "cpu" ]]; then
        echo "cpu"; return
    fi

    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
        local gpu_name cuda_ver
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "unknown GPU")
        cuda_ver=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9.]+' || echo "unknown")
        ok "GPU detected: ${gpu_name} (CUDA ${cuda_ver})"
        echo "cuda"
    else
        warn "No GPU detected — running in CPU mode (slower but functional)"
        echo "cpu"
    fi
}

DETECTED_DEVICE=$(detect_device)
export CASTLE_DEVICE="${CASTLE_DEVICE:-${DETECTED_DEVICE}}"
info "Compute device: ${CASTLE_DEVICE}"

# ── 2. Checkpoint download ───────────────────────────────────────────────────
CKPT_DIR="${CASTLE_CKPT_DIR:-/models}"
mkdir -p "${CKPT_DIR}"

SAM_URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
SAM_FILE="${CKPT_DIR}/sam_vit_b_01ec64.pth"

DEAOT_GDRIVE_ID="1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ"
DEAOT_FILE="${CKPT_DIR}/R50_DeAOTL_PRE_YTB_DAV.pth"

DINOV2_URL="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth"
DINOV2_FILE="${CKPT_DIR}/dinov2_vitb14_reg4_pretrain.pth"

download_file() {
    local url="$1" dest="$2" name
    name="$(basename "${dest}")"
    if [[ -f "${dest}" ]]; then
        ok "Checkpoint already present: ${name}"
        return 0
    fi
    info "Downloading ${name} …"
    if wget -q --show-progress -O "${dest}.tmp" "${url}"; then
        mv "${dest}.tmp" "${dest}"
        ok "Downloaded: ${name}"
    else
        warn "Failed to download ${name} — the app may error when using this model"
        rm -f "${dest}.tmp"
    fi
}

download_gdrive() {
    local id="$1" dest="$2" name
    name="$(basename "${dest}")"
    if [[ -f "${dest}" ]]; then
        ok "Checkpoint already present: ${name}"
        return 0
    fi
    info "Downloading ${name} from Google Drive …"
    if python -m gdown "${id}" -O "${dest}" 2>/dev/null; then
        ok "Downloaded: ${name}"
    else
        # Fallback: direct export URL
        local fallback="https://drive.google.com/uc?export=download&id=${id}&confirm=t"
        warn "gdown failed — trying direct URL …"
        download_file "${fallback}" "${dest}"
    fi
}

info "Checking model checkpoints in ${CKPT_DIR} …"
download_file   "${SAM_URL}"    "${SAM_FILE}"
download_gdrive "${DEAOT_GDRIVE_ID}" "${DEAOT_FILE}"
download_file   "${DINOV2_URL}" "${DINOV2_FILE}"

# ── 3. Runtime defaults ───────────────────────────────────────────────────────
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-7860}"
export CASTLE_DATA="${CASTLE_DATA:-/data}"

mkdir -p "${CASTLE_DATA}"

info "Data directory : ${CASTLE_DATA}"
info "Server         : ${GRADIO_SERVER_NAME}:${GRADIO_SERVER_PORT}"

# ── 4. Launch Gradio app ─────────────────────────────────────────────────────
info "Starting CASTLE Gradio UI …"
printf "\n"

exec python /app/app.py \
    --root "${CASTLE_DATA}" \
    "$@"
