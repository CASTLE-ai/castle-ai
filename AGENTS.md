# CASTLE for AI agents

Instructions for an AI assistant helping a user with CASTLE. The user may not be
a programmer: do the work yourself and only ask them the questions listed here.

- Installing: [Install CASTLE](#install-castle) (human version:
  [INSTALLATION.md](INSTALLATION.md)).
- Using it and tuning settings: [Find how a feature works](#find-how-a-feature-works).

# Install CASTLE

## Rules

- Install **only** with the installer scripts below. They install the `dev`
  branch, which is the current version.
- Do **not** use `pip install castle-ai` (PyPI is outdated), do not `git clone`
  the default branch (`main` is outdated), and do not `pip install -r
  requirements.txt` by hand.
- Do not pass `--cpu-only` / `CASTLE_CPU_ONLY` unless the user asks; the installer
  detects the GPU itself.

## 1. Run the installer

Windows (run from PowerShell or cmd):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/CASTLE-ai/castle-ai/dev/install.ps1 | iex"
```

macOS / Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/CASTLE-ai/castle-ai/dev/install.sh | bash
```

The installer is non-interactive and can run for a long time (it downloads
PyTorch and ~1 GB of model files). Do not interrupt it.

## 2. Check the result

The last line of output and the exit code tell you the outcome:

| Last line | Exit code | Meaning | What you do |
|---|---|---|---|
| `CASTLE_INSTALL_RESULT=OK` | 0 | Installed | Go to step 3. |
| `CASTLE_INSTALL_RESULT=STOPPED` | 2 | The default install folder contains non-ASCII characters (e.g. a Chinese user name). Nothing was installed. | Ask the user: install into an English-only folder (recommended, e.g. `C:\castle`), or keep the suggested folder? Then re-run with that folder (see below). |
| `CASTLE_INSTALL_RESULT=FAIL` | 1 | Something failed; the `[XX]` lines above say what. | See the table in step 4. |

Re-run with a chosen folder:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "$env:CASTLE_INSTALL_DIR='C:\castle'; irm https://raw.githubusercontent.com/CASTLE-ai/castle-ai/dev/install.ps1 | iex"
```

```bash
curl -fsSL https://raw.githubusercontent.com/CASTLE-ai/castle-ai/dev/install.sh | bash -s -- --install-dir "$HOME/castle"
```

## 3. Verify and start

Let `DIR` be the install folder (default `%USERPROFILE%\castle` on Windows,
`~/castle` on macOS/Linux; or the folder you passed).

Verify the environment (must print a `torch` version line and exit 0):

```bash
DIR/.venv/Scripts/python.exe -c "import castle.core.models, gradio; import torch; print('torch', torch.__version__)"   # Windows
DIR/.venv/bin/python        -c "import castle.core.models, gradio; import torch; print('torch', torch.__version__)"   # macOS/Linux
```

Start CASTLE from `DIR` (it is a long-running web server; start it in the
background or a separate terminal, and leave it running):

```bash
cd DIR && .venv/Scripts/python.exe app.py   # Windows
cd DIR && ./.venv/bin/python app.py         # macOS/Linux
```

It is ready when the output contains `CASTLE is running. Open
http://127.0.0.1:7860`. Tell the user to open that address in their browser (not
the `0.0.0.0` address Gradio also prints, and not a `gradio.live` share link),
and that closing the terminal stops CASTLE.

## 4. When the installer reports FAIL

| Message (`[XX]` line) | Cause | What you do |
|---|---|---|
| `These model files could not be downloaded` | Google Drive rate limit, or no access to Drive | Wait a few minutes and re-run the same install command (finished files are kept). If it keeps failing, give the user the listed links, ask them to download the files in a browser into `DIR/ckpt`, then re-run. |
| `uv installation failed` / `Could not download or unpack CASTLE` | No access to astral.sh or GitHub | Check the network; ask the user whether a proxy or firewall is in the way. |
| `PyTorch install failed` / `CASTLE install failed` | Package download or disk space | Check free disk space (about 5 GB, 10 GB with an NVIDIA GPU) and re-run. |
| `Import check failed` | Broken environment | Delete `DIR/.venv` and re-run the installer. |

Re-running the installer is always safe: it updates the code and keeps
`DIR/projects` and verified files in `DIR/ckpt`.

# Find how a feature works

The code is the source of truth. The guides in `docs/` explain the workflow and
the reasoning behind settings, but they were last revised in the first half of
2026 and can lag the code: when a guide and the code disagree, trust the code
and say so to the user. Skip `castle/aot/` and `castle/sam/` (vendored
third-party models).

## Where each UI tab lives

The web UI (`app.py` → `castle/ui/main_ui.py`) is a row of tabs. A tab's
`castle/ui/*_ui.py` file defines its widgets and their default values; the
logic it calls is in `castle/service/`, `castle/core/` and `castle/utils/`.

| Tab | UI code | Logic | Guide |
|---|---|---|---|
| 0. Project | `castle/ui/project_ui.py` | `castle/service/project_service.py`, `castle/utils/project_manager.py` | `docs/tutorials/step1-project.md` |
| 1. Upload Videos | `castle/ui/source_ui.py` | `castle/utils/video_manager.py` | `docs/tutorials/step1-project.md` |
| 2. Tracking ROIs | `castle/ui/edit_ui.py` (sub-tabs: `label_ui.py`, `knowledge_ui.py` = ROI Prompts, `track_ui.py`, `batch_track_ui.py`) | `castle/service/tracking_service.py`, `castle/utils/image_segment.py` (SAM), `castle/utils/video_object_segment.py` (DeAOT) | `docs/tutorials/step2-tracking.md` |
| 3. Pre-process (optional) | `castle/ui/preprocess_ui.py` | `castle/service/preprocessing_service.py`, `castle/core/stabilized_camera.py`, `castle/core/preprocess_session.py` | `docs/tutorials/step2_5-preprocessing.md` |
| 4. Extract Latent | `castle/ui/extract_ui.py` | `castle/service/extraction_service.py`, `castle/core/extractor.py`, `castle/core/models.py` (DINOv2/v3) | `docs/tutorials/step3-extract.md` |
| 5. Behavior Microscope → Clustering (Prepare, Explore) | `castle/ui/cluster_page_ui.py` (UMAP presets at the top), `castle/ui/cluster_handlers.py` | `castle/service/prepare_service.py`, `castle/service/clustering_service.py`, `castle/service/cluster_params.py`, `castle/core/cluster.py`, `castle/core/clustering_backends.py` | `docs/tutorials/step4-analysis.md` |
| 5. Behavior Microscope → Cluster Annotator | `castle/ui/annotator_ui.py` | `castle/service/annotator_loader.py`, `castle/service/bout_service.py`, `castle/service/annotation_service.py` | `docs/tutorials/step4-analysis.md` |
| 6. Analysis | `castle/ui/analysis_ui.py` | `castle/service/ethogram_service.py`, `castle/service/comparison_service.py` | `docs/tutorials/step5-export.md` |
| 7. Export | `castle/ui/export_ui.py` | `castle/service/export_service.py` | `docs/tutorials/step5-export.md` |

## Where settings and their defaults live

| What | Where |
|---|---|
| Defaults shared by several steps (batch sizes, bin size, UMAP / DBSCAN defaults), each with the reason for its value | `castle/defaults.py` |
| A widget's default in the UI | that tab's `castle/ui/*_ui.py` |
| Per-project saved settings | `castle/core/project_config.py`; the project's `config.json` |
| Model checkpoints and model settings | `castle/core/config.py`, `castle/configs/model_config.json` |
| Environment variables (workers, encoders, seeds, …) | `docs/technical/environment-variables.md` |
| All configuration layers at a glance | `docs/technical/configuration.md` |
| Command-line equivalents of the UI steps | `castle --help`; `castle/cli/` |

## Background and troubleshooting

- What the pipeline does and why: `docs/getting-started/concepts.md`,
  `docs/technical/algorithm.md`.
- Output files and their formats: `docs/technical/data-formats.md`.
- Reproducibility (seeds, pinned models): `docs/technical/reproducibility.md`.
- Problems: the troubleshooting table in [INSTALLATION.md](INSTALLATION.md#troubleshooting),
  `docs/faq.md`, and the terminal window running `app.py` (every error is logged
  there).
