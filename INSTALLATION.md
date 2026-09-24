# Installing CASTLE

One command installs the current CASTLE (the `dev` branch) into a single folder:
the source code, its own Python 3.10 environment (`.venv`), and the three model
checkpoints (`ckpt`, about 1 GB). The installer detects your operating system and
GPU and chooses the matching PyTorch build — there is nothing to select.

Installing through an AI assistant (Claude Code, Codex, …)? Give it
[AGENTS.md](AGENTS.md).

Requirements: internet access, about 5 GB of free disk space (about 10 GB with an NVIDIA GPU), and 16 GB of RAM
recommended ([details](docs/getting-started/gpu-requirements.md)). An NVIDIA GPU is
optional; without one CASTLE runs on the CPU, only more slowly.

## Step 1 — Install

**Windows** — open *PowerShell* (Start menu → type `PowerShell`) and paste:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/CASTLE-ai/castle-ai/dev/install.ps1 | iex"
```

**macOS / Linux** — open *Terminal* and paste:

```bash
curl -fsSL https://raw.githubusercontent.com/CASTLE-ai/castle-ai/dev/install.sh | bash
```

**It worked if** the last line is:

```text
CASTLE_INSTALL_RESULT=OK
```

The default install folder is `castle` in your home folder
(`C:\Users\<you>\castle` on Windows, `~/castle` on macOS/Linux). If the last line
says `STOPPED` or `FAIL`, see [Troubleshooting](#troubleshooting).

## Step 2 — Start CASTLE

**Windows** (PowerShell):

```powershell
cd $env:USERPROFILE\castle
.\.venv\Scripts\python.exe app.py
```

**macOS / Linux**:

```bash
cd ~/castle && ./.venv/bin/python app.py
```

**It worked if** the window shows `Running on local URL:`. Then open **http://127.0.0.1:7860** in your browser. Keep the window open while
you use CASTLE; closing it stops CASTLE.

If you installed into another folder, `cd` into that folder instead.

## Installer options

Set these before the install command (Windows: in the same PowerShell window;
macOS/Linux: append after `bash -s --`).

| What | Windows | macOS / Linux |
|---|---|---|
| Install into another folder | `$env:CASTLE_INSTALL_DIR = "C:\castle"` | `... \| bash -s -- --install-dir /path/to/castle` |
| Force CPU (ignore the GPU) | `$env:CASTLE_CPU_ONLY = "1"` | `... \| bash -s -- --cpu-only` |
| Skip the model download | `$env:CASTLE_NO_CHECKPOINTS = "1"` | `... \| bash -s -- --no-checkpoints` |
| Uninstall (keeps `projects`) | `$env:CASTLE_UNINSTALL = "1"` | `... \| bash -s -- --uninstall` |

Running the installer again updates CASTLE to the latest `dev` version. Your
`projects` folder and the downloaded checkpoints are kept.

## Troubleshooting

| Symptom | What to do |
|---|---|
| Last line `CASTLE_INSTALL_RESULT=STOPPED`, message about non-English characters | Your home folder path contains e.g. Chinese characters, which some video tools cannot handle. Pick an English-only folder (recommended) and run again, e.g. Windows: `$env:CASTLE_INSTALL_DIR = "C:\castle"`, then the install command. To keep the suggested folder anyway, set it the same way to that folder. |
| `These model files could not be downloaded` | Google Drive limits how often a file is downloaded. Open the listed links in a browser, save the files into the `ckpt` folder inside the install folder, and run the installer again (it checks the files and continues). |
| `uv installation failed` or `Could not download` | Check the internet connection (some school or company networks block GitHub or astral.sh) and run the command again. |
| The browser cannot open `http://0.0.0.0:7860` | Use **http://127.0.0.1:7860** instead. |
| Windows Firewall asks about Python when CASTLE starts | Either answer works; CASTLE on your own computer is not affected. |
| `Unsupported video: variable frame rate (VFR)` when adding a video | CASTLE only supports constant-frame-rate video. Re-export the video with a constant frame rate, then add it again. |
| A project or behavior name is refused | Names cannot contain `\ / : * ? " < > \|`, end with a period or space, or be a Windows device name such as `CON`. |
| Tracking or extraction is slow | Expected without an NVIDIA GPU. As a rough guide, a 6-second 500×500 clip took about 7 minutes to track and 12 minutes to extract on a 4-core CPU. |

### Clustering / UMAP is slow on a large prepared cache (NVIDIA GPU)

On a large prepared cache (~1M datapoints) the first UMAP can take many minutes.
This is almost always **scale, not a CPU fallback** — GPU UMAP is running, it just
has a lot of work to do.

**The common case works on GPU with no setup.** The Behavior Microscope uses GPU
UMAP/DBSCAN via **cuML (RAPIDS)** when you pick the GPU backend. CASTLE imports
PyTorch early, and PyTorch preloads its bundled CUDA 12 libraries (e.g.
`libcublas.so.12`) globally — which also satisfies cuML's dependency on the same
libraries. So on a typical `*-cu12` pip install that has both torch and cuML,
**GPU UMAP loads and runs without any `LD_LIBRARY_PATH` change.** Confirm with
`nvidia-smi`: during a UMAP run a python process will be pinning one GPU.

**If it's slow, lower the UMAP work — this is the real lever:**
- `n_epochs`: the presets hardcode **5000**; umap-learn's own large-data default is
  **200**. Dropping to 200–500 is the single biggest speedup (~10×) and barely
  changes the embedding on big data.
- `n_neighbors`: 100 → 30–50.
- Explore `k'`: use fewer PCA dimensions.

### Rare case: cuML genuinely can't import → CPU fallback

This is **not** expected on a standard torch + cuML cu12 install (see above). But if
cuML truly fails to import — torch absent so nothing preloads the CUDA libs,
mismatched versions, or an older system CUDA shadowing the wheels — CASTLE **falls
back to CPU `umap-learn`** and logs a warning (terminal + a UI banner). On a
~1M-point cache a CPU UMAP can take **hours**, which looks like the app hung.

**Symptom.** The log says `GPU UMAP (cuML) unavailable (...); falling back to CPU
umap-learn`; `nvidia-smi` shows the GPU idle during the run; a common underlying
error is `libcublas.so.12: cannot open shared object file`.

**Cause.** The CUDA libraries the `*-cu12` wheels need (e.g. `libcublas.so.12`) live
inside the venv at `site-packages/nvidia/<lib>/lib/`, which is **not** on the
dynamic-linker path. Normally PyTorch's global preload covers cuML too; this only
bites when that preload is absent, or `LD_LIBRARY_PATH` points at a *different* /
older system CUDA (e.g. `/usr/local/cuda` → CUDA 11) so cuML resolves the wrong
`libcublas` (or none) and fails to load.

**Fix — point the linker at the venv's own CUDA wheels (no sudo, no system change):**

```bash
# prepend the venv's nvidia-wheel lib dirs so the matched cu12 libs win
cd ~/castle
export LD_LIBRARY_PATH="$(find .venv/lib/python*/site-packages/nvidia -name lib -type d | tr '\n' ':')$LD_LIBRARY_PATH"
./.venv/bin/python app.py
```

Prefer this over deleting/upgrading the system CUDA: the venv already ships a
complete, version-matched CUDA 12 stack, so it doesn't need the system toolkit
at all, and touching `/usr/local/cuda` can break other projects pinned to it.

## For developers

To work on CASTLE itself (editable install, tests, linting), see
[CONTRIBUTING.md](CONTRIBUTING.md).
