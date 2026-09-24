# Installing CASTLE for a user (instructions for AI agents)

You are installing CASTLE on the user's computer. The user may not be a
programmer: run the steps yourself, and only ask them the questions listed here.
The human-facing version of these steps is [INSTALLATION.md](INSTALLATION.md).

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

It is ready when the output contains `Running on local URL:`. Tell the user to
open **http://127.0.0.1:7860** in their browser (not the `0.0.0.0` address that
is printed), and that closing the terminal stops CASTLE.

## 4. When the installer reports FAIL

| Message (`[XX]` line) | Cause | What you do |
|---|---|---|
| `These model files could not be downloaded` | Google Drive rate limit, or no access to Drive | Wait a few minutes and re-run the same install command (finished files are kept). If it keeps failing, give the user the listed links, ask them to download the files in a browser into `DIR/ckpt`, then re-run. |
| `uv installation failed` / `Could not download or unpack CASTLE` | No access to astral.sh or GitHub | Check the network; ask the user whether a proxy or firewall is in the way. |
| `PyTorch install failed` / `CASTLE install failed` | Package download or disk space | Check free disk space (about 5 GB, 10 GB with an NVIDIA GPU) and re-run. |
| `Import check failed` | Broken environment | Delete `DIR/.venv` and re-run the installer. |

Re-running the installer is always safe: it updates the code and keeps
`DIR/projects` and verified files in `DIR/ckpt`.
