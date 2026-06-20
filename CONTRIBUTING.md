# Contributing to CASTLE

Thanks for helping improve CASTLE! This guide covers contributing **code** and
**documentation**.

## Contributing Code

### Development setup

CASTLE targets **Python ≥ 3.10**.

```bash
git clone https://github.com/CASTLE-ai/castle-ai.git
cd castle-ai
python -m venv .venv && source .venv/bin/activate

# CPU-only dev environment (clustering falls back to sklearn/umap-learn):
pip install -e ".[dev]"

# GPU acceleration (RAPIDS cuML UMAP/DBSCAN, xformers) — needs the NVIDIA index:
pip install -e ".[gpu]" --extra-index-url https://pypi.nvidia.com

# Optional features:
pip install -e ".[nwb]"      # `castle ethogram export-nwb`
pip install -e ".[hdbscan]"  # HDBSCAN clustering backend
```

Print your exact runtime stack (paste into bug reports) with:

```bash
castle env
```

### Running tests

```bash
# Fast lane — what CI runs (no GPU/model weights needed):
pytest -m "not integration"

# Full suite, incl. GPU/model integration tests (needs a CUDA GPU + checkpoints):
pytest
```

Integration tests are auto-marked by their location under `tests/integration/`
(a `conftest.py` hook), so the `not integration` lane stays GPU-free.

### Linting & type checks

CI enforces the project's ruff rule set on non-vendored code (includes **F821**,
which catches undefined names / missing imports):

```bash
ruff check castle/ --exclude thirdparty,aot,sam,dinov2 \
    --select E722,T201,B006,F401,F841,F821
mypy castle    # tiered strictness — see mypy.ini
```

### CI

`.github/workflows/ci.yml` runs ruff + an import smoke-test + the non-integration
test suite on Python 3.10 and 3.12 for every push/PR. Please keep it green.

### Pull-request checklist

- [ ] `pytest -m "not integration"` passes
- [ ] `ruff check …` (the command above) reports no findings
- [ ] New behavior has a test; bug fixes have a regression test
- [ ] Changes that alter **scientific output** (clustering, extraction, figures,
      exported artifacts) note how they were validated on real data — these
      cannot be confirmed by the CPU unit suite alone
- [ ] Public functions have docstrings

---

## Contributing Documentation

### Building docs locally

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to preview.

### Adding a page

1. Create a `.md` file in the appropriate `docs/` subdirectory
2. Add an entry to the `nav:` section in `mkdocs.yml`
3. Run `mkdocs build` to verify no errors

### Writing style

- **Tone**: Friendly but professional — like explaining to a colleague
- **Code blocks**: Always specify language. Always copy-pasteable.
- **OS differences**: Use tabbed blocks for Linux/macOS/Windows when commands differ
- **Screenshots**: Use `![Description](../assets/screenshots/filename.png)` — actual images provided by maintainers
- **Admonitions**: Use `!!! tip`, `!!! warning`, `!!! note` for callouts
- **Tables**: Use markdown tables for structured data

### API reference

The `docs/reference/api.md` page uses [mkdocstrings](https://mkdocstrings.github.io/) to auto-generate from Python docstrings. To add a new module:

```markdown
::: castle.module.name
    options:
      show_root_heading: true
      show_source: true
```

Note: Some modules may fail to build if they have heavy import dependencies (torch, cuml, etc.) not available in the docs build environment.
