# Contributing to CASTLE Documentation

## Building Docs Locally

```bash
pip install -r docs/requirements.txt
mkdocs serve
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to preview.

## Adding a Page

1. Create a `.md` file in the appropriate `docs/` subdirectory
2. Add an entry to the `nav:` section in `mkdocs.yml`
3. Run `mkdocs build` to verify no errors

## Writing Style

- **Tone**: Friendly but professional — like explaining to a colleague
- **Code blocks**: Always specify language. Always copy-pasteable.
- **OS differences**: Use tabbed blocks for Linux/macOS/Windows when commands differ
- **Screenshots**: Use `![Description](../assets/screenshots/filename.png)` — actual images provided by maintainers
- **Admonitions**: Use `!!! tip`, `!!! warning`, `!!! note` for callouts
- **Tables**: Use markdown tables for structured data

## API Reference

The `docs/reference/api.md` page uses [mkdocstrings](https://mkdocstrings.github.io/) to auto-generate from Python docstrings. To add a new module:

```markdown
::: castle.module.name
    options:
      show_root_heading: true
      show_source: true
```

Note: Some modules may fail to build if they have heavy import dependencies (torch, cuml, etc.) not available in the docs build environment.
