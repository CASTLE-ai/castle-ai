# UI Handler Pattern Guide

> "Special cases aren't special enough to break the rules." — *The Zen of Python*

This document describes the **target pattern** for CASTLE Gradio UI handlers.
Existing handlers are not refactored yet; this guide shows where we are going.

---

## The Ideal: Thin Handler, Fat Service

A handler should be a *thin wrapper* around one service call.  It translates
Gradio inputs → service call → Gradio outputs.  Business logic lives in the
service layer, never inside the handler.

```python
# ✅ IDEAL — handler is a thin wrapper around a service call
def on_click_cluster(project, n_clusters, method, threshold):
    try:
        result = cluster_service.cluster(
            project,
            n_clusters=n_clusters,
            method=method,
            threshold=threshold,
        )
        return result.plot, result.summary, gr.update(visible=True)
    except ValueError as e:
        raise gr.Error(str(e))
```

The handler does **three things only**:

1. **Call** the service with forwarded arguments.
2. **Unpack** the result into Gradio output values.
3. **Convert** domain exceptions into `gr.Error` so Gradio shows a toast.

---

## Anti-patterns to Avoid

### ❌ Business logic inside the handler

```python
# BAD — handler is doing the work of a service
def on_click_cluster(project, n_clusters):
    features = np.load(project.latent_path)
    reducer = UMAP(n_components=2)
    embedding = reducer.fit_transform(features)
    labels = DBSCAN(eps=0.5).fit_predict(embedding)
    fig = plot_clusters(embedding, labels)
    return fig, f"Found {labels.max()+1} clusters", gr.update(visible=True)
```

**Why it hurts:** can't unit-test the algorithm without a Gradio context;
impossible to reuse from CLI or batch scripts.

---

### ❌ Catching every exception silently

```python
# BAD — swallows errors, user sees nothing
def on_click_cluster(project, n_clusters):
    try:
        result = cluster_service.cluster(project, n_clusters)
        return result.plot, result.summary, gr.update(visible=True)
    except Exception:
        return None, "Error occurred", gr.update(visible=False)
```

**Why it hurts:** debugging becomes guesswork; users get no actionable message.

---

### ❌ Mixed concerns / God handler

```python
# BAD — one handler does loading, clustering, plotting, and state mutation
def on_click_cluster(project, n_clusters, video_path, roi_id, ...):
    video = VideoReader(video_path)
    # ... 80 more lines ...
```

---

## Correct Error Handling

| Exception type | Action |
|---|---|
| `ValueError` (bad user input) | `raise gr.Error(str(e))` |
| `FileNotFoundError` | `raise gr.Error(f"File not found: {e}")` |
| Unexpected / bug | log it, then `raise gr.Error("Unexpected error — check logs")` |

```python
import logging
import gradio as gr

logger = logging.getLogger(__name__)

def on_click_cluster(project, n_clusters, method):
    try:
        result = cluster_service.cluster(project, n_clusters=n_clusters, method=method)
        return result.plot, result.summary, gr.update(visible=True)
    except ValueError as e:
        raise gr.Error(str(e))
    except FileNotFoundError as e:
        raise gr.Error(f"File not found: {e}")
    except Exception as e:
        logger.exception("Unexpected error in on_click_cluster")
        raise gr.Error(f"Unexpected error: {e}")
```

---

## Service Layer Contract

The service function should:

* Accept **plain Python types** — no Gradio types inside a service.
* Return a **dataclass or namedtuple** with named fields (not a raw tuple).
* Raise **`ValueError`** for invalid user inputs.
* Raise **`FileNotFoundError`** for missing paths.
* Never raise `gr.Error` — that's the handler's job.

```python
from dataclasses import dataclass
import matplotlib.pyplot as plt

@dataclass
class ClusterResult:
    plot: plt.Figure
    summary: str
    labels: list[int]

def cluster(project, *, n_clusters: int, method: str = "dbscan") -> ClusterResult:
    if n_clusters < 2:
        raise ValueError(f"n_clusters must be >= 2, got {n_clusters}")
    # ... algorithm ...
    return ClusterResult(plot=fig, summary=summary_text, labels=label_list)
```

---

## Summary Checklist

When writing or reviewing a handler, ask:

- [ ] Is the handler ≤ ~15 lines?
- [ ] Does it contain zero algorithmic logic?
- [ ] Does it call exactly one service function?
- [ ] Does it convert domain exceptions into `gr.Error`?
- [ ] Can the service be tested without importing `gradio`?

If all five are ✅ — the handler is correct.
