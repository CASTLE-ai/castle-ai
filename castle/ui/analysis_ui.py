"""
castle/ui/analysis_ui.py
Analysis UI — Sub-tab inside "4. Behavior Microscope".

Sections:
  A. Ethogram  — transition matrix heatmap, bout duration stats, raster plot
  B. Quality Metrics — silhouette, temporal coherence, bout quality
  C. Group Comparison — placeholder (requires two sessions)

All data is loaded from disk via :func:`castle.service.annotator_loader.load_annotator_data`,
using the same session-selector pattern as ``annotator_ui.py``.
"""

import logging

import gradio as gr

from castle.service.annotator_loader import load_annotator_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _refresh_sessions(storage_path: str, project_name: str):
    """Return updated session dropdown choices."""
    if not storage_path or not project_name:
        return gr.update(choices=[], value=None)

    from castle.service.session_manager import SessionManager

    mgr = SessionManager(storage_path, project_name)
    sessions = mgr.list_sessions()

    if not sessions:
        return gr.update(
            choices=[("(no sessions — run clustering first)", "")],
            value="",
        )

    choices = [
        (
            f"{s.name} — {s.n_clusters} clusters, bin_size={s.bin_size} "
            f"({s.updated_at[:16]})",
            s.session_id,
        )
        for s in sessions
    ]
    active_id = mgr.get_active_session_id()
    default = active_id or (sessions[0].session_id if sessions else None)
    return gr.update(choices=choices, value=default)


def _load_data(storage_path: str, project_name: str, session_id: str):
    """Load AnnotatorData and return it plus a status message."""
    if not storage_path or not project_name:
        return None, "**Status:** No project selected."

    sid = session_id or None
    try:
        data = load_annotator_data(storage_path, project_name, session_id=sid)
    except FileNotFoundError as exc:
        gr.Warning(
            "Analysis data not found. Please complete the clustering and annotation "
            "steps first, then reload."
        )
        return None, f"**Error:** {exc}"
    except Exception as exc:
        logger.exception("Failed to load annotator data")
        gr.Warning(
            "Failed to load analysis data. Please reload the session or check that "
            "clustering has been completed for this project."
        )
        return None, f"**Error:** {exc}"

    n_clusters = len(data.cluster_meta)
    n_bins = len(data.cluster)
    msg = (
        f"**Loaded:** {n_clusters} clusters, {n_bins} bins "
        f"(bin_size={data.bin_size}, fps={data.fps:.1f})"
    )
    gr.Info(f"Loaded {n_clusters} clusters from {project_name}")
    return data, msg


# ---------------------------------------------------------------------------
# Section A: Ethogram
# ---------------------------------------------------------------------------


def export_ethogram_csv_handler(storage_path: str, project_name: str, session_id: str):
    """Export ethogram data to CSV files and return as a ZIP download.

    Equivalent to `castle ethogram export` CLI.

    Returns:
        (status_markdown, gr.File update)
    """
    import os
    import shutil
    import tempfile
    import zipfile
    from datetime import datetime

    if not storage_path or not project_name:
        return "**❌ No project selected.**", gr.update(value=None, visible=False)

    from castle.service.ethogram_service import export_ethogram_csv

    project_path = os.path.join(storage_path, project_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tmp_dir = tempfile.mkdtemp(prefix="castle_ethogram_")
    csv_dir = os.path.join(tmp_dir, "ethogram_csv")

    try:
        export_ethogram_csv(project_path=project_path, output_path=csv_dir)
    except FileNotFoundError:
        return (
            "**❌ Export failed:** Ethogram data not found. "
            "Please generate the ethogram first by clicking '▶ Generate Ethogram'.",
            gr.update(value=None, visible=False),
        )
    except Exception as exc:
        logger.exception("Ethogram CSV export failed")
        return f"**❌ Export failed:** {exc}", gr.update(value=None, visible=False)

    # Zip the output directory
    zip_name = f"{project_name}_ethogram_{timestamp}.zip"
    zip_path = os.path.join(tmp_dir, zip_name)
    try:
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname in os.listdir(csv_dir):
                src = os.path.join(csv_dir, fname)
                if os.path.isfile(src):
                    staging = os.path.join(tmp_dir, f"_stage_{fname}")
                    shutil.copyfile(src, staging)
                    zf.write(staging, fname)
                    os.unlink(staging)
    except Exception as exc:
        logger.exception("Ethogram CSV zip failed")
        return f"**❌ Zip failed:** {exc}", gr.update(value=None, visible=False)

    return (
        f"**✅ Ethogram CSV exported!** {zip_name}",
        gr.update(value=zip_path, visible=True),
    )


def generate_ethogram(annotator_data):
    """Compute ethogram from AnnotatorData and return (heatmap fig, stats df, raster fig)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    if annotator_data is None:
        return None, None, None

    cluster_labels = annotator_data.cluster
    fps = annotator_data.fps or 30.0

    # Build cluster_names dict from cluster_meta
    cluster_names = {
        cid: meta["name"]
        for cid, meta in annotator_data.cluster_meta.items()
    }

    # Compute ethogram via service layer
    from castle.service.ethogram_service import compute_ethogram_from_data

    try:
        ethogram = compute_ethogram_from_data(cluster_labels, fps=fps, cluster_names=cluster_names)
    except Exception as exc:
        logger.exception("compute_ethogram failed")
        gr.Warning(
            "Ethogram computation failed. Make sure clusters are labeled before generating "
            f"the ethogram (Step 4b). Details: {exc}"
        )
        return None, None, None

    # --- A1: Transition matrix heatmap ---
    from castle.visualization.ethogram_plots import (
        plot_transition_heatmap,
        plot_ethogram_raster,
    )

    heatmap_fig = plot_transition_heatmap(ethogram.transition_matrix)

    # --- A2: Bout duration statistics dataframe ---
    rows = []
    for cid in sorted(ethogram.bout_stats.keys()):
        bs = ethogram.bout_stats[cid]
        rows.append(
            {
                "Cluster": bs.cluster_name,
                "N Bouts": bs.n_bouts,
                "Freq (%)": f"{bs.frequency * 100:.1f}",
                "Mean Dur (s)": f"{bs.mean_duration_s:.3f}",
                "Median Dur (s)": f"{bs.median_duration_s:.3f}",
                "Std Dur (s)": f"{bs.std_duration_s:.3f}",
                "CV": f"{bs.cv_duration:.3f}",
                "Mean IBI (s)": f"{bs.mean_inter_bout_interval_s:.3f}",
            }
        )
    stats_df = pd.DataFrame(rows)

    # --- A3: Ethogram raster ---
    raster_fig = plot_ethogram_raster(ethogram)

    plt.close("all")
    return heatmap_fig, stats_df, raster_fig


# ---------------------------------------------------------------------------
# Section B: Quality Metrics
# ---------------------------------------------------------------------------


def compute_quality_metrics(annotator_data):
    """Compute clustering quality metrics and return a dataframe."""
    import pandas as pd

    if annotator_data is None:
        return None

    cluster_labels = annotator_data.cluster
    embedding = annotator_data.embedding  # may be None

    from castle.core.metrics import evaluate_clustering

    try:
        report = evaluate_clustering(
            labels=cluster_labels,
            embedding=embedding if embedding is not None and len(embedding) > 0 else None,
            fps=annotator_data.fps or 30.0,
        )
    except Exception as exc:
        logger.exception("evaluate_clustering failed")
        gr.Warning(
            "Quality metrics computation failed. Embedding data may be missing — "
            f"try re-running UMAP to generate embeddings. Details: {exc}"
        )
        return None

    def _fmt(v):
        if v is None:
            return "N/A"
        return f"{v:.4f}"

    rows = [
        {"Metric": "Temporal Coherence", "Value": _fmt(report.temporal_coherence),
         "Note": "↑ better  (>0.95 = GOOD)"},
        {"Metric": "Silhouette Score", "Value": _fmt(report.silhouette_sample),
         "Note": "↑ better  (>0 = clusters separated)"},
        {"Metric": "Calinski-Harabasz", "Value": _fmt(report.calinski_harabasz),
         "Note": "↑ better  (higher = more compact)"},
        {"Metric": "Davies-Bouldin", "Value": _fmt(report.davies_bouldin),
         "Note": "↓ better  (<1 = well separated)"},
        {"Metric": "Single-Frame Bout Ratio", "Value": _fmt(report.single_frame_ratio),
         "Note": "↓ better  (<0.1 = stable bouts)"},
        {"Metric": "Median Bout Duration (frames)",
         "Value": _fmt(report.median_bout_duration_frames), "Note": ""},
        {"Metric": "Bout Duration CV", "Value": _fmt(report.bout_duration_cv),
         "Note": ""},
        {"Metric": "Verdict", "Value": report.verdict, "Note": ""},
    ]

    if report.warnings:
        for w in report.warnings:
            rows.append({"Metric": "⚠ Warning", "Value": w, "Note": ""})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# UI Construction
# ---------------------------------------------------------------------------


def create_analysis_ui(storage_path, project_name, analysis_tab=None):
    """Create the Analysis sub-tab UI.

    Args:
        storage_path: gr.State with storage path string.
        project_name: gr.State with project name string.
        analysis_tab: gr.Tab reference for auto-loading sessions on tab enter.

    Returns:
        dict of UI components (for caller visibility).
    """
    ui = {}

    # Per-tab state: loaded AnnotatorData
    analysis_data = gr.State(None)

    with gr.Column():
        gr.Markdown("### 🔬 Analysis")

        # ---- Session Selector ----
        with gr.Row():
            ui["session_dropdown"] = gr.Dropdown(
                label="Select Session",
                choices=[],
                interactive=True,
                scale=3,
            )
            ui["load_btn"] = gr.Button("📂 Load Data", variant="primary", scale=1)

        ui["load_status"] = gr.Markdown("**Status:** Not loaded")

        gr.Markdown("---")

        # ---- Section A: Ethogram ----
        with gr.Accordion("📊 Section A: Ethogram", open=True):
            gr.Markdown(
                "Analyse behavioral sequences: transition probabilities, "
                "bout durations, and temporal structure."
            )

            with gr.Row():
                ui["ethogram_btn"] = gr.Button("▶ Generate Ethogram", variant="primary", scale=3)
                ui["export_csv_btn"] = gr.Button("📥 Export CSV", variant="secondary", scale=1)

            with gr.Row():
                with gr.Column(scale=1):
                    ui["transition_plot"] = gr.Plot(
                        label="Transition Matrix Heatmap",
                    )
                with gr.Column(scale=1):
                    ui["raster_plot"] = gr.Plot(
                        label="Behavior Timeline (Raster)",
                    )

            ui["bout_stats_df"] = gr.Dataframe(
                label="Bout Duration Statistics",
                interactive=False,
            )

            ui["export_csv_status"] = gr.Markdown("")
            ui["export_csv_file"] = gr.File(
                label="⬇️ Download Ethogram CSV (ZIP)", visible=False, interactive=False
            )

        gr.Markdown("---")

        # ---- Section B: Quality Metrics ----
        with gr.Accordion("📐 Section B: Quality Metrics", open=False):
            gr.Markdown(
                "Evaluate clustering quality using internal validation metrics. "
                "Embedding-based metrics (silhouette, CH, DB) require UMAP embeddings."
            )

            ui["metrics_btn"] = gr.Button("▶ Compute Metrics", variant="primary")

            ui["metrics_df"] = gr.Dataframe(
                label="Clustering Quality Report",
                interactive=False,
            )

        gr.Markdown("---")

        # ---- Section C: Group Comparison ----
        with gr.Accordion("🔄 Section C: Group Comparison", open=False):
            gr.Markdown(
                "**Coming soon.** Group comparison requires selecting two independent "
                "sessions or projects (e.g., control vs. treatment). "
                "The backend (`castle.core.comparison`) is ready — "
                "this UI panel will be expanded in a future update.\n\n"
                "> _Select two sessions to compare behavioral fingerprints, "
                "transition matrices, and bout statistics across groups._"
            )

    # ---- Event Bindings ----

    # Auto-load session list when entering the tab
    if analysis_tab is not None:
        analysis_tab.select(
            fn=_refresh_sessions,
            inputs=[storage_path, project_name],
            outputs=[ui["session_dropdown"]],
        )

    # Load button: refresh dropdown, then load data
    ui["load_btn"].click(
        fn=_refresh_sessions,
        inputs=[storage_path, project_name],
        outputs=[ui["session_dropdown"]],
    ).then(
        fn=_load_data,
        inputs=[storage_path, project_name, ui["session_dropdown"]],
        outputs=[analysis_data, ui["load_status"]],
    )

    # Generate Ethogram
    ui["ethogram_btn"].click(
        fn=generate_ethogram,
        inputs=[analysis_data],
        outputs=[ui["transition_plot"], ui["bout_stats_df"], ui["raster_plot"]],
    )

    # Export Ethogram CSV
    ui["export_csv_btn"].click(
        fn=export_ethogram_csv_handler,
        inputs=[storage_path, project_name, ui["session_dropdown"]],
        outputs=[ui["export_csv_status"], ui["export_csv_file"]],
    )

    # Compute Metrics
    ui["metrics_btn"].click(
        fn=compute_quality_metrics,
        inputs=[analysis_data],
        outputs=[ui["metrics_df"]],
    )

    return ui
