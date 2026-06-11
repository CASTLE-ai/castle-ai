"""
castle/analysis/report.py
Comprehensive HTML report generator for CASTLE projects.

Generates a self-contained HTML report with:
  - Project metadata and pipeline configuration
  - Ethogram timeline visualisation (inline base64 plot)
  - Quality metrics summary
  - Statistical tables
  - Optional group comparison section

Usage::

    from castle.analysis.report import ReportGenerator
    gen = ReportGenerator("/storage/my_project")
    path = gen.generate("report.html", include_ethogram=True, include_quality=True)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSS styles (inline, self-contained)
# ---------------------------------------------------------------------------

_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0; padding: 0; background: #f5f7fa; color: #2d3748;
}
.container { max-width: 1100px; margin: 0 auto; padding: 32px 24px; }
h1 { color: #1a202c; font-size: 2em; margin-bottom: 4px; }
h2 { color: #2b6cb0; font-size: 1.3em; border-bottom: 2px solid #bee3f8;
     padding-bottom: 6px; margin-top: 32px; }
h3 { color: #4a5568; font-size: 1.05em; margin-top: 20px; }
.subtitle { color: #718096; margin-bottom: 32px; }
.meta-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
             gap: 16px; margin-bottom: 24px; }
.meta-card { background: #fff; border-radius: 8px; padding: 16px;
             box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.meta-card .label { font-size: 0.78em; color: #718096; text-transform: uppercase;
                    letter-spacing: 0.05em; margin-bottom: 4px; }
.meta-card .value { font-size: 1.1em; font-weight: 600; color: #2d3748; word-break: break-all; }
table { width: 100%; border-collapse: collapse; background: #fff;
        border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-top: 12px; }
th { background: #2b6cb0; color: #fff; padding: 10px 14px;
     text-align: left; font-size: 0.85em; }
td { padding: 9px 14px; border-bottom: 1px solid #edf2f7; font-size: 0.9em; }
tr:last-child td { border-bottom: none; }
tr:hover td { background: #ebf8ff; }
.plot-container { background: #fff; border-radius: 8px; padding: 16px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-top: 12px; text-align: center; }
.plot-container img { max-width: 100%; height: auto; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 12px;
         font-size: 0.8em; font-weight: 600; }
.badge-ok { background: #c6f6d5; color: #276749; }
.badge-warn { background: #fefcbf; color: #744210; }
.badge-err { background: #fed7d7; color: #822727; }
.section { margin-bottom: 40px; }
.footer { color: #a0aec0; font-size: 0.8em; text-align: center; margin-top: 48px;
          padding-top: 16px; border-top: 1px solid #e2e8f0; }
@media print {
    body { background: #fff; }
    .container { max-width: 100%; padding: 16px; }
    table { box-shadow: none; }
}
"""


# ---------------------------------------------------------------------------
# Helper: figure → base64 PNG
# ---------------------------------------------------------------------------


def _fig_to_b64(fig) -> str:  # type: ignore[type-arg]
    """Encode a matplotlib Figure as a base64 PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# ReportGenerator
# ---------------------------------------------------------------------------


class ReportGenerator:
    """Generate comprehensive HTML analysis reports for a CASTLE project.

    Args:
        project_path: Absolute (or relative) path to the project directory.
        session_id:   Optional session/experiment identifier shown in the header.
    """

    def __init__(self, project_path: str, session_id: Optional[str] = None) -> None:
        self.project_path = os.path.abspath(project_path)
        self.project_name = os.path.basename(self.project_path)
        self.session_id = session_id
        self._generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Lazy-loaded data
        self._ethogram_data: Optional[dict] = None
        self._quality_data: Optional[dict] = None
        self._comparison_data: Optional[dict] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        output_path: Optional[str] = None,
        include_ethogram: bool = True,
        include_quality: bool = True,
        include_comparison: bool = False,
    ) -> str:
        """Generate the full HTML report and write it to *output_path*.

        Args:
            output_path:        Destination file. Defaults to
                                ``<project_path>/reports/report_<timestamp>.html``.
            include_ethogram:   Add ethogram section (requires cluster data).
            include_quality:    Add quality-metrics section (requires cluster data).
            include_comparison: Add group-comparison section (requires ≥2 projects).

        Returns:
            Absolute path to the written HTML file.
        """
        if output_path is None:
            report_dir = os.path.join(self.project_path, "reports")
            os.makedirs(report_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(report_dir, f"report_{ts}.html")

        # Build sections
        sections: list[str] = [
            self._render_header(),
        ]

        if include_ethogram:
            sections.append(self._render_ethogram_section())

        if include_quality:
            sections.append(self._render_quality_section())

        if include_comparison:
            sections.append(self._render_comparison_section())

        sections.append(self._render_footer())

        html = self._wrap_html("\n".join(sections))

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)

        logger.info("Report written to %s", output_path)
        return output_path

    # ------------------------------------------------------------------
    # HTML wrappers
    # ------------------------------------------------------------------

    def _wrap_html(self, body: str) -> str:
        title = f"CASTLE Report — {self.project_name}"
        return (
            "<!DOCTYPE html>\n"
            '<html lang="en">\n'
            "<head>\n"
            f'  <meta charset="UTF-8">\n'
            f'  <title>{title}</title>\n'
            f"  <style>{_CSS}</style>\n"
            "</head>\n"
            f"<body>\n<div class='container'>\n{body}\n</div>\n</body>\n</html>"
        )

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _render_header(self) -> str:
        """Render the report header with project metadata."""
        # Try to load project config
        config_data: dict = {}
        config_path = os.path.join(self.project_path, "castle_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path) as f:
                    config_data = json.load(f)
            except Exception:
                pass

        # Count data artefacts
        cluster_dir = os.path.join(self.project_path, "cluster")
        n_clusters = "—"
        n_frames = "—"
        if os.path.isdir(cluster_dir):
            try:
                import csv

                ts_files = [
                    f
                    for f in os.listdir(cluster_dir)
                    if f.startswith("time_series_") and f.endswith(".csv")
                ]
                labels: list[int] = []
                for tsf in ts_files:
                    with open(os.path.join(cluster_dir, tsf)) as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            labels.append(int(row.get("behavior", 0)))
                if labels:
                    n_frames = str(len(labels))
                    n_clusters = str(len(set(labels)))
            except Exception:
                pass

        session_str = self.session_id or "—"
        tracking_model = config_data.get("tracking", {}).get("model", "—")
        extraction_model = config_data.get("extraction", {}).get("model", "—")

        meta_cards = [
            ("Project", self.project_name),
            ("Session", session_str),
            ("Path", self.project_path),
            ("Generated", self._generated_at),
            ("Total Frames", n_frames),
            ("Clusters", n_clusters),
            ("Tracking Model", tracking_model),
            ("Extraction Model", extraction_model),
        ]

        cards_html = "\n".join(
            f"<div class='meta-card'>"
            f"<div class='label'>{label}</div>"
            f"<div class='value'>{value}</div>"
            f"</div>"
            for label, value in meta_cards
        )

        return (
            f"<h1>🏰 CASTLE Analysis Report</h1>\n"
            f"<p class='subtitle'>{self.project_name}</p>\n"
            f"<div class='meta-grid'>{cards_html}</div>\n"
        )

    def _render_ethogram_section(self) -> str:
        """Render ethogram timeline and bout statistics."""
        try:
            from castle.service.ethogram_service import analyze_ethogram  # noqa: PLC0415

            # fps=None → analyze_ethogram reads each video's real fps
            # (contract C-2); do not hard-code 30.
            data = analyze_ethogram(self.project_path)
            self._ethogram_data = data
        except Exception as exc:
            return (
                "<div class='section'><h2>📊 Ethogram</h2>"
                f"<p style='color:#e53e3e'>Could not load ethogram data: {exc}</p>"
                "</div>"
            )

        if data.get("status") != "success":
            return (
                "<div class='section'><h2>📊 Ethogram</h2>"
                f"<p style='color:#e53e3e'>{data.get('message', 'Unknown error')}</p>"
                "</div>"
            )

        parts: list[str] = ["<div class='section'><h2>📊 Ethogram</h2>"]

        # --- Inline ethogram plot ---
        plot_html = self._try_render_ethogram_plot(data)
        if plot_html:
            parts.append(plot_html)

        # --- Summary metrics ---
        parts.append("<h3>Summary</h3>")
        parts.append("<table><tr><th>Metric</th><th>Value</th></tr>")
        parts.append(f"<tr><td>Total frames</td><td>{data.get('n_frames', '—')}</td></tr>")
        parts.append(f"<tr><td>Clusters</td><td>{data.get('n_clusters', '—')}</td></tr>")
        parts.append(f"<tr><td>FPS</td><td>{data.get('fps', '—')}</td></tr>")
        parts.append(f"<tr><td>Temporal coherence</td><td>{data.get('temporal_coherence', '—')}</td></tr>")
        parts.append(f"<tr><td>Total bouts</td><td>{data.get('n_bouts_total', '—')}</td></tr>")
        parts.append("</table>")

        # --- Bout statistics table ---
        bout_stats = data.get("bout_stats", {})
        if bout_stats:
            parts.append("<h3>Bout Statistics</h3>")
            parts.append(
                "<table><tr>"
                "<th>Cluster</th><th>Name</th><th>Bouts</th>"
                "<th>Freq (%)</th><th>Mean dur (s)</th>"
                "<th>Median dur (s)</th><th>CV</th>"
                "</tr>"
            )
            for cid_str, bs in sorted(bout_stats.items(), key=lambda x: int(x[0])):
                freq_pct = f"{float(bs.get('frequency', 0)) * 100:.1f}"
                parts.append(
                    f"<tr>"
                    f"<td>{cid_str}</td>"
                    f"<td>{bs.get('cluster_name', '—')}</td>"
                    f"<td>{bs.get('n_bouts', '—')}</td>"
                    f"<td>{freq_pct}</td>"
                    f"<td>{bs.get('mean_duration_s', '—'):.3f}</td>"
                    f"<td>{bs.get('median_duration_s', '—'):.3f}</td>"
                    f"<td>{bs.get('cv_duration', '—'):.2f}</td>"
                    f"</tr>"
                )
            parts.append("</table>")

        # --- Transition matrix ---
        tm = data.get("transition_matrix", {})
        matrix = tm.get("matrix")
        cluster_names = tm.get("cluster_names", [])
        if matrix and cluster_names:
            parts.append("<h3>Transition Matrix</h3>")
            parts.append(
                f"<p style='color:#4a5568; font-size:0.9em'>"
                f"Transitions: {tm.get('n_transitions', '—')} &nbsp;|&nbsp; "
                f"Entropy: {tm.get('entropy', '—')} bits &nbsp;|&nbsp; "
                f"Stationarity: {tm.get('stationarity', '—')}"
                f"</p>"
            )
            header = "<tr><th></th>" + "".join(f"<th>{n[:14]}</th>" for n in cluster_names) + "</tr>"
            parts.append(f"<table>{header}")
            for i, name in enumerate(cluster_names):
                row = f"<tr><td><strong>{name[:14]}</strong></td>"
                for j in range(len(cluster_names)):
                    val = matrix[i][j]
                    row += f"<td>{val:.3f}</td>"
                row += "</tr>"
                parts.append(row)
            parts.append("</table>")

        parts.append("</div>")
        return "\n".join(parts)

    def _try_render_ethogram_plot(self, data: dict) -> str:
        """Attempt to render an ethogram bar chart. Returns HTML or ''."""
        try:
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415

            bout_stats = data.get("bout_stats", {})
            if not bout_stats:
                return ""

            names = [bs.get("cluster_name", cid) for cid, bs in sorted(bout_stats.items(), key=lambda x: int(x[0]))]
            freqs = [float(bs.get("frequency", 0)) * 100 for bs in sorted(bout_stats.values(), key=lambda _: 0)]
            colors = plt.cm.tab20(np.linspace(0, 1, len(names)))  # type: ignore[attr-defined]

            fig, ax = plt.subplots(figsize=(9, 4))
            bars = ax.barh(names, freqs, color=colors)
            ax.set_xlabel("Frequency (%)")
            ax.set_title("Cluster Frequency (Ethogram)")
            ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=8)
            ax.set_xlim(0, max(freqs) * 1.2 + 1)
            plt.tight_layout()

            b64 = _fig_to_b64(fig)
            plt.close(fig)

            return (
                "<div class='plot-container'>"
                f"<img src='data:image/png;base64,{b64}' alt='Ethogram plot'/>"
                "</div>"
            )
        except Exception as exc:
            logger.debug("Could not render ethogram plot: %s", exc)
            return ""

    def _render_quality_section(self) -> str:
        """Render clustering quality metrics."""
        try:
            from castle.service.metrics_service import evaluate_project_clustering  # noqa: PLC0415

            data = evaluate_project_clustering(self.project_path)
            self._quality_data = data
        except Exception as exc:
            return (
                "<div class='section'><h2>🔬 Quality Metrics</h2>"
                f"<p style='color:#e53e3e'>Could not load quality data: {exc}</p>"
                "</div>"
            )

        if "error" in data:
            return (
                "<div class='section'><h2>🔬 Quality Metrics</h2>"
                f"<p style='color:#e53e3e'>{data['error']}</p>"
                "</div>"
            )

        parts: list[str] = ["<div class='section'><h2>🔬 Quality Metrics</h2>"]

        # --- Inline scatter plot ---
        plot_html = self._try_render_quality_plot(data)
        if plot_html:
            parts.append(plot_html)

        # --- Metrics table ---
        def _badge(val: float, *, high_good: bool = True) -> str:
            if high_good:
                cls = "badge-ok" if val >= 0.5 else ("badge-warn" if val >= 0.2 else "badge-err")
            else:
                cls = "badge-ok" if val <= 0.5 else ("badge-warn" if val <= 0.8 else "badge-err")
            return f"<span class='badge {cls}'>{val:.4f}</span>"

        parts.append("<table><tr><th>Metric</th><th>Value</th><th>Notes</th></tr>")

        # Keys must match metrics_service's asdict(ClusterQualityReport): the
        # fields are silhouette_sample / calinski_harabasz / davies_bouldin (no
        # "_score" suffix) and there is no "inertia" (DBSCAN, not KMeans). The old
        # *_score / inertia keys rendered every row as N/A.
        metric_rows = [
            ("silhouette_sample", "Silhouette (sampled)", True, "Higher is better (range −1 to 1)"),
            ("calinski_harabasz", "Calinski-Harabász", True, "Higher is better"),
            ("davies_bouldin", "Davies-Bouldin", False, "Lower is better"),
            ("temporal_coherence", "Temporal coherence", True, "Frame-to-frame label continuity (0–1)"),
        ]

        for key, label, high_good, note in metric_rows:
            val = data.get(key)
            if val is None:
                val_html = "<em style='color:#a0aec0'>N/A</em>"
            else:
                try:
                    val_html = _badge(float(val), high_good=high_good)
                except Exception:
                    val_html = str(val)
            parts.append(f"<tr><td>{label}</td><td>{val_html}</td><td style='color:#718096'>{note}</td></tr>")

        # Any other numeric fields
        skip = {"error", *{k for k, *_ in metric_rows}}
        for k, v in data.items():
            if k in skip:
                continue
            if isinstance(v, (int, float)):
                parts.append(f"<tr><td>{k}</td><td>{v}</td><td></td></tr>")

        parts.append("</table></div>")
        return "\n".join(parts)

    def _try_render_quality_plot(self, data: dict) -> str:
        """Attempt to render a 2-D embedding scatter if available."""
        try:
            import matplotlib  # noqa: PLC0415

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: PLC0415
            import numpy as np  # noqa: PLC0415

            emb = data.get("embedding_2d")
            labels = data.get("labels")
            if emb is None or labels is None:
                return ""

            emb_arr = np.array(emb)
            lab_arr = np.array(labels)
            unique = np.unique(lab_arr)
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique)))  # type: ignore[attr-defined]

            fig, ax = plt.subplots(figsize=(7, 6))
            for i, uid in enumerate(unique):
                mask = lab_arr == uid
                ax.scatter(emb_arr[mask, 0], emb_arr[mask, 1], s=2, alpha=0.4, color=colors[i], label=str(uid))
            ax.set_title("Embedding (2-D)")
            ax.legend(markerscale=4, fontsize=7, bbox_to_anchor=(1, 1))
            plt.tight_layout()

            b64 = _fig_to_b64(fig)
            plt.close(fig)

            return (
                "<div class='plot-container'>"
                f"<img src='data:image/png;base64,{b64}' alt='Embedding scatter'/>"
                "</div>"
            )
        except Exception as exc:
            logger.debug("Could not render quality plot: %s", exc)
            return ""

    def _render_comparison_section(self) -> str:
        """Render group-comparison section (placeholder for single-project reports)."""
        return (
            "<div class='section'><h2>📈 Group Comparison</h2>"
            "<p style='color:#718096'>Group comparison requires multiple projects. "
            "Use <code>BatchRunner.generate_summary()</code> for cross-project analysis.</p>"
            "</div>"
        )

    def _render_footer(self) -> str:
        return (
            f"<div class='footer'>"
            f"Generated by CASTLE on {self._generated_at}"
            f"</div>"
        )
