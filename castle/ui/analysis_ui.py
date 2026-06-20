"""
castle/ui/analysis_ui.py
Analysis UI — Sub-tab inside "4. Behavior Microscope".

Sections:
  A. Ethogram  — transition matrix heatmap, bout duration stats, raster plot
                 + annotated video export
  B. Quality Metrics — silhouette, temporal coherence, bout quality
  C. Group Comparison — placeholder (requires two sessions)

All data is loaded from disk via :func:`castle.service.annotator_loader.load_annotator_data`,
using the same session-selector pattern as ``annotator_ui.py``.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple

import cv2
import numpy as np
import gradio as gr

from castle.service.annotator_loader import load_annotator_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ethogram video helpers
# ---------------------------------------------------------------------------

def _hex_to_bgr(hex_str: str) -> Tuple[int, int, int]:
    """Convert #RRGGBB hex to (B, G, R) tuple for OpenCV."""
    h = hex_str.lstrip('#')
    if len(h) == 6:
        try:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return (b, g, r)
        except ValueError:
            pass
    return (128, 128, 128)


def _draw_label_at(
    bgr_img: np.ndarray,
    label: str,
    color_bgr: Tuple[int, int, int],
    anchor_x: int,
    anchor_y: int,
) -> None:
    """Draw a cluster label badge anchored at (anchor_x, anchor_y) on a BGR image.

    The badge is clamped inside the image boundaries.
    anchor_x / anchor_y is the top-left corner of the badge.
    """
    ih, iw = bgr_img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.4, iw / 1280)
    thickness = max(1, int(font_scale * 2))
    (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    pad = 4
    bw, bh = tw + pad * 2, th + baseline + pad * 2
    # Clamp so badge stays inside frame
    x0 = max(0, min(anchor_x, iw - bw - 1))
    y0 = max(0, min(anchor_y, ih - bh - 1))
    x1, y1 = x0 + bw, y0 + bh
    cv2.rectangle(bgr_img, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.rectangle(bgr_img, (x0, y0), (x1, y1), color_bgr, 1)
    cv2.putText(bgr_img, label, (x0 + pad, y0 + th + pad - 1),
                font, font_scale, color_bgr, thickness, cv2.LINE_AA)


def _populate_video_choices(analysis_data) -> dict:
    """Return gr.update for the annotated-video selector dropdown."""
    if analysis_data is None or not getattr(analysis_data, 'videos_meta', None):
        return gr.update(choices=["All Videos"], value="All Videos")
    basenames = [os.path.basename(v) for _, v in analysis_data.videos_meta]
    return gr.update(choices=["All Videos"] + basenames, value="All Videos")


def _populate_etho_videos(analysis_data) -> dict:
    """Per-subject ethogram selector: list the videos (no 'All'), default first.

    The ethogram/bout stats are computed one video at a time so durations use
    each video's own fps and no bout is merged across a video boundary.
    """
    if analysis_data is None or not getattr(analysis_data, 'videos_meta', None):
        return gr.update(choices=[], value=None)
    basenames = [os.path.basename(v) for _, v in analysis_data.videos_meta]
    return gr.update(choices=basenames, value=basenames[0] if basenames else None)


def generate_ethogram_video(
    analysis_data,
    selected_video: str,
    view_mode: str = "DBSCAN clusters",
    progress=gr.Progress(),
) -> Tuple[Optional[str], str]:
    """Render an annotated ethogram video with cluster overlay and label.

    Three composited layers per frame:
    a. Original video frame
    b. ROI mask contour in the cluster's assigned colour (requires mask_list.h5)
    c. Behaviour label badge (top-left corner)

    Args:
        view_mode: ``"DBSCAN clusters"`` uses the original per-cluster colour
            and name; ``"Annotation labels"`` uses the merged palette colour and
            the human behavior label (or "Unannotated").

    Returns:
        (output_file_path_or_None, status_markdown)
    """
    if analysis_data is None:
        gr.Warning("Please load cluster data first.")
        return None, "**❌ No data loaded.**"

    if selected_video == "All Videos":
        target_videos = list(analysis_data.videos_meta)
    else:
        target_videos = [
            (n, v) for n, v in analysis_data.videos_meta
            if os.path.basename(v) == selected_video
        ]
    if not target_videos:
        gr.Warning("No matching video found in loaded data.")
        return None, "**❌ No matching video.**"

    tmp_dir = tempfile.mkdtemp(prefix="castle_etho_video_")
    safe_name = selected_video.replace(" ", "_").replace("/", "-")
    raw_path = os.path.join(tmp_dir, f"raw_{safe_name}.mp4")
    final_path = os.path.join(tmp_dir, f"{safe_name}.mp4")

    bin_size = max(1, analysis_data.bin_size)
    fps = analysis_data.fps if analysis_data.fps > 0 else 30.0
    cluster_arr = analysis_data.cluster
    total_bins_count = max(sum(n for n, _ in target_videos), 1)

    # Load human-annotated behavior labels (name → behavior_label)
    behavior_labels: dict = {}
    annotations = _load_annotations_for(analysis_data)
    for bm_name, info in annotations.items():
        if info.get("behavior_label"):
            behavior_labels[bm_name] = info["behavior_label"]

    # In merged mode build per-cluster-ID colour and label overrides.
    merged_color_map: dict = {}   # old cluster_id → hex color
    merged_label_map: dict = {}   # old cluster_id → display label
    if view_mode == "Annotation labels":
        label_remap, merged_names, merged_colors = _build_annotation_remap(
            analysis_data, annotations
        )
        for cid, synth_id in label_remap.items():
            merged_color_map[cid] = merged_colors[synth_id]
            merged_label_map[cid] = merged_names[synth_id]

    writer: Optional[cv2.VideoWriter] = None
    out_w = out_h = None
    global_bin_offset = 0

    try:
        import av as _av
    except ImportError:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        gr.Warning("PyAV is required for video generation but is not installed.")
        return None, "**❌ PyAV not available.**"

    from castle.service.bout_service import _load_mask_contours

    for n_bins, video_name in target_videos:
        video_path = os.path.join(analysis_data.source_path, video_name)
        video_basename = os.path.basename(video_name)
        mask_path = os.path.join(
            analysis_data.project_path, "track", video_basename, "mask_list.h5"
        )
        has_mask = os.path.exists(mask_path)

        try:
            container = _av.open(video_path)
        except Exception as exc:
            logger.warning("Cannot open video %s: %s", video_path, exc)
            global_bin_offset += n_bins
            continue

        video_stream = container.streams.video[0]
        video_stream.thread_type = "AUTO"

        # Map each ORIGINAL frame to its cluster label via the FrameIndexMap.
        # bin_size/n_bins are in DECIMATED frames for a prepared cache and
        # decimation is non-uniform, so the legacy `frame_idx // bin_size` is
        # wrong there (it saturates and paints most of the video with the last
        # window's label). expand_labels_to_orig handles both: for a legacy
        # (window=1) map it reproduces the np.repeat(bin_size) behaviour exactly.
        per_frame_labels = None
        fim = getattr(analysis_data, "frame_index_map", None)
        if fim is not None:
            try:
                names = [str(x) for x in fim.base.video_names]
                if video_name in names:
                    vidx = names.index(video_name)
                else:
                    bn = os.path.basename(video_name)
                    vidx = next(
                        (i for i, nm in enumerate(names) if os.path.basename(nm) == bn),
                        None,
                    )
                if vidx is not None:
                    win_labels = cluster_arr[global_bin_offset:global_bin_offset + n_bins]
                    per_frame_labels = fim.expand_labels_to_orig(win_labels, vidx)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "FrameIndexMap expand failed for %s: %s; falling back to bin arithmetic",
                    video_name, exc,
                )
                per_frame_labels = None

        frame_idx = 0
        try:
            for av_frame in container.decode(video_stream):
                frame_rgb: np.ndarray = av_frame.to_rgb().to_ndarray()
                fh, fw = frame_rgb.shape[:2]

                if writer is None:
                    out_w = max(fw & ~1, 2)
                    out_h = max(fh & ~1, 2)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(raw_path, fourcc, fps, (out_w, out_h))

                canvas_rgb = (
                    cv2.resize(frame_rgb, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                    if (fw != out_w or fh != out_h) else frame_rgb.copy()
                )
                # Convert to BGR once — all subsequent drawing uses BGR colour tuples
                canvas_bgr = cv2.cvtColor(canvas_rgb, cv2.COLOR_RGB2BGR)

                if per_frame_labels is not None:
                    cluster_id = (
                        int(per_frame_labels[frame_idx])
                        if frame_idx < len(per_frame_labels) else -1
                    )
                else:
                    local_bin = min(frame_idx // bin_size, n_bins - 1)
                    abs_bin = global_bin_offset + local_bin
                    cluster_id = int(cluster_arr[abs_bin]) if abs_bin < len(cluster_arr) else -1

                meta = analysis_data.cluster_meta.get(cluster_id, {})
                if merged_color_map:
                    color_bgr = _hex_to_bgr(merged_color_map.get(cluster_id, "#808080"))
                    label = merged_label_map.get(cluster_id, f"cluster {cluster_id}")
                else:
                    bm_name = meta.get("name", "")
                    # Resolve an empty (engine-default) colour live by name so the
                    # overlay video matches the figures' mode-aware palette.
                    from castle.core.config import color_for_name
                    color_bgr = _hex_to_bgr(meta.get("color") or color_for_name(bm_name))
                    label = behavior_labels.get(bm_name) or bm_name or f"cluster {cluster_id}"

                # Fix 2: track contour bounding box to anchor label near ROI
                label_x, label_y = 6, 6  # fallback: top-left
                if has_mask:
                    # Fix 1: draw on BGR canvas so color_bgr is interpreted correctly
                    result = _load_mask_contours(mask_path, frame_idx)
                    if result is not None:
                        contours, (mh, mw) = result
                        sx, sy = out_w / mw, out_h / mh
                        scaled_contours = []
                        for cnt in contours:
                            scaled = cnt.copy().astype(np.float64)
                            scaled[:, :, 0] *= sx
                            scaled[:, :, 1] *= sy
                            scaled = np.clip(
                                scaled.astype(np.int32), 0, [out_w - 1, out_h - 1]
                            )
                            scaled_contours.append(scaled)
                        cv2.drawContours(canvas_bgr, scaled_contours, -1, color_bgr, 2)
                        # Place label at top of the contour bounding box
                        all_pts = np.vstack(scaled_contours).reshape(-1, 2)
                        label_x = int(all_pts[:, 0].min())
                        label_y = max(0, int(all_pts[:, 1].min()) - 2)

                # Layer C: behavior label anchored near the ROI
                _draw_label_at(canvas_bgr, label, color_bgr, label_x, label_y)
                writer.write(canvas_bgr)

                frame_idx += 1
                if frame_idx % 60 == 0:
                    done = global_bin_offset + min(frame_idx // bin_size, n_bins)
                    progress(done / total_bins_count, desc=f"Encoding {video_basename}…")
        finally:
            container.close()

        global_bin_offset += n_bins

    if writer is not None:
        writer.release()
    else:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, "**❌ Could not open any video file.**"

    # Re-encode to H.264 for browser playback
    chosen = None
    status = "**❌ Video generation failed.**"
    try:
        ret = subprocess.run(
            ["ffmpeg", "-y", "-i", raw_path,
             "-vcodec", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", final_path],
            capture_output=True, timeout=600,
        )
        if ret.returncode == 0 and os.path.exists(final_path):
            chosen = final_path
            status = "**✅ Ethogram video ready for download!**"
    except Exception as exc:
        logger.warning("ffmpeg re-encode failed: %s", exc)
    if chosen is None and os.path.exists(raw_path):
        chosen = raw_path
        status = "**✅ Ethogram video ready (mp4v fallback).**"

    if chosen is None:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None, status

    # Move the chosen artifact into a stable project-scoped dir, then remove the
    # temp dir — otherwise every click leaks a full re-encoded MP4 in system temp.
    out_dir = os.path.join(analysis_data.project_path, "analysis", "ethogram_video")
    try:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"{safe_name}{os.path.splitext(chosen)[1]}")
        shutil.move(chosen, out_path)
    except OSError as exc:
        logger.warning("Could not relocate ethogram video out of temp dir: %s", exc)
        return chosen, status  # leave tmp_dir so the returned path stays valid
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return out_path, status


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
# Annotation-merge helpers
# ---------------------------------------------------------------------------


def _load_annotations_for(annotator_data) -> dict:
    """Load annotations dict for the given AnnotatorData. Returns {} on failure."""
    try:
        from castle.service.annotation_service import load_annotations as _load_ann
        project_path = annotator_data.project_path
        storage_path = os.path.dirname(project_path)
        project_name = os.path.basename(project_path)
        return _load_ann(storage_path, project_name, session_id=annotator_data.session_id)
    except Exception as exc:
        logger.debug("Could not load annotations: %s", exc)
        return {}


def _build_annotation_remap(annotator_data, annotations: dict):
    """Build cluster-ID remapping for annotation-merged view mode.

    Clusters sharing the same human behavior_label are collapsed to a single
    synthetic ID.  Clusters without a label are grouped as "Unannotated".

    Returns:
        label_remap   – Dict[int, int] mapping original cluster_id → synthetic_id
        merged_names  – Dict[int, str] synthetic_id → display name
        merged_colors – Dict[int, str] synthetic_id → hex color (#rrggbb)
    """
    cid_to_label: dict = {}
    for cid, meta in (annotator_data.cluster_meta or {}).items():
        bm_name = meta.get("name", "")
        ann = annotations.get(bm_name)
        label = (ann.get("behavior_label") if ann else None) or "Unannotated"
        cid_to_label[int(cid)] = label

    # Deterministic ordering: annotated labels alphabetically, "Unannotated" last
    annotated = sorted({lbl for lbl in cid_to_label.values() if lbl != "Unannotated"})
    has_unannotated = "Unannotated" in cid_to_label.values()
    unique_labels = annotated + (["Unannotated"] if has_unannotated else [])

    label_to_synth = {lbl: i for i, lbl in enumerate(unique_labels)}
    label_remap = {cid: label_to_synth[lbl] for cid, lbl in cid_to_label.items()}
    merged_names = {sid: lbl for lbl, sid in label_to_synth.items()}

    # Colour merged annotation labels through the unified mode-aware engine so
    # this view matches the figures; "Unannotated" stays neutral grey.
    from castle.core.config import palette_color
    merged_colors: dict = {}
    palette_idx = 0
    for lbl, sid in label_to_synth.items():
        if lbl == "Unannotated":
            merged_colors[sid] = "#aaaaaa"
        else:
            merged_colors[sid] = palette_color(palette_idx)
            palette_idx += 1

    return label_remap, merged_names, merged_colors


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
        export_ethogram_csv(project_path=project_path, output_path=csv_dir, session_id=session_id or None)
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


def generate_ethogram(annotator_data, selected_video, view_mode="DBSCAN clusters"):
    """Compute a per-video ethogram; return (heatmap fig, stats df, raster fig).

    Per-subject: computes for ``selected_video`` only, from that video's
    per-frame time_series and its own fps — no cross-video bout merging and no
    mixed-fps duration errors.

    Args:
        view_mode: ``"DBSCAN clusters"`` (default) uses raw cluster IDs;
            ``"Annotation labels"`` merges clusters sharing the same human
            behavior label into a single category, with colours reassigned from
            the global palette.  Unannotated clusters are grouped as
            "Unannotated".  Within-label transitions become self-transitions
            and are excluded from the transition matrix (only state *changes*
            are counted, same rule as DBSCAN mode).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    if annotator_data is None:
        return None, None, None
    if not selected_video:
        gr.Warning("Please select a video to analyse.")
        return None, None, None

    annotations = _load_annotations_for(annotator_data)

    label_remap = None
    if view_mode == "Annotation labels":
        label_remap, merged_names, _ = _build_annotation_remap(annotator_data, annotations)
        if not merged_names or set(merged_names.values()) == {"Unannotated"}:
            gr.Warning(
                "No behavior annotations found. Annotate clusters in the Annotator tab "
                "first, then switch to 'Annotation labels' view."
            )
        cluster_names = merged_names
    else:
        def _display_name(bm_name: str) -> str:
            ann = annotations.get(bm_name)
            if ann and ann.get("behavior_label"):
                return f"{ann['behavior_label']} \u2014 {bm_name}"
            return bm_name

        cluster_names = {
            cid: _display_name(meta["name"])
            for cid, meta in annotator_data.cluster_meta.items()
        }

    # Compute per-video ethogram via the service layer (own fps, no cross-video
    # bouts/transitions). Reads the selected video's per-frame time_series.
    from castle.service.ethogram_service import compute_video_ethogram

    try:
        ethogram = compute_video_ethogram(
            annotator_data.project_path, selected_video,
            cluster_names=cluster_names,
            label_remap=label_remap,
        )
    except FileNotFoundError as exc:
        gr.Warning(
            f"No clustering time-series found for {selected_video}. Make sure clustering "
            f"has been submitted for this video. Details: {exc}"
        )
        return None, None, None
    except Exception as exc:
        logger.exception("compute_video_ethogram failed")
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
    # Report unlabeled (noise / unclustered) frames separately — they are
    # excluded from every bout/transition statistic above, not treated as a
    # behavior. Surface the fraction both persistently (a marked row) and as
    # a toast so it isn't missed for a published figure.
    unlabeled_pct = ethogram.unlabeled_fraction * 100.0
    rows.append(
        {
            "Cluster": "Unlabeled (noise, excluded)",
            "N Bouts": "—",
            "Freq (%)": f"{unlabeled_pct:.1f}",
            "Mean Dur (s)": "—",
            "Median Dur (s)": "—",
            "Std Dur (s)": "—",
            "CV": "—",
            "Mean IBI (s)": "—",
        }
    )
    stats_df = pd.DataFrame(rows)
    gr.Info(
        f"{selected_video}: {unlabeled_pct:.1f}% of frames are unlabeled "
        f"(noise / unclustered) and were excluded from bout & transition statistics."
    )

    # --- A3: Ethogram raster ---
    raster_fig = plot_ethogram_raster(ethogram)

    plt.close("all")
    return heatmap_fig, stats_df, raster_fig


# ---------------------------------------------------------------------------
# Section B: Quality Metrics
# ---------------------------------------------------------------------------


def compute_quality_metrics(annotator_data, view_mode="DBSCAN clusters"):
    """Compute clustering quality metrics and return a dataframe.

    Args:
        view_mode: ``"DBSCAN clusters"`` (default) evaluates the raw DBSCAN
            partitioning; ``"Annotation labels"`` first merges clusters that
            share the same human behavior label before computing metrics.
    """
    import pandas as pd
    import numpy as np

    if annotator_data is None:
        return None

    cluster_labels = annotator_data.cluster
    embedding = annotator_data.embedding  # may be None

    if view_mode == "Annotation labels":
        annotations = _load_annotations_for(annotator_data)
        label_remap, _, _ = _build_annotation_remap(annotator_data, annotations)
        cluster_labels = np.array(
            [label_remap.get(int(lbl), int(lbl)) for lbl in cluster_labels],
            dtype=np.int32,
        )
        # Embedding stays aligned; only labels change.

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
                "bout durations, and temporal structure. **One ethogram per video** "
                "(per subject) — durations use each video's own fps. "
                "_Export CSV_ writes every video (long-format with a `video` column)."
            )

            ui["etho_video_selector"] = gr.Dropdown(
                label="Video (one ethogram per video / subject)",
                choices=[], value=None, interactive=True,
            )

            with gr.Row():
                ui["view_mode"] = gr.Radio(
                    choices=["DBSCAN clusters", "Annotation labels"],
                    value="DBSCAN clusters",
                    label="View mode",
                    scale=2,
                )
                ui["ethogram_btn"] = gr.Button("▶ Generate Ethogram", variant="primary", scale=2)
                ui["export_csv_btn"] = gr.Button("📥 Export CSV (all videos)", variant="secondary", scale=1)

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
            gr.Markdown("#### 🎬 Annotated Ethogram Video")
            gr.Markdown(
                "Render an annotated video with three layers: "
                "**a)** original frames, "
                "**b)** cluster-coloured ROI contour, "
                "**c)** behaviour label badge."
            )

            with gr.Row():
                ui["ethogram_video_selector"] = gr.Dropdown(
                    label="Video to visualise",
                    choices=["All Videos"],
                    value="All Videos",
                    interactive=True,
                    scale=3,
                )
                ui["ethogram_video_btn"] = gr.Button(
                    "🎬 Generate Video", variant="secondary", scale=1
                )

            ui["ethogram_video_status"] = gr.Markdown("")
            ui["ethogram_video_file"] = gr.File(
                label="⬇️ Download Ethogram Video", visible=True, interactive=False
            )

        gr.Markdown("---")

        # ---- Section B: Quality Metrics ----
        with gr.Accordion("📐 Section B: Quality Metrics", open=False):
            gr.Markdown(
                "Evaluate clustering quality using internal validation metrics. "
                "Embedding-based metrics (silhouette, CH, DB) require UMAP embeddings. "
                "_View mode (DBSCAN clusters / Annotation labels) is shared with Section A._"
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

    # Load button: refresh dropdown → load data → populate video selector
    ui["load_btn"].click(
        fn=_refresh_sessions,
        inputs=[storage_path, project_name],
        outputs=[ui["session_dropdown"]],
    ).then(
        fn=_load_data,
        inputs=[storage_path, project_name, ui["session_dropdown"]],
        outputs=[analysis_data, ui["load_status"]],
    ).then(
        fn=_populate_video_choices,
        inputs=[analysis_data],
        outputs=[ui["ethogram_video_selector"]],
    ).then(
        fn=_populate_etho_videos,
        inputs=[analysis_data],
        outputs=[ui["etho_video_selector"]],
    )

    # Generate Ethogram (per selected video)
    ui["ethogram_btn"].click(
        fn=generate_ethogram,
        inputs=[analysis_data, ui["etho_video_selector"], ui["view_mode"]],
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
        inputs=[analysis_data, ui["view_mode"]],
        outputs=[ui["metrics_df"]],
    )

    # Generate Ethogram Video
    ui["ethogram_video_btn"].click(
        fn=generate_ethogram_video,
        inputs=[analysis_data, ui["ethogram_video_selector"], ui["view_mode"]],
        outputs=[ui["ethogram_video_file"], ui["ethogram_video_status"]],
    )

    return ui
