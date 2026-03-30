#!/usr/bin/env python3
"""Streamlit RTSS diff viewer app."""

from __future__ import annotations

import sys
import tempfile
from math import sqrt
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rtssdiffviewer.dcm_to_json import dcm_to_json  # noqa: E402
from rtssdiffviewer.diff_core import (  # noqa: E402
    COMPONENT_KEYS,
    DEFAULT_VOLATILE_TAG_PREFIXES,
    contour_diff_text,
    get_contour_slices_structured,
    normalize_value,
    pretty_json_text,
    select_component,
    unified_diff_text,
)

try:
    from st_diff_viewer import diff_viewer
except Exception:
    diff_viewer = None


def dcm_bytes_to_json_dict(payload: bytes) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp.write(payload)
        tmp_path = Path(tmp.name)
    try:
        return dcm_to_json(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def ensure_state() -> None:
    st.session_state.setdefault("left_name", None)
    st.session_state.setdefault("right_name", None)
    st.session_state.setdefault("left_json_raw", None)
    st.session_state.setdefault("right_json_raw", None)
    st.session_state.setdefault("batch_variants", {})
    st.session_state.setdefault("pair_step", "Upload Pair")
    st.session_state.setdefault("pair_diff_requested", False)
    st.session_state.setdefault("pair_diff_sig", None)
    st.session_state.setdefault("pair_focus_slice_z", None)
    st.session_state.setdefault("axial_target_slice_z", None)
    st.session_state.setdefault("contour_detail_target_slice_z", None)


def _point_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _greedy_point_matches(
    left_points: list[tuple[float, float, float]],
    right_points: list[tuple[float, float, float]],
) -> list[tuple[int, int, float]]:
    if not left_points or not right_points:
        return []

    candidates: list[tuple[float, int, int]] = []
    for li, lp in enumerate(left_points):
        for ri, rp in enumerate(right_points):
            candidates.append((_point_distance(lp, rp), li, ri))
    candidates.sort(key=lambda x: x[0])

    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for dist, li, ri in candidates:
        if li in used_left or ri in used_right:
            continue
        used_left.add(li)
        used_right.add(ri)
        matches.append((li, ri, dist))
    return matches


def _slice_point_counts(
    left_rois: dict[str, list[tuple[float, float, float]]],
    right_rois: dict[str, list[tuple[float, float, float]]],
) -> tuple[int, int, list[tuple[float, float, float]], list[tuple[float, float, float]]]:
    left_points = [p for pts in left_rois.values() for p in pts]
    right_points = [p for pts in right_rois.values() for p in pts]
    return len(left_points), len(right_points), left_points, right_points


def _slice_match_metrics(
    left_rois: dict[str, list[tuple[float, float, float]]],
    right_rois: dict[str, list[tuple[float, float, float]]],
    tolerance_mm: float,
) -> dict[str, Any]:
    left_count, right_count, left_points, right_points = _slice_point_counts(left_rois, right_rois)
    matches = _greedy_point_matches(left_points, right_points)
    matched_in_tol = sum(1 for _, _, dist in matches if dist <= tolerance_mm)
    dice = (2.0 * matched_in_tol) / (left_count + right_count) if (left_count + right_count) > 0 else 1.0
    mismatch_count = left_count + right_count - (2 * matched_in_tol)
    count_delta = abs(left_count - right_count)

    left_roi_names = set(left_rois.keys())
    right_roi_names = set(right_rois.keys())
    all_roi_names = sorted(left_roi_names | right_roi_names)
    identical_slice = left_roi_names == right_roi_names and all(
        sorted(left_rois.get(roi, [])) == sorted(right_rois.get(roi, []))
        for roi in all_roi_names
    )

    return {
        "left_count": left_count,
        "right_count": right_count,
        "dice": dice,
        "mismatch_count": mismatch_count,
        "count_delta": count_delta,
        "left_roi_names": left_roi_names,
        "right_roi_names": right_roi_names,
        "all_roi_names": all_roi_names,
        "identical_slice": identical_slice,
    }


def _safe_key_fragment(value: str) -> str:
    out = []
    for ch in value:
        out.append(ch if ch.isalnum() else "_")
    return "".join(out)


def _nearest_slice_value(slices: list[float], target: float | None) -> float | None:
    if not slices:
        return None
    if target is None:
        return slices[0]
    return min(slices, key=lambda z: (abs(z - target), z))


def _step_slice_value(slices: list[float], current: float, step: int) -> float:
    if not slices:
        return current
    try:
        idx = slices.index(current)
    except ValueError:
        idx = 0
    next_idx = max(0, min(len(slices) - 1, idx + step))
    return slices[next_idx]


def _format_slice_rois_text(rois: dict[str, list[tuple[float, float, float]]], precision: int) -> str:
    if not rois:
        return "(no contours on this slice)"

    lines: list[str] = []
    for roi_name in sorted(rois.keys()):
        points = rois[roi_name]
        lines.append(f"{roi_name}: {len(points)} points")
        for i, (x, y, z) in enumerate(points, start=1):
            lines.append(f"  {i:03d}: ({x:.{precision}f}, {y:.{precision}f}, {z:.{precision}f})")
        lines.append("")
    return "\n".join(lines).strip()


def _extract_ordered_contours_by_slice(
    rtss_json: dict[str, Any],
    precision: int = 4,
) -> dict[float, list[dict[str, Any]]]:
    """Extract ordered contour polylines grouped by axial slice (z)."""
    slices: dict[float, list[dict[str, Any]]] = {}
    roi_contour_seq = rtss_json.get("(3006,0039) ROIContourSequence", [])
    if not isinstance(roi_contour_seq, list):
        return slices

    for roi_idx, roi_item in enumerate(roi_contour_seq):
        if not isinstance(roi_item, dict):
            continue

        roi_number = roi_item.get("(3006,0084) ReferencedROINumber")
        if roi_number is None:
            roi_number = roi_idx
        roi_name = f"ROI {roi_number}"

        contour_seq = roi_item.get("(3006,0040) ContourSequence", [])
        if not isinstance(contour_seq, list):
            continue

        for contour_idx, contour_item in enumerate(contour_seq, start=1):
            if not isinstance(contour_item, dict):
                continue

            points = _extract_xyz_points(contour_item.get("(3006,0050) ContourData", []))
            if not points:
                continue

            z_mean = sum(p[2] for p in points) / len(points)
            z_key = round(z_mean, precision)
            contour_number = contour_item.get("(3006,0048) ContourNumber", contour_idx)

            slices.setdefault(z_key, []).append(
                {
                    "roi_name": roi_name,
                    "contour_label": f"{roi_name} | Contour {contour_number}",
                    "points": points,
                }
            )

    return slices


def _add_direction_annotation(
    fig: go.Figure,
    points: list[tuple[float, float, float]],
    color: str,
    text: str,
) -> None:
    if len(points) < 2:
        return
    x0, y0, _ = points[0]
    x1, y1, _ = points[1]
    fig.add_annotation(
        x=x1,
        y=y1,
        ax=x0,
        ay=y0,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=3,
        arrowsize=1,
        arrowwidth=1.6,
        arrowcolor=color,
        text=text,
        font={"size": 11, "color": color},
        align="left",
        xanchor="left",
    )


def _add_contour_traces_2d(
    fig: go.Figure,
    contours: list[dict[str, Any]],
    side_name: str,
    line_color: str,
    line_dash: str,
) -> None:
    for idx, contour in enumerate(contours, start=1):
        points = contour["points"]
        if not points:
            continue

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        point_ids = list(range(1, len(points) + 1))
        contour_name = contour["contour_label"]

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                line={"width": 2, "color": line_color, "dash": line_dash},
                marker={"size": 5, "color": line_color},
                name=f"{side_name}: {contour_name}",
                legendgroup=f"{side_name}_{idx}",
                hovertemplate=(
                    f"{side_name}<br>{contour_name}<br>Point %{{customdata}}"
                    "<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"
                ),
                customdata=point_ids,
            )
        )

        first = points[0]
        last = points[-1]
        fig.add_trace(
            go.Scatter(
                x=[first[0]],
                y=[first[1]],
                mode="markers+text",
                marker={"size": 11, "color": line_color, "symbol": "star"},
                text=["Start"],
                textposition="top center",
                name=f"{side_name} start",
                legendgroup=f"{side_name}_{idx}",
                showlegend=False,
                hovertemplate=(
                    f"{side_name}<br>{contour_name}<br>Start point (index 1)"
                    "<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[last[0]],
                y=[last[1]],
                mode="markers+text",
                marker={"size": 10, "color": line_color, "symbol": "x"},
                text=[f"End ({len(points)})"],
                textposition="bottom center",
                name=f"{side_name} end",
                legendgroup=f"{side_name}_{idx}",
                showlegend=False,
                hovertemplate=(
                    f"{side_name}<br>{contour_name}<br>End point (index {len(points)})"
                    "<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>"
                ),
            )
        )

        _add_direction_annotation(fig, points, line_color, f"{side_name} direction")


def render_axial_contour_view(
    *,
    left_name: str,
    right_name: str,
    left_raw: dict[str, Any],
    right_raw: dict[str, Any],
    precision: int = 4,
) -> None:
    left_by_slice = _extract_ordered_contours_by_slice(left_raw, precision=precision)
    right_by_slice = _extract_ordered_contours_by_slice(right_raw, precision=precision)
    all_slices = sorted(set(left_by_slice.keys()) | set(right_by_slice.keys()))

    if not all_slices:
        st.warning("No contour data found in either RTSS file.")
        return

    summary_left = sum(len(v) for v in left_by_slice.values())
    summary_right = sum(len(v) for v in right_by_slice.values())
    only_left = [z for z in all_slices if z in left_by_slice and z not in right_by_slice]
    only_right = [z for z in all_slices if z in right_by_slice and z not in left_by_slice]

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(f"{left_name} slices", f"{len(left_by_slice):,}")
    with m2:
        st.metric(f"{right_name} slices", f"{len(right_by_slice):,}")
    with m3:
        st.metric(f"{left_name} contours", f"{summary_left:,}")
    with m4:
        st.metric(f"{right_name} contours", f"{summary_right:,}")

    if only_left or only_right:
        notes: list[str] = []
        if only_left:
            notes.append(f"Slices only in {left_name}: {len(only_left)}")
        if only_right:
            notes.append(f"Slices only in {right_name}: {len(only_right)}")
        st.warning(" | ".join(notes))
    else:
        st.success("Both files have contours on the same set of axial slice positions.")

    target_z = st.session_state.axial_target_slice_z
    if target_z is not None:
        aligned_target = _nearest_slice_value(all_slices, target_z)
        if aligned_target is not None:
            st.session_state.axial_slice_select = aligned_target
        st.session_state.axial_target_slice_z = None
    elif st.session_state.get("axial_slice_select") not in all_slices:
        st.session_state.axial_slice_select = all_slices[0]

    nav_col_1, nav_col_2, nav_col_3 = st.columns(3)
    with nav_col_1:
        if st.button("Previous Slice", key="axial_prev_slice"):
            st.session_state.axial_slice_select = _step_slice_value(
                all_slices,
                st.session_state.get("axial_slice_select", all_slices[0]),
                -1,
            )
            st.rerun()
    with nav_col_2:
        if st.button("Next Slice", key="axial_next_slice"):
            st.session_state.axial_slice_select = _step_slice_value(
                all_slices,
                st.session_state.get("axial_slice_select", all_slices[0]),
                1,
            )
            st.rerun()
    with nav_col_3:
        if st.button("Sync To Contour Detail", key="axial_open_contour_detail"):
            current_z = st.session_state.get("axial_slice_select", all_slices[0])
            st.session_state.contour_detail_target_slice_z = current_z
            st.session_state.pair_focus_slice_z = current_z
            st.info(f"Slice z={current_z} synced. Switch to the Contour Detail tab.")

    selected_z = st.select_slider(
        "Axial slice z (mm)",
        options=all_slices,
        format_func=lambda z: f"{z:.{precision}f}",
        value=st.session_state.get("axial_slice_select", all_slices[0]),
        key="axial_slice_select",
    )
    st.session_state.pair_focus_slice_z = selected_z

    left_contours = left_by_slice.get(selected_z, [])
    right_contours = right_by_slice.get(selected_z, [])

    left_labels = {c["contour_label"] for c in left_contours}
    right_labels = {c["contour_label"] for c in right_contours}
    only_left_contours = sorted(left_labels - right_labels)
    only_right_contours = sorted(right_labels - left_labels)

    if left_contours and not right_contours:
        st.error(f"Slice z={selected_z:.{precision}f} is present only in {left_name}.")
    elif right_contours and not left_contours:
        st.error(f"Slice z={selected_z:.{precision}f} is present only in {right_name}.")
    else:
        st.info(
            f"Slice z={selected_z:.{precision}f} has contours in both files: "
            f"{len(left_contours)} vs {len(right_contours)}"
        )

    if only_left_contours or only_right_contours:
        mismatch_notes: list[str] = []
        if only_left_contours:
            mismatch_notes.append(f"Contours only in {left_name}: {len(only_left_contours)}")
        if only_right_contours:
            mismatch_notes.append(f"Contours only in {right_name}: {len(only_right_contours)}")
        st.warning(" | ".join(mismatch_notes))

    fig = go.Figure()
    if left_contours:
        _add_contour_traces_2d(
            fig,
            contours=left_contours,
            side_name=left_name,
            line_color="#d62728",
            line_dash="solid",
        )
    if right_contours:
        _add_contour_traces_2d(
            fig,
            contours=right_contours,
            side_name=right_name,
            line_color="#2ca02c",
            line_dash="dot",
        )

    if not fig.data:
        st.info("No contours to display for this slice.")
        return

    fig.update_layout(
        title=f"Axial Contour View at z={selected_z:.{precision}f} mm",
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        xaxis={"fixedrange": False},
        yaxis={"scaleanchor": "x", "scaleratio": 1, "fixedrange": False},
        dragmode="zoom",
        margin={"l": 10, "r": 10, "t": 48, "b": 10},
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "scrollZoom": True,
            "displaylogo": False,
            "modeBarButtonsToAdd": ["zoom2d", "pan2d", "resetScale2d"],
        },
    )

    with st.expander("Slice contour details", expanded=False):
        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown(f"**{left_name}**")
            if not left_contours:
                st.caption("No contours on this slice.")
            else:
                for c in left_contours:
                    pts = c["points"]
                    st.write(
                        f"{c['contour_label']}: {len(pts)} points | "
                        f"start=({pts[0][0]:.{precision}f}, {pts[0][1]:.{precision}f}) | "
                        f"end=({pts[-1][0]:.{precision}f}, {pts[-1][1]:.{precision}f})"
                    )


def render_contour_detail_text_view(
    *,
    left_name: str,
    right_name: str,
    left_raw: dict[str, Any],
    right_raw: dict[str, Any],
    precision: int = 4,
) -> None:
    left_contours = select_component(left_raw, "contours")
    right_contours = select_component(right_raw, "contours")
    left_slices, right_slices = get_contour_slices_structured(left_contours, right_contours, precision=precision)
    all_slices = sorted(set(left_slices.keys()) | set(right_slices.keys()))

    if not all_slices:
        st.warning("No contour data found in either file.")
        return

    target_z = st.session_state.contour_detail_target_slice_z
    if target_z is not None:
        aligned_target = _nearest_slice_value(all_slices, target_z)
        if aligned_target is not None:
            st.session_state.contour_detail_slice_select = aligned_target
        st.session_state.contour_detail_target_slice_z = None
    elif st.session_state.get("contour_detail_slice_select") not in all_slices:
        st.session_state.contour_detail_slice_select = all_slices[0]

    nav_col_1, nav_col_2, nav_col_3 = st.columns(3)
    with nav_col_1:
        if st.button("Previous Slice", key="detail_prev_slice"):
            st.session_state.contour_detail_slice_select = _step_slice_value(
                all_slices,
                st.session_state.get("contour_detail_slice_select", all_slices[0]),
                -1,
            )
            st.rerun()
    with nav_col_2:
        if st.button("Next Slice", key="detail_next_slice"):
            st.session_state.contour_detail_slice_select = _step_slice_value(
                all_slices,
                st.session_state.get("contour_detail_slice_select", all_slices[0]),
                1,
            )
            st.rerun()
    with nav_col_3:
        if st.button("Sync To Visual Comparison", key="detail_back_to_visual"):
            current_z = st.session_state.get("contour_detail_slice_select", all_slices[0])
            st.session_state.axial_target_slice_z = current_z
            st.session_state.pair_focus_slice_z = current_z
            st.session_state.pair_component = "contours"
            st.session_state.pair_diff_requested = True
            st.session_state.pair_diff_sig = "force"
            st.info(f"Slice z={current_z} synced. Switch to the Diff tab.")
            st.rerun()

    selected_z = st.select_slider(
        "Contour slice z (mm)",
        options=all_slices,
        format_func=lambda z: f"{z:.{precision}f}",
        value=st.session_state.get("contour_detail_slice_select", all_slices[0]),
        key="contour_detail_slice_select",
    )
    st.session_state.pair_focus_slice_z = selected_z

    left_rois = left_slices.get(selected_z, {})
    right_rois = right_slices.get(selected_z, {})

    st.markdown(f"### Contour Text Comparison At z={selected_z:.{precision}f}")
    col_1, col_2 = st.columns(2)
    with col_1:
        st.markdown(f"**{left_name}**")
        st.code(_format_slice_rois_text(left_rois, precision), language="text")
    with col_2:
        st.markdown(f"**{right_name}**")
        st.code(_format_slice_rois_text(right_rois, precision), language="text")

    left_text = _format_slice_rois_text(left_rois, precision)
    right_text = _format_slice_rois_text(right_rois, precision)
    diff_text = unified_diff_text(left_text, right_text, left_name, right_name)
    st.markdown("#### Slice Unified Diff")
    if diff_text.strip():
        st.code(diff_text, language="diff")
    else:
        st.success("No textual differences on this slice.")


def render_diff_panel(
    *,
    left_name: str,
    right_name: str,
    left_raw: dict[str, Any],
    right_raw: dict[str, Any],
    component: str,
    precision: int,
    keep_volatile: bool,
    allow_rich_view: bool,
    max_rich_chars: int,
    max_rich_lines: int,
    key_prefix: str,
) -> None:
    ignore_prefixes: set[str] = set()
    if not keep_volatile:
        ignore_prefixes.update(DEFAULT_VOLATILE_TAG_PREFIXES)

    # Get selected components
    left_selected = select_component(left_raw, component)
    right_selected = select_component(right_raw, component)

    # Check if components are identical
    if left_selected == right_selected:
        st.info(f"**{component.capitalize()} components are identical** — no differences to compare.")
        col1, col2 = st.columns(2)
        with col1:
            left_json_text = pretty_json_text(left_selected)
            st.download_button(
                "Download left JSON",
                data=left_json_text,
                file_name=f"{Path(left_name).stem}.{component}.json",
                mime="application/json",
                key=f"{key_prefix}_dl_left",
            )
        with col2:
            right_json_text = pretty_json_text(right_selected)
            st.download_button(
                "Download right JSON",
                data=right_json_text,
                file_name=f"{Path(right_name).stem}.{component}.json",
                mime="application/json",
                key=f"{key_prefix}_dl_right",
            )
        return

    # Special handling for contour diffing with two-panel layout
    if component == "contours":
        left_slices, right_slices = get_contour_slices_structured(left_selected, right_selected, precision=precision)

        col1, col2, col3 = st.columns(3)
        with col1:
            left_json_text = pretty_json_text(left_selected)
            st.download_button(
                "Download left JSON",
                data=left_json_text,
                file_name=f"{Path(left_name).stem}.{component}.json",
                mime="application/json",
                key=f"{key_prefix}_dl_left",
            )
        with col2:
            right_json_text = pretty_json_text(right_selected)
            st.download_button(
                "Download right JSON",
                data=right_json_text,
                file_name=f"{Path(right_name).stem}.{component}.json",
                mime="application/json",
                key=f"{key_prefix}_dl_right",
            )
        with col3:
            diff_text = contour_diff_text(left_selected, right_selected, left_name, right_name, precision=precision)
            st.download_button(
                "Download contour diff",
                data=diff_text,
                file_name=f"{Path(left_name).stem}__{Path(right_name).stem}.{component}.diff",
                mime="text/plain",
                key=f"{key_prefix}_dl_diff",
            )

        st.markdown("### Contour Diff View (by Slice Plane)")
        st.caption(
            "Contour points are compared by slice plane (z-coordinate). "
            "Left panel shows the first file, right panel shows the second file."
        )

        if not left_slices and not right_slices:
            st.warning("No contour data found in either file.")
            return

        global_tolerance = st.slider(
            "Correspondence tolerance (mm)",
            min_value=0.1,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key=f"{key_prefix}_corr_tol",
        )

        filter_col_1, filter_col_2, filter_col_3, filter_col_4 = st.columns(4)
        with filter_col_1:
            hide_identical_slices = st.checkbox(
                "Hide identical slices",
                value=False,
                key=f"{key_prefix}_hide_identical_slices",
            )
        with filter_col_2:
            max_dice_filter = st.slider(
                "Max dice to include",
                min_value=0.0,
                max_value=1.0,
                value=1.0,
                step=0.01,
                key=f"{key_prefix}_max_dice",
            )
        with filter_col_3:
            min_mismatch_filter = st.number_input(
                "Min mismatch",
                min_value=0,
                value=0,
                step=1,
                key=f"{key_prefix}_min_mismatch",
            )
        with filter_col_4:
            sort_mode = st.selectbox(
                "Sort slices",
                options=[
                    "Worst dice first",
                    "Best dice first",
                    "Highest mismatch first",
                    "Slice z ascending",
                    "Slice z descending",
                ],
                index=0,
                key=f"{key_prefix}_slice_sort",
            )

        # Build slice metrics first, then filter/sort.
        all_z_coords = sorted(set(left_slices.keys()) | set(right_slices.keys()))
        slice_rows: list[dict[str, Any]] = []
        for z in all_z_coords:
            left_rois = left_slices.get(z, {})
            right_rois = right_slices.get(z, {})
            metrics = _slice_match_metrics(left_rois, right_rois, global_tolerance)
            slice_rows.append(
                {
                    "z": z,
                    "left_rois": left_rois,
                    "right_rois": right_rois,
                    "left_count": metrics["left_count"],
                    "right_count": metrics["right_count"],
                    "dice": metrics["dice"],
                    "mismatch_count": metrics["mismatch_count"],
                    "count_delta": metrics["count_delta"],
                    "identical_slice": metrics["identical_slice"],
                    "left_roi_names": metrics["left_roi_names"],
                    "right_roi_names": metrics["right_roi_names"],
                    "all_roi_names": metrics["all_roi_names"],
                }
            )

        filtered_rows = [
            row
            for row in slice_rows
            if row["dice"] <= max_dice_filter
            and row["mismatch_count"] >= min_mismatch_filter
            and (not hide_identical_slices or not row["identical_slice"])
        ]

        if sort_mode == "Worst dice first":
            filtered_rows.sort(key=lambda r: (r["dice"], -r["mismatch_count"], r["z"]))
        elif sort_mode == "Best dice first":
            filtered_rows.sort(key=lambda r: (-r["dice"], -r["mismatch_count"], r["z"]))
        elif sort_mode == "Highest mismatch first":
            filtered_rows.sort(key=lambda r: (-r["mismatch_count"], r["dice"], r["z"]))
        elif sort_mode == "Slice z descending":
            filtered_rows.sort(key=lambda r: r["z"], reverse=True)
        else:
            filtered_rows.sort(key=lambda r: r["z"])

        st.caption(
            f"Showing {len(filtered_rows)} / {len(slice_rows)} slices after filters "
            f"(tolerance={global_tolerance:.1f} mm)."
        )

        if not filtered_rows:
            st.info("No slices match the current filters.")
            return

        focus_slice = _nearest_slice_value(
            [float(row["z"]) for row in filtered_rows],
            st.session_state.pair_focus_slice_z,
        )
        if focus_slice is not None:
            st.caption(f"Focused slice: z={focus_slice:.{precision}f}")

        # Render two-panel layout for each filtered slice
        for row in filtered_rows:
            z = row["z"]
            left_rois = row["left_rois"]
            right_rois = row["right_rois"]
            left_count = row["left_count"]
            right_count = row["right_count"]
            mismatch_count = row["mismatch_count"]
            count_delta = row["count_delta"]
            dice = row["dice"]
            left_roi_names = row["left_roi_names"]
            right_roi_names = row["right_roi_names"]
            all_roi_names = row["all_roi_names"]
            identical_slice = row["identical_slice"]

            header = (
                f"Slice z={z:.{precision}f} | L={left_count} R={right_count} "
                f"| delta={count_delta} | mismatch={mismatch_count} | dice={dice:.3f}"
            )

            expanded = focus_slice is not None and z == focus_slice
            with st.expander(header, expanded=expanded):
                if st.button(
                    "Open This Slice In 2D Axial View",
                    key=f"{key_prefix}_open_axial_{_safe_key_fragment(f'{z:.{precision}f}')}",
                ):
                    st.session_state.axial_target_slice_z = z
                    st.session_state.pair_focus_slice_z = z

                if identical_slice:
                    st.caption("This slice is identical in both files.")

                # Two-column layout for this slice
                left_col, right_col = st.columns(2)

                # LEFT PANEL
                with left_col:
                    st.markdown(f"**{left_name}**")
                    if z not in left_slices:
                        st.caption("No contour data on this slice.")
                    else:
                        for roi_name in sorted(left_rois.keys()):
                            points = sorted(left_rois[roi_name])
                            is_different = roi_name not in right_rois or sorted(right_rois[roi_name]) != points
                            marker = "[diff]" if is_different else "[same]"
                            st.markdown(f"{marker} *{roi_name}* ({len(points)} points)")
                            points_text = "\n".join(
                                [f"({x:.{precision}f}, {y:.{precision}f}, {z:.{precision}f})" for x, y, z in points]
                            )
                            st.code(points_text, language="")
                        
                        # Show ROIs that only exist in right
                        for roi_name in sorted(right_roi_names - left_roi_names):
                            st.markdown(f"⊘ *{roi_name}* (only in {right_name})")

                # RIGHT PANEL
                with right_col:
                    st.markdown(f"**{right_name}**")
                    if z not in right_slices:
                        st.caption("No contour data on this slice.")
                    else:
                        for roi_name in sorted(right_rois.keys()):
                            points = sorted(right_rois[roi_name])
                            is_different = roi_name not in left_rois or sorted(left_rois[roi_name]) != points
                            marker = "[diff]" if is_different else "[same]"
                            st.markdown(f"{marker} *{roi_name}* ({len(points)} points)")
                            points_text = "\n".join(
                                [f"({x:.{precision}f}, {y:.{precision}f}, {z:.{precision}f})" for x, y, z in points]
                            )
                            st.code(points_text, language="")
                        
                        # Show ROIs that only exist in left
                        for roi_name in sorted(left_roi_names - right_roi_names):
                            st.markdown(f"⊘ *{roi_name}* (only in {left_name})")

                st.markdown("#### Correspondence Explorer")
                common_rois = sorted(left_roi_names & right_roi_names)
                if not common_rois:
                    st.caption("No common ROIs on this slice to match.")
                else:
                    z_key = _safe_key_fragment(f"{z:.{precision}f}")
                    roi_choice = st.selectbox(
                        "ROI",
                        options=common_rois,
                        key=f"{key_prefix}_roi_{z_key}",
                    )
                    left_roi_points = sorted(left_rois.get(roi_choice, []))
                    right_roi_points = sorted(right_rois.get(roi_choice, []))
                    roi_matches = _greedy_point_matches(left_roi_points, right_roi_points)

                    if not roi_matches:
                        st.caption("No points available for correspondence on this ROI.")
                    else:
                        option_labels: list[str] = []
                        default_labels: list[str] = []
                        for li, ri, dist in roi_matches:
                            label = (
                                f"L{li + 1} {left_roi_points[li]} <-> "
                                f"R{ri + 1} {right_roi_points[ri]} | d={dist:.3f}"
                            )
                            option_labels.append(label)
                            if dist <= global_tolerance:
                                default_labels.append(label)

                        selected_pairs = st.multiselect(
                            "Select correspondences",
                            options=option_labels,
                            default=default_labels,
                            key=f"{key_prefix}_corr_{z_key}_{_safe_key_fragment(roi_choice)}",
                        )
                        st.caption(
                            f"Auto-suggested correspondences within tolerance: {len(default_labels)} / {len(roi_matches)}"
                        )
                        if selected_pairs:
                            st.code("\n".join(selected_pairs), language="text")
                        else:
                            st.caption("No correspondences selected.")

        return

    # Standard text-based diffing for other components
    left_norm = normalize_value(left_selected, precision, ignore_prefixes)
    right_norm = normalize_value(right_selected, precision, ignore_prefixes)

    left_text = pretty_json_text(left_norm)
    right_text = pretty_json_text(right_norm)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Download left JSON",
            data=left_text,
            file_name=f"{Path(left_name).stem}.{component}.json",
            mime="application/json",
            key=f"{key_prefix}_dl_left",
        )
    with col2:
        st.download_button(
            "Download right JSON",
            data=right_text,
            file_name=f"{Path(right_name).stem}.{component}.json",
            mime="application/json",
            key=f"{key_prefix}_dl_right",
        )

    diff_text = unified_diff_text(left_text, right_text, left_name, right_name)
    with col3:
        st.download_button(
            "Download unified diff",
            data=diff_text,
            file_name=f"{Path(left_name).stem}__{Path(right_name).stem}.{component}.diff",
            mime="text/plain",
            key=f"{key_prefix}_dl_diff",
        )

    use_unified_only, reason = should_use_unified_only(
        left_text,
        right_text,
        allow_rich_view=allow_rich_view,
        max_rich_chars=max_rich_chars,
        max_rich_lines=max_rich_lines,
    )

    st.markdown("### Diff View")
    if use_unified_only:
        if reason:
            st.caption(reason)
        if diff_text.strip():
            st.code(diff_text, language="diff")
        else:
            st.success("No differences after normalization and filtering.")
    elif diff_viewer is not None:
        diff_viewer(left_text, right_text, split_view=True)
    else:
        if diff_text.strip():
            st.code(diff_text, language="diff")
        else:
            st.success("No differences after normalization and filtering.")


def should_use_unified_only(
    left_text: str,
    right_text: str,
    *,
    allow_rich_view: bool,
    max_rich_chars: int,
    max_rich_lines: int,
) -> tuple[bool, str]:
    if not allow_rich_view:
        return True, "Showing unified diff text only for this mode."

    total_chars = len(left_text) + len(right_text)
    total_lines = left_text.count("\n") + right_text.count("\n") + 2
    if total_chars > max_rich_chars or total_lines > max_rich_lines:
        return (
            True,
            (
                "Large comparison detected. Showing unified diff text for faster loading. "
                f"(chars={total_chars:,}, lines={total_lines:,})"
            ),
        )

    return False, ""


def _extract_xyz_points(value: Any) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []

    if isinstance(value, list):
        if len(value) == 3 and all(isinstance(v, (int, float)) for v in value):
            points.append((float(value[0]), float(value[1]), float(value[2])))
        else:
            for item in value:
                points.extend(_extract_xyz_points(item))

    return points


def extract_contour_points(rtss_json: dict[str, Any]) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if "ContourData" in key:
                    points.extend(_extract_xyz_points(value))
                else:
                    walk(value)
            return

        if isinstance(node, list):
            for item in node:
                walk(item)

    walk(rtss_json)
    return points


def _as_float_list(value: Any, expected_len: int | None = None) -> list[float] | None:
    if not isinstance(value, list):
        return None
    try:
        out = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if expected_len is not None and len(out) != expected_len:
        return None
    return out


def _as_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _find_first_keyword_value(node: Any, keyword: str) -> Any | None:
    if isinstance(node, dict):
        for key, value in node.items():
            if keyword in key:
                return value
            found = _find_first_keyword_value(value, keyword)
            if found is not None:
                return found
    elif isinstance(node, list):
        for item in node:
            found = _find_first_keyword_value(item, keyword)
            if found is not None:
                return found
    return None


def _vadd(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _vscale(v: tuple[float, float, float], s: float) -> tuple[float, float, float]:
    return (v[0] * s, v[1] * s, v[2] * s)


def _vnorm(v: tuple[float, float, float]) -> float:
    return sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def _vunit(v: tuple[float, float, float]) -> tuple[float, float, float] | None:
    n = _vnorm(v)
    if n == 0:
        return None
    return (v[0] / n, v[1] / n, v[2] / n)


def _cross(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _bounds_from_points(points: list[tuple[float, float, float]]) -> dict[str, float] | None:
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    return {
        "x_min": min(xs),
        "x_max": max(xs),
        "y_min": min(ys),
        "y_max": max(ys),
        "z_min": min(zs),
        "z_max": max(zs),
    }


def _extract_volume_bounds_from_rtss(rtss_json: dict[str, Any]) -> tuple[dict[str, float] | None, str]:
    origin_raw = _find_first_keyword_value(rtss_json, "ImagePositionPatient")
    orient_raw = _find_first_keyword_value(rtss_json, "ImageOrientationPatient")
    pixel_spacing_raw = _find_first_keyword_value(rtss_json, "PixelSpacing")
    rows_raw = _find_first_keyword_value(rtss_json, "Rows")
    cols_raw = _find_first_keyword_value(rtss_json, "Columns")
    frames_raw = _find_first_keyword_value(rtss_json, "NumberOfFrames")
    spacing_between_raw = _find_first_keyword_value(rtss_json, "SpacingBetweenSlices")
    slice_thickness_raw = _find_first_keyword_value(rtss_json, "SliceThickness")

    origin = _as_float_list(origin_raw, expected_len=3)
    orient = _as_float_list(orient_raw, expected_len=6)
    pixel_spacing = _as_float_list(pixel_spacing_raw, expected_len=2)
    rows = _as_int(rows_raw)
    cols = _as_int(cols_raw)
    frames = _as_int(frames_raw)

    if frames is None:
        frames = 1

    slice_spacing = None
    if spacing_between_raw is not None:
        try:
            slice_spacing = float(spacing_between_raw)
        except (TypeError, ValueError):
            slice_spacing = None
    if slice_spacing is None and slice_thickness_raw is not None:
        try:
            slice_spacing = float(slice_thickness_raw)
        except (TypeError, ValueError):
            slice_spacing = None
    if slice_spacing is None:
        slice_spacing = 1.0

    if origin is None or orient is None or pixel_spacing is None or rows is None or cols is None:
        return (
            None,
            "Volume geometry metadata is incomplete in RTSS. Falling back to contour-point bounds.",
        )

    row_dir = _vunit((orient[0], orient[1], orient[2]))
    col_dir = _vunit((orient[3], orient[4], orient[5]))
    if row_dir is None or col_dir is None:
        return (
            None,
            "Image orientation metadata is invalid. Falling back to contour-point bounds.",
        )

    normal = _vunit(_cross(row_dir, col_dir))
    if normal is None:
        return (
            None,
            "Unable to derive slice-normal direction from orientation. Falling back to contour-point bounds.",
        )

    row_spacing, col_spacing = pixel_spacing[0], pixel_spacing[1]
    row_extent = max(rows - 1, 0) * row_spacing
    col_extent = max(cols - 1, 0) * col_spacing
    depth_extent = max(frames - 1, 0) * slice_spacing
    origin_xyz = (origin[0], origin[1], origin[2])

    corners: list[tuple[float, float, float]] = []
    for r in (0.0, row_extent):
        for c in (0.0, col_extent):
            for d in (0.0, depth_extent):
                corner = origin_xyz
                corner = _vadd(corner, _vscale(row_dir, r))
                corner = _vadd(corner, _vscale(col_dir, c))
                corner = _vadd(corner, _vscale(normal, d))
                corners.append(corner)

    bounds = _bounds_from_points(corners)
    if bounds is None:
        return (
            None,
            "Unable to derive volume bounds from RTSS metadata. Falling back to contour-point bounds.",
        )

    return bounds, "Volume extents derived from RTSS geometry metadata."


def _merge_bounds(a: dict[str, float] | None, b: dict[str, float] | None) -> dict[str, float] | None:
    if a is None:
        return b
    if b is None:
        return a
    return {
        "x_min": min(a["x_min"], b["x_min"]),
        "x_max": max(a["x_max"], b["x_max"]),
        "y_min": min(a["y_min"], b["y_min"]),
        "y_max": max(a["y_max"], b["y_max"]),
        "z_min": min(a["z_min"], b["z_min"]),
        "z_max": max(a["z_max"], b["z_max"]),
    }


def _add_bounds_box(fig: go.Figure, bounds: dict[str, float], color: str, name: str) -> None:
    x0, x1 = bounds["x_min"], bounds["x_max"]
    y0, y1 = bounds["y_min"], bounds["y_max"]
    z0, z1 = bounds["z_min"], bounds["z_max"]

    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    for idx, (a, b) in enumerate(edges):
        xa, ya, za = corners[a]
        xb, yb, zb = corners[b]
        fig.add_trace(
            go.Scatter3d(
                x=[xa, xb],
                y=[ya, yb],
                z=[za, zb],
                mode="lines",
                line={"width": 2, "color": color},
                name=name if idx == 0 else name,
                legendgroup=name,
                showlegend=(idx == 0),
                opacity=0.35,
            )
        )


def _add_axes_markers(fig: go.Figure, bounds: dict[str, float]) -> None:
    origin = (bounds["x_min"], bounds["y_min"], bounds["z_min"])
    x_range = max(bounds["x_max"] - bounds["x_min"], 1.0)
    y_range = max(bounds["y_max"] - bounds["y_min"], 1.0)
    z_range = max(bounds["z_max"] - bounds["z_min"], 1.0)
    axis_len = max(x_range, y_range, z_range) * 0.15

    x_end = (origin[0] + axis_len, origin[1], origin[2])
    y_end = (origin[0], origin[1] + axis_len, origin[2])
    z_end = (origin[0], origin[1], origin[2] + axis_len)

    fig.add_trace(
        go.Scatter3d(
            x=[origin[0], x_end[0]],
            y=[origin[1], x_end[1]],
            z=[origin[2], x_end[2]],
            mode="lines+markers+text",
            line={"width": 5, "color": "#1f77b4"},
            marker={"size": [3, 5], "color": "#1f77b4"},
            text=["", "X+"],
            textposition="top center",
            name="X axis",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[origin[0], y_end[0]],
            y=[origin[1], y_end[1]],
            z=[origin[2], y_end[2]],
            mode="lines+markers+text",
            line={"width": 5, "color": "#ff7f0e"},
            marker={"size": [3, 5], "color": "#ff7f0e"},
            text=["", "Y+"],
            textposition="top center",
            name="Y axis",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[origin[0], z_end[0]],
            y=[origin[1], z_end[1]],
            z=[origin[2], z_end[2]],
            mode="lines+markers+text",
            line={"width": 5, "color": "#2ca02c"},
            marker={"size": [3, 5], "color": "#2ca02c"},
            text=["", "Z+"],
            textposition="top center",
            name="Z axis",
        )
    )


def render_contour_point_cloud(
    *,
    left_name: str,
    right_name: str,
    left_raw: dict[str, Any],
    right_raw: dict[str, Any],
) -> None:
    left_points = extract_contour_points(left_raw)
    right_points = extract_contour_points(right_raw)
    left_volume_bounds, left_volume_msg = _extract_volume_bounds_from_rtss(left_raw)
    right_volume_bounds, right_volume_msg = _extract_volume_bounds_from_rtss(right_raw)

    c1, c2 = st.columns(2)
    with c1:
        st.metric(f"{left_name} points", f"{len(left_points):,}")
    with c2:
        st.metric(f"{right_name} points", f"{len(right_points):,}")

    if not left_points and not right_points:
        st.warning("No contour points found in the selected RTSS files.")
        return

    contour_bounds = _bounds_from_points([*left_points, *right_points])
    volume_bounds = _merge_bounds(left_volume_bounds, right_volume_bounds)
    display_bounds = volume_bounds or contour_bounds
    if display_bounds is None:
        st.warning("Unable to determine display bounds for the contour plot.")
        return

    if volume_bounds is not None:
        st.caption(
            "Plot extents are aligned to RTSS-derived imaging volume bounds. "
            f"Left: {left_volume_msg} Right: {right_volume_msg}"
        )
    else:
        st.caption(
            "Imaging volume bounds were not available in RTSS metadata. "
            "Using contour-point bounds instead."
        )

    control_1, control_2, control_3 = st.columns(3)
    with control_1:
        show_left = st.checkbox(f"Show {left_name}", value=True, key="pc_show_left")
    with control_2:
        show_right = st.checkbox(f"Show {right_name}", value=True, key="pc_show_right")
    with control_3:
        point_size = st.slider("Point size", min_value=1, max_value=8, value=3, key="pc_point_size")

    fig = go.Figure()

    if show_left and left_points:
        lx, ly, lz = zip(*left_points)
        fig.add_trace(
            go.Scatter3d(
                x=lx,
                y=ly,
                z=lz,
                mode="markers",
                marker={"size": point_size, "color": "red", "opacity": 0.7},
                name=f"First RTSS: {left_name}",
            )
        )

    if show_right and right_points:
        rx, ry, rz = zip(*right_points)
        fig.add_trace(
            go.Scatter3d(
                x=rx,
                y=ry,
                z=rz,
                mode="markers",
                marker={"size": point_size, "color": "green", "opacity": 0.7},
                name=f"Second RTSS: {right_name}",
            )
        )

    if not fig.data:
        st.info("Both point groups are hidden. Turn at least one group back on.")
        return

    _add_bounds_box(fig, display_bounds, color="#555555", name="Display bounds")
    _add_axes_markers(fig, display_bounds)

    x_span = max(display_bounds["x_max"] - display_bounds["x_min"], 1.0)
    y_span = max(display_bounds["y_max"] - display_bounds["y_min"], 1.0)
    z_span = max(display_bounds["z_max"] - display_bounds["z_min"], 1.0)
    pad_ratio = 0.03
    x_pad = x_span * pad_ratio
    y_pad = y_span * pad_ratio
    z_pad = z_span * pad_ratio

    fig.update_layout(
        title="RTSS Contour Point Cloud Diff",
        scene={
            "xaxis_title": "X (mm, patient Left +)",
            "yaxis_title": "Y (mm, patient Posterior +)",
            "zaxis_title": "Z (mm, patient Superior +)",
            "xaxis": {"range": [display_bounds["x_min"] - x_pad, display_bounds["x_max"] + x_pad]},
            "yaxis": {"range": [display_bounds["y_min"] - y_pad, display_bounds["y_max"] + y_pad]},
            "zaxis": {"range": [display_bounds["z_min"] - z_pad, display_bounds["z_max"] + z_pad]},
        },
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        legend={"orientation": "h", "y": 1.02, "x": 0},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Coordinate system uses DICOM patient coordinates (LPS): X increases toward patient Left, "
        "Y increases toward Posterior, and Z increases toward Superior. "
        "Use drag to rotate, scroll to zoom, and legend clicks to show or hide each group."
    )


def main() -> None:
    st.set_page_config(page_title="RTSS Diff Viewer", layout="wide")
    ensure_state()

    # ------------------------------------------------------------------ Sidebar
    with st.sidebar:
        st.markdown("## RTSS Diff Viewer")
        st.caption("Compare RTSS DICOM files using textual diffs and 2D axial contour overlays.")
        st.divider()

        app_mode = st.radio(
            "View",
            options=["Instructions", "Pair Mode", "Batch Compare"],
            label_visibility="collapsed",
        )

        if app_mode != "Instructions":
            st.divider()
            st.markdown("**Settings**")
            mode_prefix = "pair" if app_mode == "Pair Mode" else "batch"
            precision = st.slider(
                "Float precision",
                min_value=2,
                max_value=10,
                value=6,
                key=f"{mode_prefix}_precision",
                help="Number of decimal places shown for contour point coordinates.",
            )
            keep_volatile = st.checkbox(
                "Keep volatile tags",
                value=False,
                key=f"{mode_prefix}_keep_volatile",
                help=(
                    "When off, UID and timestamp tags that change on every export "
                    "are excluded from the metadata diff."
                ),
            )

    # -------------------------------------------------------- Instructions view
    if app_mode == "Instructions":
        st.markdown("## Quick-Start Guide")
        st.info(
            "This tool compares two RTSS DICOM files side-by-side. "
            "It breaks each file into four components — **metadata**, **structures**, **references**, and **contours** — "
            "and shows exactly what changed between them."
        )

        st.markdown("---")
        st.markdown("### Modes at a Glance")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(
                "#### Pair Mode\n"
                "Compare exactly **two** RTSS files in full detail.\n\n"
                "Best for: reviewing a before/after pair, checking a re-plan, "
                "or validating an export against a reference."
            )
        with col_b:
            st.markdown(
                "#### Batch Compare\n"
                "Upload **many** files at once and pick any two to compare.\n\n"
                "Best for: screening a set of variants, comparing multiple "
                "plan versions, or running QA across a cohort."
            )

        st.markdown("---")
        st.markdown("### Pair Mode — Step by Step")
        st.markdown(
            "**1. Upload tab**\n"
            "- Upload the *Left* (reference) and *Right* (modified) RTSS `.dcm` files.\n"
            "- Click **Convert pair** — the files are parsed once and cached for the session.\n\n"
            "**2. Diff tab — choose a component**\n\n"
            "| Component | What it contains | Diff method |\n"
            "|-----------|-----------------|-------------|\n"
            "| `metadata` | DICOM package-level tags (e.g. SOP UID, study date) | Unified text diff |\n"
            "| `structures` | ROI names, numbers, and type codes | Unified text diff |\n"
            "| `references` | Referenced frame-of-reference sequences | Unified text diff |\n"
            "| `contours` | 2D point lists grouped by axial slice (z-plane) | Visual overlay + per-slice diff |\n\n"
            "For **metadata / structures / references**: a red/green unified diff is shown immediately. "
            "Identical components show a confirmation banner.\n\n"
            "For **contours**: an axial overlay canvas renders both files' contours in the same colour scheme. "
            "Use the z-slice slider or Previous / Next buttons to step through slices. "
            "Click **Sync To Contour Detail** on any slice to sync the selected slice to the Contour Detail tab.\n\n"
            "**3. Contour Detail tab**\n"
            "- Shows the raw point list for the selected slice in both files, side-by-side.\n"
            "- A unified diff below highlights added/removed points precisely.\n"
            "- Use **Previous Slice / Next Slice** to walk through changes.\n"
            "- **Sync To Visual Comparison** syncs the slice back to the Diff tab.\n\n"
            "**4. Downloads**\n"
            "- On any diff panel, use the **Download left JSON** / **Download right JSON** buttons "
            "to export the parsed component for offline inspection."
        )

        st.markdown("---")
        st.markdown("### Batch Compare — Step by Step")
        st.markdown(
            "1. In the Upload tab, upload all RTSS `.dcm` files you want to screen.\n"
            "2. Click **Convert uploaded set** — each file is parsed once.\n"
            "3. In the Compare tab, choose a **Left** and **Right** file from the dropdowns.\n"
            "4. Select the **component** to compare (same four options as Pair Mode).\n"
            "5. Review the diff or visual overlay; download JSON outputs if needed.\n\n"
            "> **Note:** Batch Compare does not include a Contour Detail view. "
            "For detailed point-level inspection, load the two files of interest in Pair Mode."
        )

        st.markdown("---")
        st.markdown("### Settings")
        st.markdown(
            "Settings are available in the sidebar when Pair Mode or Batch Compare is active.\n\n"
            "**Float precision** — controls how many decimal places are shown for contour "
            "point coordinates. Higher precision reveals sub-millimetre differences; lower precision "
            "reduces noise from floating-point rounding.\n\n"
            "**Keep volatile tags** — when off (default), tags that change on every export "
            "(e.g. SOP Instance UID, timestamps) are stripped before diffing so they do not clutter "
            "the metadata diff. Enable to include every tag."
        )

        st.markdown("---")
        st.markdown("### Common Workflows")
        st.markdown(
            "**Checking whether a re-export changed anything meaningful**\n"
            "1. Pair Mode — upload original (Left) and re-export (Right).\n"
            "2. Check `metadata` with *Keep volatile tags* off — expect no diff.\n"
            "3. Check `structures` and `references` — expect no diff.\n"
            "4. Check `contours` — step through slices; diverging overlays indicate geometric change.\n\n"
            "**Finding which slice changed after a re-plan**\n"
            "1. Pair Mode — upload pre-plan (Left) and post-plan (Right).\n"
            "2. In the Diff tab select `contours`; step through slices until the overlay diverges.\n"
            "3. Click **Sync To Contour Detail** and switch to the Contour Detail tab.\n\n"
            "**Screening many files quickly**\n"
            "1. Batch Compare — upload all variants.\n"
            "2. For each pair of interest, select `structures` first — if ROI names differ, "
            "contour comparison may not be meaningful.\n"
            "3. Then compare `contours` for geometric QA."
        )

    # ------------------------------------------------------------- Pair Mode
    elif app_mode == "Pair Mode":
        st.markdown("## Pair Mode")
        tab_upload, tab_diff, tab_detail = st.tabs(["Upload", "Diff", "Contour Detail"])

        with tab_upload:
            left_col, right_col = st.columns(2)
            with left_col:
                left_file = st.file_uploader("Left RTSS (.dcm)", type=["dcm"], key="left_pair")
            with right_col:
                right_file = st.file_uploader("Right RTSS (.dcm)", type=["dcm"], key="right_pair")

            st.divider()
            if st.button("Convert pair", type="primary"):
                if left_file is None or right_file is None:
                    st.warning("Upload both files before converting.")
                else:
                    with st.spinner("Converting files..."):
                        st.session_state.left_json_raw = dcm_bytes_to_json_dict(left_file.getvalue())
                        st.session_state.right_json_raw = dcm_bytes_to_json_dict(right_file.getvalue())
                        st.session_state.left_name = left_file.name
                        st.session_state.right_name = right_file.name
                    st.session_state.pair_diff_requested = False
                    st.session_state.pair_diff_sig = None
                    st.success("Files converted. Switch to the Diff tab to compare.")

            if st.session_state.left_json_raw is not None:
                st.info(
                    f"Loaded: **{st.session_state.left_name}** (left) "
                    f"and **{st.session_state.right_name}** (right)."
                )

        with tab_diff:
            left_raw = st.session_state.left_json_raw
            right_raw = st.session_state.right_json_raw
            if left_raw is None or right_raw is None:
                st.info("Upload and convert a pair of files in the Upload tab first.")
            else:
                component_options = ["metadata", "structures", "references", "contours"]
                col_comp, col_btn2 = st.columns([3, 1])
                with col_comp:
                    component = st.selectbox(
                        "Component",
                        options=component_options,
                        key="pair_component",
                    )
                with col_btn2:
                    st.markdown("<br>", unsafe_allow_html=True)
                    compute_clicked = st.button("Compute Diff", type="primary", key="pair_compute_diff")

                current_sig = (
                    component,
                    precision,
                    keep_volatile,
                    st.session_state.left_name,
                    st.session_state.right_name,
                )
                if compute_clicked:
                    st.session_state.pair_diff_requested = True
                    st.session_state.pair_diff_sig = current_sig

                pair_diff_sig = st.session_state.pair_diff_sig
                should_render_diff = st.session_state.pair_diff_requested and (
                    pair_diff_sig == current_sig or pair_diff_sig == "force"
                )

                if should_render_diff:
                    if component == "contours":
                        render_axial_contour_view(
                            left_name=st.session_state.left_name,
                            right_name=st.session_state.right_name,
                            left_raw=left_raw,
                            right_raw=right_raw,
                            precision=precision,
                        )
                    else:
                        with st.spinner("Computing diff..."):
                            render_diff_panel(
                                left_name=st.session_state.left_name,
                                right_name=st.session_state.right_name,
                                left_raw=left_raw,
                                right_raw=right_raw,
                                component=component,
                                precision=precision,
                                keep_volatile=keep_volatile,
                                allow_rich_view=True,
                                max_rich_chars=400_000,
                                max_rich_lines=5_000,
                                key_prefix="pair",
                            )
                    st.session_state.pair_diff_sig = current_sig
                else:
                    st.caption("Select a component and click Compute Diff to run the comparison.")

        with tab_detail:
            left_raw = st.session_state.left_json_raw
            right_raw = st.session_state.right_json_raw
            if left_raw is None or right_raw is None:
                st.info("Upload and convert a pair of files in the Upload tab first.")
            else:
                with st.spinner("Preparing contour slice text comparison..."):
                    render_contour_detail_text_view(
                        left_name=st.session_state.left_name,
                        right_name=st.session_state.right_name,
                        left_raw=left_raw,
                        right_raw=right_raw,
                        precision=precision,
                    )

    # --------------------------------------------------------- Batch Compare
    else:
        st.markdown("## Batch Compare")
        tab_upload, tab_compare = st.tabs(["Upload", "Compare"])

        with tab_upload:
            files = st.file_uploader(
                "Upload RTSS variant set (.dcm)",
                type=["dcm"],
                accept_multiple_files=True,
                key="batch_upload",
            )
            st.divider()
            if st.button("Convert uploaded set", type="primary", key="batch_convert"):
                if not files:
                    st.warning("Upload at least two RTSS files.")
                else:
                    converted: dict[str, dict[str, Any]] = {}
                    with st.spinner("Converting variant set..."):
                        for idx, file in enumerate(files, start=1):
                            name = file.name
                            if name in converted:
                                name = f"{idx:02d}_{name}"
                            converted[name] = dcm_bytes_to_json_dict(file.getvalue())
                    st.session_state.batch_variants = converted
                    st.success(f"Converted {len(converted)} file(s). Switch to the Compare tab.")

            if st.session_state.batch_variants:
                st.info(f"{len(st.session_state.batch_variants)} file(s) loaded.")

        with tab_compare:
            variants = st.session_state.batch_variants
            if not variants:
                st.info("Upload and convert files in the Upload tab first.")
            else:
                names = sorted(variants.keys())
                if len(names) < 2:
                    st.warning("Need at least two variants.")
                else:
                    col_left, col_right, col_comp = st.columns(3)
                    with col_left:
                        left_name = st.selectbox("Left variant", names, index=0, key="batch_left")
                    with col_right:
                        right_name = st.selectbox(
                            "Right variant",
                            names,
                            index=min(1, len(names) - 1),
                            key="batch_right",
                        )
                    with col_comp:
                        component = st.selectbox(
                            "Component",
                            options=["metadata", "structures", "references", "contours"],
                            index=0,
                            key="batch_component",
                        )

                    if left_name == right_name:
                        st.warning("Select two different variants.")
                    else:
                        render_diff_panel(
                            left_name=left_name,
                            right_name=right_name,
                            left_raw=variants[left_name],
                            right_raw=variants[right_name],
                            component=component,
                            precision=precision,
                            keep_volatile=keep_volatile,
                            allow_rich_view=False,
                            max_rich_chars=0,
                            max_rich_lines=0,
                            key_prefix="batch",
                        )


if __name__ == "__main__":
    main()
