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

    left_selected = select_component(left_raw, component)
    right_selected = select_component(right_raw, component)

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

    st.title("RTSS Diff Viewer")
    st.caption("Compare RTSS DICOM files using textual diffs and 3D contour overlays.")

    app_mode = st.radio(
        "Mode",
        options=["Instructions", "Pair Mode", "Batch Compare"],
        index=0,
        horizontal=True,
    )

    if app_mode == "Instructions":
        st.markdown("### What This App Does")
        st.write("This app compares RTSS DICOM files using text-based and visual workflows.")

        st.markdown("### Pair Mode")
        st.write("Use Pair Mode to compare exactly two RTSS files in detail.")
        st.write("1. Open Pair Mode.")
        st.write("2. In Upload Pair, upload the first and second RTSS .dcm files.")
        st.write("3. Click Convert pair.")
        st.write("4. Open Pair Diff to review JSON differences by component.")
        st.write("5. Open 3D Contour View to compare contour points visually.")
        st.write("6. In 3D view, red points are from the first RTSS and green points are from the second RTSS.")

        st.markdown("### Batch Compare")
        st.write("Use Batch Compare to compare any two files from a larger uploaded set.")
        st.write("1. Open Batch Compare.")
        st.write("2. Upload multiple RTSS .dcm files.")
        st.write("3. Click Convert uploaded set.")
        st.write("4. Pick any two variants from the dropdowns.")
        st.write("5. Review the unified text diff and download outputs if needed.")

        st.markdown("### Tips")
        st.write("- Large comparisons automatically switch to unified text diff for faster loading.")
        st.write("- In 3D view, drag to rotate and scroll to zoom.")

    elif app_mode == "Pair Mode":
        tab_upload, tab_diff, tab_points_3d = st.tabs(
            ["1) Upload Pair", "2) Pair Diff", "3) 3D Contour View"]
        )

        with tab_upload:
            st.info("Step 1 of 3: Upload two RTSS files and click Convert pair.")
            left_col, right_col = st.columns(2)
            with left_col:
                left_file = st.file_uploader("Left RTSS (.dcm)", type=["dcm"], key="left_pair")
            with right_col:
                right_file = st.file_uploader("Right RTSS (.dcm)", type=["dcm"], key="right_pair")

            if st.button("Convert pair", type="primary"):
                if left_file is None or right_file is None:
                    st.warning("Upload both files first.")
                else:
                    with st.spinner("Converting files..."):
                        st.session_state.left_json_raw = dcm_bytes_to_json_dict(left_file.getvalue())
                        st.session_state.right_json_raw = dcm_bytes_to_json_dict(right_file.getvalue())
                        st.session_state.left_name = left_file.name
                        st.session_state.right_name = right_file.name
                    st.success("Pair conversion complete.")
                    st.info("Next step: open tab 2) Pair Diff to review differences.")

        with tab_diff:
            st.info("Step 2 of 3: Review Pair Diff, then move to tab 3) 3D Contour View.")
            component_options = ["all", "metadata", *COMPONENT_KEYS.keys()]
            control_col_1, control_col_2, control_col_3 = st.columns(3)
            with control_col_1:
                component = st.selectbox("Component", options=component_options, index=0, key="pair_component")
            with control_col_2:
                precision = st.slider("Float precision", min_value=2, max_value=10, value=6, key="pair_precision")
            with control_col_3:
                keep_volatile = st.checkbox("Keep volatile UID/time tags", value=False, key="pair_keep_volatile")

            left_raw = st.session_state.left_json_raw
            right_raw = st.session_state.right_json_raw
            if left_raw is None or right_raw is None:
                st.info("Convert a pair in tab 1 first.")
            else:
                st.caption("If files are large, diff generation may take some time while JSON is normalized and compared.")
                with st.spinner("Preparing normalized JSON and building diff view. This may take time for large files..."):
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
                st.info("Next step: open tab 3) 3D Contour View for point cloud comparison.")

        with tab_points_3d:
            st.markdown("### 3D Contour Point Cloud")
            st.info("Step 3 of 3: Inspect contour points in 3D. Red is first RTSS, green is second RTSS.")
            left_raw = st.session_state.left_json_raw
            right_raw = st.session_state.right_json_raw
            if left_raw is None or right_raw is None:
                st.info("Convert a pair in tab 1 first.")
            else:
                st.caption("Contour extraction and 3D plot generation can take time for large RTSS files.")
                with st.spinner("Extracting contour points and rendering interactive 3D view. Please wait..."):
                    render_contour_point_cloud(
                        left_name=st.session_state.left_name,
                        right_name=st.session_state.right_name,
                        left_raw=left_raw,
                        right_raw=right_raw,
                    )

    else:
        st.markdown("### Batch Compare")
        files = st.file_uploader(
            "Upload RTSS variant set (.dcm)",
            type=["dcm"],
            accept_multiple_files=True,
            key="batch_upload",
        )

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
                st.success(f"Converted {len(converted)} file(s).")

        variants = st.session_state.batch_variants
        if not variants:
            st.info("No batch set loaded yet.")
        else:
            names = sorted(variants.keys())
            if len(names) < 2:
                st.warning("Need at least two variants.")
            else:
                component_options = ["all", "metadata", *COMPONENT_KEYS.keys()]
                control_col_1, control_col_2, control_col_3 = st.columns(3)
                with control_col_1:
                    component = st.selectbox("Component", options=component_options, index=0, key="batch_component")
                with control_col_2:
                    precision = st.slider("Float precision", min_value=2, max_value=10, value=6, key="batch_precision")
                with control_col_3:
                    keep_volatile = st.checkbox("Keep volatile UID/time tags", value=False, key="batch_keep_volatile")

                left_col, right_col = st.columns(2)
                with left_col:
                    left_name = st.selectbox("Left variant", names, index=0, key="batch_left")
                with right_col:
                    right_name = st.selectbox("Right variant", names, index=1, key="batch_right")

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
