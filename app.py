#!/usr/bin/env python3
"""Streamlit RTSS diff viewer app."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any

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

    st.markdown("### Diff View")
    if diff_viewer is not None:
        diff_viewer(left_text, right_text, split_view=True)
    else:
        if diff_text.strip():
            st.code(diff_text, language="diff")
        else:
            st.success("No differences after normalization and filtering.")


def main() -> None:
    st.set_page_config(page_title="RTSS Diff Viewer", layout="wide")
    ensure_state()

    st.title("RTSS Diff Viewer")
    st.caption("Upload RTSS .dcm files, convert to JSON, and compare with git-like diffs.")

    with st.sidebar:
        st.header("Diff Controls")
        component_options = ["all", "metadata", *COMPONENT_KEYS.keys()]
        component = st.selectbox("Component", options=component_options, index=0)
        precision = st.slider("Float precision", min_value=2, max_value=10, value=6)
        keep_volatile = st.checkbox("Keep volatile UID/time tags", value=False)

    tab_upload, tab_diff, tab_batch = st.tabs(
        ["1) Upload Pair", "2) Pair Diff", "3) Batch Compare"]
    )

    with tab_upload:
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

    with tab_diff:
        left_raw = st.session_state.left_json_raw
        right_raw = st.session_state.right_json_raw
        if left_raw is None or right_raw is None:
            st.info("Convert a pair in tab 1 first.")
        else:
            render_diff_panel(
                left_name=st.session_state.left_name,
                right_name=st.session_state.right_name,
                left_raw=left_raw,
                right_raw=right_raw,
                component=component,
                precision=precision,
                keep_volatile=keep_volatile,
                key_prefix="pair",
            )

    with tab_batch:
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
            return

        names = sorted(variants.keys())
        if len(names) < 2:
            st.warning("Need at least two variants.")
            return

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
                key_prefix="batch",
            )


if __name__ == "__main__":
    main()
