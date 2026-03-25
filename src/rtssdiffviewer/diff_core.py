"""Core RTSS JSON normalization and diff helpers."""

from __future__ import annotations

import difflib
import json
from typing import Any

DEFAULT_VOLATILE_TAG_PREFIXES = {
    "(0008,0012)",
    "(0008,0013)",
    "(0008,0018)",
    "(0020,000D)",
    "(0020,000E)",
    "(0020,0052)",
    "(3006,0008)",
    "(3006,0009)",
}

COMPONENT_KEYS = {
    "references": ["(3006,0010) ReferencedFrameOfReferenceSequence"],
    "structures": [
        "(3006,0020) StructureSetROISequence",
        "(3006,0080) RTROIObservationsSequence",
    ],
    "contours": ["(3006,0039) ROIContourSequence"],
}


def is_volatile_tag(key: str, ignore_prefixes: set[str]) -> bool:
    if not key.startswith("("):
        return False
    tag_prefix = key.split(" ", 1)[0]
    return tag_prefix in ignore_prefixes


def normalize_value(value: Any, precision: int, ignore_prefixes: set[str]) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key in sorted(value.keys()):
            if is_volatile_tag(key, ignore_prefixes):
                continue
            out[key] = normalize_value(value[key], precision, ignore_prefixes)
        return out

    if isinstance(value, list):
        return [normalize_value(v, precision, ignore_prefixes) for v in value]

    if isinstance(value, float):
        return round(value, precision)

    return value


def select_component(data: dict[str, Any], component: str) -> dict[str, Any]:
    if component == "all":
        return data

    if component == "metadata":
        excluded: set[str] = set()
        for keys in COMPONENT_KEYS.values():
            excluded.update(keys)
        return {k: v for k, v in data.items() if k not in excluded}

    keys = COMPONENT_KEYS[component]
    return {k: data[k] for k in keys if k in data}


def pretty_json_text(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True)


def unified_diff_text(left_text: str, right_text: str, left_name: str, right_name: str) -> str:
    left_lines = [line + "\n" for line in left_text.splitlines()]
    right_lines = [line + "\n" for line in right_text.splitlines()]
    diff = difflib.unified_diff(
        left_lines,
        right_lines,
        fromfile=left_name,
        tofile=right_name,
        lineterm="",
    )
    return "\n".join(diff)


def _extract_xyz_points(value: Any) -> list[tuple[float, float, float]]:
    """Recursively extract (x, y, z) coordinate tuples from nested structures."""
    points: list[tuple[float, float, float]] = []

    if isinstance(value, list):
        if len(value) == 3 and all(isinstance(v, (int, float)) for v in value):
            points.append((float(value[0]), float(value[1]), float(value[2])))
        else:
            for item in value:
                points.extend(_extract_xyz_points(item))

    return points


def extract_contours_by_slice(
    contour_data: dict[str, Any], precision: int
) -> dict[float, dict[str, list[tuple[float, float, float]]]]:
    """
    Extract contours grouped by slice (z-coordinate, rounded to precision).
    
    Returns a dict mapping z-coordinate -> {roi_name -> list of (x, y, z) points}
    """
    slices: dict[float, dict[str, list[tuple[float, float, float]]]] = {}

    roi_contour_seq = contour_data.get("(3006,0039) ROIContourSequence", [])
    if not isinstance(roi_contour_seq, list):
        return slices

    for roi_idx, roi_item in enumerate(roi_contour_seq):
        if not isinstance(roi_item, dict):
            continue

        # Try to get ROI identifier - use ReferencedROINumber if available, otherwise use index
        roi_number = roi_item.get("(3006,0084) ReferencedROINumber")
        if roi_number is None:
            roi_number = roi_idx
        roi_name = f"ROI {roi_number}"

        # Extract contour sequences
        contour_seq = roi_item.get("(3006,0040) ContourSequence", [])
        if not isinstance(contour_seq, list):
            continue

        for contour_item in contour_seq:
            if not isinstance(contour_item, dict):
                continue

            points = _extract_xyz_points(contour_item.get("(3006,0050) ContourData", []))
            if not points:
                continue

            # Group by z-coordinate (rounded to precision)
            for x, y, z in points:
                z_rounded = round(z, precision)
                if z_rounded not in slices:
                    slices[z_rounded] = {}
                if roi_name not in slices[z_rounded]:
                    slices[z_rounded][roi_name] = []
                slices[z_rounded][roi_name].append((x, y, z))

    return slices


def _format_points_for_display(points: list[tuple[float, float, float]], precision: int = 4) -> str:
    """Format a list of points as a readable string."""
    if not points:
        return "  (no points)"
    lines = []
    for i, (x, y, z) in enumerate(sorted(points), 1):
        lines.append(f"Point {i}: x={x:.{precision}f}, y={y:.{precision}f}, z={z:.{precision}f}")
    return "\n".join(lines)


def get_contour_slices_structured(
    left_data: dict[str, Any],
    right_data: dict[str, Any],
    precision: int = 4,
) -> tuple[dict[float, dict[str, list[tuple[float, float, float]]]], dict[float, dict[str, list[tuple[float, float, float]]]]]:
    """
    Extract and return structured slice data for both left and right files.
    
    Returns: (left_slices, right_slices) where each is a dict mapping
    z-coordinate -> {roi_name -> list of (x, y, z) points}
    """
    left_slices = extract_contours_by_slice(left_data, precision)
    right_slices = extract_contours_by_slice(right_data, precision)
    return left_slices, right_slices


def contour_diff_text(
    left_data: dict[str, Any],
    right_data: dict[str, Any],
    left_name: str,
    right_name: str,
    precision: int = 4,
) -> str:
    """
    Generate an intelligent diff for contour data grouped by slice plane.
    
    Compares contours by slice (z-coordinate) and shows:
    - Slices only in left file
    - Slices only in right file  
    - Slices in both with point differences
    """
    left_slices = extract_contours_by_slice(left_data, precision)
    right_slices = extract_contours_by_slice(right_data, precision)

    if not left_slices and not right_slices:
        return "No contour data found in either file."

    lines: list[str] = []
    lines.append(f"=== Contour Diff by Slice Plane ===")
    lines.append(f"Left:  {left_name}")
    lines.append(f"Right: {right_name}")
    lines.append("")

    all_z_coords = sorted(set(left_slices.keys()) | set(right_slices.keys()))

    for z in all_z_coords:
        lines.append(f"--- Slice z={z:.{precision}f} ---")

        left_rois = left_slices.get(z, {})
        right_rois = right_slices.get(z, {})

        if z not in left_slices:
            lines.append(f"Only in {right_name}:")
            for roi_name, points in right_rois.items():
                lines.append(f"  ROI: {roi_name} ({len(points)} points)")
                lines.append(_format_points_for_display(points, precision))
            lines.append("")
            continue

        if z not in right_slices:
            lines.append(f"Only in {left_name}:")
            for roi_name, points in left_rois.items():
                lines.append(f"  ROI: {roi_name} ({len(points)} points)")
                lines.append(_format_points_for_display(points, precision))
            lines.append("")
            continue

        # Both have this slice - compare ROIs
        all_roi_names = sorted(set(left_rois.keys()) | set(right_rois.keys()))
        has_differences = False

        for roi_name in all_roi_names:
            left_points = sorted(left_rois.get(roi_name, []))
            right_points = sorted(right_rois.get(roi_name, []))

            if left_points == right_points:
                continue

            has_differences = True
            lines.append(f"  ROI: {roi_name}")

            if roi_name not in left_rois:
                lines.append(f"    Only in {right_name}: {len(right_points)} points")
                lines.append(_format_points_for_display(right_points, precision))
            elif roi_name not in right_rois:
                lines.append(f"    Only in {left_name}: {len(left_points)} points")
                lines.append(_format_points_for_display(left_points, precision))
            else:
                lines.append(f"    {left_name}: {len(left_points)} points")
                lines.append(_format_points_for_display(left_points, precision))
                lines.append(f"    {right_name}: {len(right_points)} points")
                lines.append(_format_points_for_display(right_points, precision))

        if has_differences:
            lines.append("")
        else:
            lines.append(f"  All ROIs identical on this slice")
            lines.append("")

    return "\n".join(lines)
