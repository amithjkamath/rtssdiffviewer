from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import (
    _format_slice_rois_text,
    _greedy_point_matches,
    _extract_ordered_contours_by_slice,
    _extract_volume_bounds_from_rtss,
    _merge_bounds,
    _nearest_slice_value,
    _safe_key_fragment,
    _slice_match_metrics,
    _step_slice_value,
    extract_contour_points,
    should_use_unified_only,
)


def test_extract_contour_points_collects_all_triplets() -> None:
    sample = {
        "(3006,0039) ROIContourSequence": [
            {
                "(3006,0040) ContourSequence": [
                    {
                        "(3006,0050) ContourData": [
                            [1.0, 2.0, 3.0],
                            [4, 5, 6],
                        ]
                    },
                    {
                        "(3006,0050) ContourData": [
                            [7.5, 8.5, 9.5],
                        ]
                    },
                ]
            }
        ]
    }

    points = extract_contour_points(sample)

    assert points == [
        (1.0, 2.0, 3.0),
        (4.0, 5.0, 6.0),
        (7.5, 8.5, 9.5),
    ]


def test_extract_contour_points_ignores_non_triplet_data() -> None:
    sample = {
        "(3006,0050) ContourData": [
            [1.0, 2.0],
            [3.0, 4.0, 5.0, 6.0],
            "not-a-point",
            {"unexpected": "shape"},
        ]
    }

    points = extract_contour_points(sample)

    assert points == []


def test_should_use_unified_only_for_batch_text_mode() -> None:
    use_unified_only, reason = should_use_unified_only(
        "left",
        "right",
        allow_rich_view=False,
        max_rich_chars=400_000,
        max_rich_lines=5_000,
    )

    assert use_unified_only is True
    assert "unified diff text only" in reason


def test_should_use_unified_only_for_large_input() -> None:
    left = "a" * 250_000
    right = "b" * 250_000

    use_unified_only, reason = should_use_unified_only(
        left,
        right,
        allow_rich_view=True,
        max_rich_chars=400_000,
        max_rich_lines=5_000,
    )

    assert use_unified_only is True
    assert "Large comparison detected" in reason


def test_should_use_rich_view_for_small_input() -> None:
    use_unified_only, reason = should_use_unified_only(
        "small-left\n",
        "small-right\n",
        allow_rich_view=True,
        max_rich_chars=400_000,
        max_rich_lines=5_000,
    )

    assert use_unified_only is False
    assert reason == ""


def test_extract_volume_bounds_from_rtss_metadata() -> None:
    sample = {
        "(3006,0039) ROIContourSequence": [
            {
                "(3006,004A) SourcePixelPlanesCharacteristicsSequence": [
                    {
                        "(0020,0032) ImagePositionPatient": [10.0, 20.0, 30.0],
                        "(0020,0037) ImageOrientationPatient": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                        "(0028,0030) PixelSpacing": [2.0, 1.0],
                        "(0028,0010) Rows": 2,
                        "(0028,0011) Columns": 3,
                        "(0028,0008) NumberOfFrames": 4,
                        "(0018,0088) SpacingBetweenSlices": 2.0,
                    }
                ]
            }
        ]
    }

    bounds, msg = _extract_volume_bounds_from_rtss(sample)

    assert bounds is not None
    assert bounds["x_min"] == 10.0
    assert bounds["x_max"] == 12.0
    assert bounds["y_min"] == 20.0
    assert bounds["y_max"] == 22.0
    assert bounds["z_min"] == 30.0
    assert bounds["z_max"] == 36.0
    assert "derived" in msg


def test_extract_volume_bounds_from_rtss_missing_metadata() -> None:
    sample = {
        "(3006,0039) ROIContourSequence": [
            {"(3006,0040) ContourSequence": [{"(3006,0050) ContourData": [[1.0, 2.0, 3.0]]}]}
        ]
    }

    bounds, msg = _extract_volume_bounds_from_rtss(sample)

    assert bounds is None
    assert "incomplete" in msg


def test_merge_bounds() -> None:
    a = {"x_min": 0.0, "x_max": 1.0, "y_min": 2.0, "y_max": 3.0, "z_min": 4.0, "z_max": 5.0}
    b = {"x_min": -1.0, "x_max": 2.0, "y_min": 1.5, "y_max": 3.5, "z_min": 3.0, "z_max": 6.0}

    merged = _merge_bounds(a, b)

    assert merged == {
        "x_min": -1.0,
        "x_max": 2.0,
        "y_min": 1.5,
        "y_max": 3.5,
        "z_min": 3.0,
        "z_max": 6.0,
    }


def test_greedy_point_matches_unique_pairs() -> None:
    left = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    right = [(0.2, 0.0, 0.0), (9.8, 0.0, 0.0)]

    matches = _greedy_point_matches(left, right)

    assert len(matches) == 2
    assert {m[0] for m in matches} == {0, 1}
    assert {m[1] for m in matches} == {0, 1}


def test_slice_match_metrics_dice_and_mismatch() -> None:
    left_rois = {
        "ROI 1": [
            (0.0, 0.0, 1.0),
            (10.0, 0.0, 1.0),
        ]
    }
    right_rois = {
        "ROI 1": [
            (0.2, 0.0, 1.0),
            (11.5, 0.0, 1.0),
        ]
    }

    metrics = _slice_match_metrics(left_rois, right_rois, tolerance_mm=1.0)

    assert metrics["left_count"] == 2
    assert metrics["right_count"] == 2
    assert metrics["mismatch_count"] == 2
    assert metrics["count_delta"] == 0
    assert metrics["dice"] == 0.5
    assert metrics["identical_slice"] is False


def test_slice_match_metrics_identical_slice() -> None:
    rois = {
        "ROI 1": [
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 3.0),
        ]
    }

    metrics = _slice_match_metrics(rois, rois, tolerance_mm=0.1)

    assert metrics["identical_slice"] is True
    assert metrics["mismatch_count"] == 0
    assert metrics["dice"] == 1.0


def test_safe_key_fragment_replaces_non_alnum() -> None:
    assert _safe_key_fragment("z=12.5/ROI 1") == "z_12_5_ROI_1"


def test_nearest_slice_value_selects_closest() -> None:
    slices = [1.0, 2.5, 5.0]

    assert _nearest_slice_value(slices, 2.7) == 2.5
    assert _nearest_slice_value(slices, None) == 1.0


def test_nearest_slice_value_returns_none_for_empty() -> None:
    assert _nearest_slice_value([], 10.0) is None


def test_extract_ordered_contours_by_slice_preserves_point_order() -> None:
    sample = {
        "(3006,0039) ROIContourSequence": [
            {
                "(3006,0084) ReferencedROINumber": 7,
                "(3006,0040) ContourSequence": [
                    {
                        "(3006,0048) ContourNumber": 3,
                        "(3006,0050) ContourData": [
                            [10.0, 5.0, 1.0001],
                            [11.0, 6.0, 1.0001],
                            [12.0, 7.0, 1.0001],
                        ],
                    }
                ],
            }
        ]
    }

    slices = _extract_ordered_contours_by_slice(sample, precision=3)

    assert list(slices.keys()) == [1.0]
    assert len(slices[1.0]) == 1
    contour = slices[1.0][0]
    assert contour["contour_label"] == "ROI 7 | Contour 3"
    assert contour["points"] == [
        (10.0, 5.0, 1.0001),
        (11.0, 6.0, 1.0001),
        (12.0, 7.0, 1.0001),
    ]


def test_step_slice_value_bounds_and_steps() -> None:
    slices = [1.0, 2.0, 3.0]

    assert _step_slice_value(slices, 2.0, -1) == 1.0
    assert _step_slice_value(slices, 2.0, 1) == 3.0
    assert _step_slice_value(slices, 1.0, -1) == 1.0
    assert _step_slice_value(slices, 3.0, 1) == 3.0


def test_step_slice_value_handles_missing_current() -> None:
    slices = [1.0, 2.0, 3.0]
    assert _step_slice_value(slices, 99.0, 1) == 2.0


def test_format_slice_rois_text_includes_index_and_order() -> None:
    rois = {
        "ROI 2": [(2.0, 2.0, 5.0)],
        "ROI 1": [(1.0, 1.0, 5.0), (3.0, 3.0, 5.0)],
    }

    text = _format_slice_rois_text(rois, precision=2)

    assert "ROI 1: 2 points" in text
    assert "001: (1.00, 1.00, 5.00)" in text
    assert "002: (3.00, 3.00, 5.00)" in text
    assert "ROI 2: 1 points" in text


def test_format_slice_rois_text_empty() -> None:
    assert _format_slice_rois_text({}, precision=4) == "(no contours on this slice)"
