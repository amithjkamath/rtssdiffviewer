from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import (
    _extract_volume_bounds_from_rtss,
    _merge_bounds,
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
