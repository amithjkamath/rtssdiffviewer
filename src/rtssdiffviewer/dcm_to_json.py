"""DICOM RTSS to JSON conversion helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pydicom
from pydicom.dataelem import DataElement
from pydicom.dataset import Dataset
from pydicom.uid import UID


def _tag_key(elem: DataElement) -> str:
    tag_str = f"({elem.tag.group:04X},{elem.tag.element:04X})"
    keyword = elem.keyword or ""
    return f"{tag_str} {keyword}".strip()


def _bytes_summary(data: bytes) -> dict[str, Any]:
    return {
        "_type": "bytes",
        "length": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, bytes):
        return _bytes_summary(value)
    return str(value)


def _convert_value(elem: DataElement) -> Any:
    vr = elem.VR
    if vr == "SQ":
        return [_dataset_to_dict(item) for item in (elem.value or [])]

    if vr in ("OB", "OW", "OD", "OF", "OL", "OV", "UN"):
        raw = elem.value
        if isinstance(raw, bytes):
            return _bytes_summary(raw)
        return {"_type": "bytes", "length": 0, "sha256": ""}

    if vr == "DS":
        raw = elem.value
        if raw is None:
            return None
        if hasattr(raw, "__iter__") and not isinstance(raw, str):
            try:
                return [float(v) for v in raw]
            except (TypeError, ValueError):
                return [str(v) for v in raw]
        try:
            return float(raw)
        except (TypeError, ValueError):
            return str(raw)

    if vr == "IS":
        raw = elem.value
        if raw is None:
            return None
        if hasattr(raw, "__iter__") and not isinstance(raw, str):
            try:
                return [int(v) for v in raw]
            except (TypeError, ValueError):
                return [str(v) for v in raw]
        try:
            return int(raw)
        except (TypeError, ValueError):
            return str(raw)

    raw = elem.value
    if hasattr(raw, "__iter__") and not isinstance(raw, (str, bytes, UID)):
        items = list(raw)
        if len(items) == 1:
            return _scalar(items[0])
        return [_scalar(v) for v in items]

    return _scalar(raw)


def _dataset_to_dict(ds: Dataset) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for elem in ds:
        key = _tag_key(elem)
        if elem.keyword == "ContourData" and elem.value:
            flat = [float(v) for v in elem.value]
            out[key] = [flat[i : i + 3] for i in range(0, len(flat), 3)]
        else:
            out[key] = _convert_value(elem)
    return out


def dcm_to_json(dcm_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    """Convert DICOM file to a deterministic JSON dict and optionally write it."""
    dcm_path = Path(dcm_path)
    ds = pydicom.dcmread(str(dcm_path), force=True)

    result: dict[str, Any] = {}
    if hasattr(ds, "file_meta") and ds.file_meta:
        result["_file_meta"] = _dataset_to_dict(ds.file_meta)
    result.update(_dataset_to_dict(ds))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False, sort_keys=False)

    return result
