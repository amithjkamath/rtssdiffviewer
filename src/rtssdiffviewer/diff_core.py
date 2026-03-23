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
