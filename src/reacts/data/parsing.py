from __future__ import annotations

import ast
import json
import math
import re
from typing import Any

PATENT_ROUTE_SUFFIX = re.compile(r"^(?P<patent>.+)-(?P<route>\d{4,})$")
INTERMEDIATE_RE = re.compile(r"^M\d+$", re.IGNORECASE)


def first_scalar(value: Any) -> Any:
    """Collapse duplicate-column pandas Series to the first non-null scalar."""
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, list, tuple, dict)):
        try:
            values = value.tolist()
            if not isinstance(values, list):
                values = [values]
            for item in values:
                if item is not None and not (isinstance(item, float) and math.isnan(item)):
                    return item
            return None
        except Exception:
            pass
    return value


def parse_list(value: Any) -> list[str]:
    # Parquet list columns are commonly materialized as numpy arrays or Arrow
    # scalars. Handle those before first_scalar(), which is intentionally aimed
    # at duplicate-column pandas Series and would otherwise keep only the first
    # element of a multi-valued chemistry list.
    module_root = type(value).__module__.split(".", 1)[0]
    if module_root == "numpy" and hasattr(value, "tolist"):
        value = value.tolist()
    elif module_root == "pyarrow":
        if hasattr(value, "as_py"):
            value = value.as_py()
        elif hasattr(value, "to_pylist"):
            value = value.to_pylist()

    value = first_scalar(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if x is not None and str(x).strip()]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if x is not None and str(x).strip()]
            except Exception:
                pass
    return [text]


def patent_document_id(route_id: str) -> str:
    route_id = str(first_scalar(route_id) or "").strip()
    match = PATENT_ROUTE_SUFFIX.match(route_id)
    return match.group("patent") if match else route_id


def stable_group_split(group_id: str) -> str:
    import hashlib

    value = int(hashlib.sha1(group_id.encode("utf-8")).hexdigest()[:12], 16) % 10_000
    if value < 8_000:
        return "train"
    if value < 9_000:
        return "val"
    return "test"


def stable_hash(payload: str, length: int = 16) -> str:
    import hashlib

    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]
