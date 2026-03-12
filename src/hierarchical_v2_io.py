import json
from typing import Any, Callable


def normalize_reader_content(content: Any, safe_jsonable: Callable[[Any], Any]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (bytes, bytearray)):
        return content.decode("utf-8", errors="ignore")

    value = getattr(content, "value", None)
    if value is not None and value is not content:
        return normalize_reader_content(value, safe_jsonable)

    if isinstance(content, (list, tuple)):
        parts = [normalize_reader_content(v, safe_jsonable).strip() for v in content]
        parts = [p for p in parts if p]
        return "\n\n".join(parts)

    if isinstance(content, dict):
        for key in ("text", "content", "markdown", "body"):
            if key in content:
                return normalize_reader_content(content[key], safe_jsonable)
        return json.dumps(safe_jsonable(content), ensure_ascii=False)

    try:
        iterator = iter(content)
    except Exception:
        iterator = None

    if iterator is not None:
        try:
            parts = [normalize_reader_content(v, safe_jsonable).strip() for v in iterator]
            parts = [p for p in parts if p]
            if parts:
                return "\n\n".join(parts)
        except Exception:
            pass

    return str(content)
