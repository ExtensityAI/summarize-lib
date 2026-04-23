import json
import re
from typing import Any, Callable


# Invisible / typographic characters that are safe to strip before summarization.
# These carry no semantic meaning in prose and tend to leak through to downstream
# LLM outputs (e.g. soft hyphens inside German compound words appearing in TOC
# titles even when the extracted text never had them elsewhere). Stripping at
# the reader-content boundary removes an entire class of downstream artifacts.
#
# Kept:
#   - U+00A0 NON-BREAKING SPACE (has typographic meaning: "keep these words together")
#   - Curly quotes, em/en dashes, ellipses (intentional typography)
_INVISIBLE_CHARS_PATTERN = re.compile(
    "["
    "\u00AD"  # SOFT HYPHEN
    "\u200B"  # ZERO WIDTH SPACE
    "\u200C"  # ZERO WIDTH NON-JOINER
    "\u200D"  # ZERO WIDTH JOINER
    "\u2060"  # WORD JOINER
    "\uFEFF"  # ZERO WIDTH NO-BREAK SPACE / BOM
    "\u180E"  # MONGOLIAN VOWEL SEPARATOR
    "]"
)


def _strip_invisible_chars(text: str) -> str:
    """Remove invisible / typographic artifacts that have no semantic meaning.

    Used before passing reader content to summarization. Prevents soft hyphens
    and zero-width characters from flowing into asset summaries (and, from
    there, into prompts and LLM outputs where they cause visible corruption
    such as "Pass\u00adivit\u00e4t" appearing in chapter titles).
    """
    if not text:
        return text
    return _INVISIBLE_CHARS_PATTERN.sub("", text)


def normalize_reader_content(content: Any, safe_jsonable: Callable[[Any], Any]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return _strip_invisible_chars(content)
    if isinstance(content, (bytes, bytearray)):
        return _strip_invisible_chars(content.decode("utf-8", errors="ignore"))

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
        return _strip_invisible_chars(json.dumps(safe_jsonable(content), ensure_ascii=False))

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
