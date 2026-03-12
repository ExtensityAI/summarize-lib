import re
from typing import List


def is_substantive_chunk(text: str, min_alnum_chars: int = 40) -> bool:
    raw = str(text or "").strip()
    if not raw:
        return False
    cleaned = re.sub(r"\[\[DOCUMENT::[^\]]+\]\]:\s*<<<", " ", raw)
    cleaned = cleaned.replace(">>>", " ")
    cleaned = re.sub(r"\[\[PURPOSE\]\].*", " ", cleaned, flags=re.DOTALL)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    alnum_count = sum(1 for ch in cleaned if ch.isalnum())
    return alnum_count >= min_alnum_chars


def split_structural_segments(text: str) -> List[str]:
    raw = text.strip()
    if not raw:
        return []

    heading_or_slide = re.compile(r"^(#{1,6}\s+.+|\s*[-=]{3,}\s*$|\s*[A-Z][A-Z\s0-9,:;\-]{8,}$)")
    lines = raw.splitlines()

    segments: List[str] = []
    current: List[str] = []

    def flush_current() -> None:
        if current:
            chunk = "\n".join(current).strip()
            if chunk:
                segments.append(chunk)
            current.clear()

    for line in lines:
        stripped = line.strip()
        if heading_or_slide.match(stripped):
            flush_current()
            current.append(line)
            continue
        if stripped == "":
            if current and current[-1] != "":
                current.append("")
            continue
        current.append(line)

    flush_current()

    if len(segments) <= 1:
        by_paragraph = [s.strip() for s in re.split(r"\n\s*\n", raw) if s.strip()]
        return by_paragraph if by_paragraph else [raw]

    return segments
