import re
from dataclasses import dataclass
from typing import Dict, FrozenSet, List


def normalize_text_key(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    text = re.sub(r"[\"'`“”’]+", "", text)
    return text


def _tokenize_key(text: str) -> FrozenSet[str]:
    return frozenset(tok for tok in re.findall(r"[a-z0-9]+", text) if len(tok) >= 2)


def _token_overlap(a_tokens: FrozenSet[str], b_tokens: FrozenSet[str]) -> tuple[float, float]:
    if not a_tokens or not b_tokens:
        return 0.0, 0.0
    overlap = len(a_tokens & b_tokens)
    if overlap == 0:
        return 0.0, 0.0
    union = len(a_tokens | b_tokens)
    jaccard = overlap / max(1, union)
    coverage = overlap / max(1, min(len(a_tokens), len(b_tokens)))
    return jaccard, coverage


def _extract_numeric_markers(text: str) -> FrozenSet[str]:
    return frozenset(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text))


@dataclass(frozen=True)
class TextSignature:
    key: str
    tokens: FrozenSet[str]
    numbers: FrozenSet[str]


def _build_signature(key: str) -> TextSignature:
    return TextSignature(
        key=key,
        tokens=_tokenize_key(key),
        numbers=_extract_numeric_markers(key),
    )


def _is_near_duplicate(a: TextSignature, b: TextSignature, *, paragraph_mode: bool = False) -> bool:
    if not a.key or not b.key:
        return False
    if a.key == b.key:
        return True

    if a.numbers and b.numbers and a.numbers != b.numbers:
        return False

    shorter, longer = (a.key, b.key) if len(a.key) <= len(b.key) else (b.key, a.key)
    if len(shorter) >= 24 and shorter in longer:
        length_ratio = len(shorter) / max(1, len(longer))
        if length_ratio >= 0.82:
            return True

    jaccard, coverage = _token_overlap(a.tokens, b.tokens)

    if paragraph_mode:
        return jaccard >= 0.90 or coverage >= 0.97
    return jaccard >= 0.87 or coverage >= 0.95


def deduplicate_string_items(items: List[str], *, keep_quotes_verbatim: bool) -> List[str]:
    seen: Dict[str, str] = {}
    order: List[str] = []

    for raw in items:
        if not isinstance(raw, str):
            continue
        text = raw.strip()
        if not text:
            continue
        key = normalize_text_key(text)
        if not key:
            continue
        if key not in seen:
            seen[key] = text
            order.append(key)

    unique_keys: List[str] = []
    signatures = {key: _build_signature(key) for key in order}
    for key in order:
        keep = True
        for kept in unique_keys:
            if _is_near_duplicate(signatures[key], signatures[kept]):
                keep = False
                break
        if keep:
            unique_keys.append(key)

    values = [seen[k] for k in unique_keys]

    if keep_quotes_verbatim:
        return values

    pruned: List[str] = []
    for i, x in enumerate(values):
        xl = x.lower()
        if len(xl) < 40:
            longer_exists = any(
                i != j
                and len(values[j]) >= len(x) + 20
                and xl in values[j].lower()
                for j in range(len(values))
            )
            if longer_exists:
                continue
        pruned.append(x)
    return pruned


def dedupe_unbounded_summary_text(text: str) -> str:
    raw = str(text or "").strip()
    if not raw:
        return ""

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    if len(paragraphs) <= 1:
        return raw

    seen: Dict[str, str] = {}
    order: List[str] = []
    for p in paragraphs:
        key = normalize_text_key(p)
        if key and key not in seen:
            seen[key] = p
            order.append(key)

    unique: List[str] = []
    signatures = {key: _build_signature(key) for key in order}
    for key in order:
        keep = True
        for prev in unique:
            if _is_near_duplicate(signatures[key], signatures[prev], paragraph_mode=True):
                keep = False
                break
        if keep:
            unique.append(key)

    cleaned = [seen[k] for k in unique]
    return "\n\n".join(cleaned)
