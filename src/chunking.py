"""Reusable hybrid (structural + semantic) text chunker.

Extracted from ``HierarchicalSummaryV2`` so the same chunking strategy can be
reused outside the summarization pipeline (e.g. RAG indexing). The algorithm is
self-contained here; everything provider/host-specific is dependency-injected:

  * ``embed_batch``      – embed a list of strings -> list of vectors (or None on
                           failure). Defaults to ``Symbol(texts).embed()`` (uses
                           the configured symai embedding engine, e.g. OpenAI
                           ``text-embedding-3-small``).
  * ``count_tokens``     – ``(text, chunk_size) -> int`` token estimate used for
                           boundary decisions. Defaults to a tokenizer/heuristic.
  * ``recursive_split``  – ``(text, chunk_size) -> list[str]`` fallback splitter
                           for a single oversized segment. Defaults to a chonkie
                           ``RecursiveChunker``.
  * ``prepare_segments`` – sanitize + split segments to fit the embedding input
                           limit before embedding. Default provided.

``HierarchicalSummaryV2`` injects its own callables so its behaviour is
unchanged; standalone callers (storyone RAG) rely on the defaults.
"""

import math
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .hierarchical_v2_chunking import split_structural_segments

# soft_q, soft_fill_ratio, hard_q, hard_fill_ratio
DEFAULT_BOUNDARY_CONFIG: Tuple[float, float, float, float] = (0.35, 0.42, 0.18, 0.30)
_FALLBACK_CHARS_PER_TOKEN = 4


class HybridChunker:
    def __init__(
        self,
        *,
        max_chunk_size: int = 2000,
        min_chunk_size: int = 400,
        semantic: bool = True,
        boundary_config: Tuple[float, float, float, float] = DEFAULT_BOUNDARY_CONFIG,
        embed_batch: Optional[Callable[[List[str]], Optional[List[List[float]]]]] = None,
        count_tokens: Optional[Callable[[str, int], int]] = None,
        recursive_split: Optional[Callable[[str, int], List[str]]] = None,
        prepare_segments: Optional[Callable[[List[str]], List[str]]] = None,
        embed_item_token_limit: int = 1800,
        tokenizer_name: str = "gpt2",
    ) -> None:
        self.max_chunk_size = max(1, int(max_chunk_size))
        self.min_chunk_size = max(1, int(min_chunk_size))
        self.semantic = bool(semantic)
        self.boundary_config = boundary_config
        self.embed_item_token_limit = max(128, int(embed_item_token_limit))
        self.tokenizer_name = tokenizer_name
        self._embed_batch = embed_batch
        self._count_tokens = count_tokens
        self._recursive_split = recursive_split
        self._prepare_segments = prepare_segments
        self._chonkie = None  # lazy default recursive splitter
        self._tokenizer = None  # lazy default tokenizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def chunk(self, text: str, *, chunk_size: Optional[int] = None) -> List[str]:
        """Split ``text`` into chunks no larger than ``chunk_size`` tokens.

        Structural seams first; then semantic packing (if enabled and embeddings
        succeed), else structural packing; recursive split as a last resort.
        """
        text = str(text or "")
        if not text.strip():
            return []
        size = max(1, int(chunk_size or self.max_chunk_size))

        segments = split_structural_segments(text)
        if len(segments) > 1:
            if self.semantic:
                chunks = self.pack_semantic(segments, size)
                if chunks:
                    return chunks
            chunks = self.pack_structural(segments, size)
            if chunks:
                return chunks

        return [c for c in self._recursive(text, size) if c.strip()] or [text]

    # ------------------------------------------------------------------
    # Packing (faithful port of HierarchicalSummaryV2._pack_segments_*)
    # ------------------------------------------------------------------
    def pack_semantic(
        self, segments: List[str], chunk_size: int, *, stats: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        prepared_segments = self.prepare(segments, stats=stats)
        vectors = self._embed(prepared_segments)
        if not vectors or len(vectors) != len(prepared_segments):
            return []

        sims: List[float] = []
        for i in range(len(prepared_segments) - 1):
            sims.append(self._cosine_similarity(vectors[i], vectors[i + 1]))

        soft_q, soft_fill_ratio, hard_q, hard_fill_ratio = self.boundary_config
        hard_threshold = 0.0
        soft_threshold = 0.0
        if sims:
            sorted_sims = sorted(sims)
            soft_threshold = self._quantile(sorted_sims, soft_q)
            hard_threshold = self._quantile(sorted_sims, hard_q)
        if stats is not None:
            stats["semantic_boundary_threshold_soft"] = soft_threshold
            stats["semantic_boundary_threshold_hard"] = hard_threshold

        chunks: List[str] = []
        current_parts: List[str] = []
        current_tokens = 0

        for i, segment in enumerate(prepared_segments):
            segment_tokens = self._tokens(segment, chunk_size)

            if segment_tokens >= chunk_size:
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_tokens = 0
                chunks.extend(s for s in self._recursive(segment, chunk_size) if s.strip())
                continue

            semantic_boundary_soft = False
            semantic_boundary_hard = False
            if i > 0 and i - 1 < len(sims):
                sim = sims[i - 1]
                semantic_boundary_soft = sim <= soft_threshold
                semantic_boundary_hard = sim <= hard_threshold

            should_flush = False
            if current_parts and current_tokens + segment_tokens > chunk_size:
                should_flush = True
            elif current_parts and semantic_boundary_hard and current_tokens >= int(chunk_size * hard_fill_ratio):
                should_flush = True
            elif current_parts and semantic_boundary_soft and current_tokens >= int(chunk_size * soft_fill_ratio):
                should_flush = True

            if should_flush:
                chunks.append("\n\n".join(current_parts))
                current_parts = [segment]
                current_tokens = segment_tokens
            else:
                current_parts.append(segment)
                current_tokens += segment_tokens

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return [c for c in chunks if c.strip()]

    def pack_structural(self, segments: List[str], chunk_size: int) -> List[str]:
        if not segments:
            return []

        chunks: List[str] = []
        current_parts: List[str] = []
        current_tokens = 0

        for segment in segments:
            segment_tokens = self._tokens(segment, chunk_size)
            if segment_tokens >= chunk_size:
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_tokens = 0
                chunks.extend(s for s in self._recursive(segment, chunk_size) if s.strip())
                continue

            if current_tokens + segment_tokens > chunk_size and current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = [segment]
                current_tokens = segment_tokens
            else:
                current_parts.append(segment)
                current_tokens += segment_tokens

        if current_parts:
            chunks.append("\n\n".join(current_parts))

        return [c for c in chunks if c.strip()]

    def prepare(self, segments: List[str], *, stats: Optional[Dict[str, Any]] = None) -> List[str]:
        if self._prepare_segments is not None:
            return self._prepare_segments(segments)
        return self._default_prepare(segments, stats=stats)

    # ------------------------------------------------------------------
    # Injected-or-default dependencies
    # ------------------------------------------------------------------
    def _tokens(self, text: str, chunk_size: int) -> int:
        if self._count_tokens is not None:
            return int(self._count_tokens(text, chunk_size))
        return self._default_count_tokens(text)

    def _recursive(self, text: str, chunk_size: int) -> List[str]:
        if self._recursive_split is not None:
            try:
                return [str(s) for s in self._recursive_split(text, chunk_size)]
            except Exception:
                return [text]
        return self._default_recursive_split(text, chunk_size)

    def _embed(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not texts:
            return []
        fn = self._embed_batch if self._embed_batch is not None else self._default_embed_batch
        try:
            return fn(texts)
        except Exception:
            return None

    def _default_prepare(self, segments: List[str], *, stats: Optional[Dict[str, Any]] = None) -> List[str]:
        if not segments:
            return []
        token_limit = self.embed_item_token_limit
        expanded: List[str] = []
        expanded_count = 0
        for segment in segments:
            segment = self._sanitize_embedding_text(segment)
            if not segment:
                continue
            if self._default_count_tokens(segment) <= token_limit:
                expanded.append(segment)
                continue
            parts = [self._sanitize_embedding_text(p) for p in self._recursive(segment, token_limit)]
            parts = [p for p in parts if p]
            if not parts:
                expanded.append(segment)
                continue
            expanded.extend(parts)
            expanded_count += max(0, len(parts) - 1)
        if stats is not None:
            stats["semantic_segments_expanded"] = expanded_count
        return [s for s in expanded if s]

    def _default_count_tokens(self, text: str) -> int:
        tok = self._get_tokenizer()
        if tok is not None:
            try:
                return len(tok.encode(str(text)))
            except Exception:
                pass
        return max(1, len(str(text)) // _FALLBACK_CHARS_PER_TOKEN)

    def _get_tokenizer(self):
        if self._tokenizer is None:
            try:
                from .hierarchical_v2 import get_current_tokenizer

                self._tokenizer = get_current_tokenizer(self.tokenizer_name) or False
            except Exception:
                self._tokenizer = False
        return self._tokenizer or None

    def _get_chonkie(self):
        if self._chonkie is None:
            from symai.components import ChonkieChunker

            self._chonkie = ChonkieChunker(tokenizer_name=self.tokenizer_name)
        return self._chonkie

    def _default_recursive_split(self, text: str, chunk_size: int) -> List[str]:
        try:
            from symai import Symbol

            split = self._get_chonkie()(
                data=Symbol(text), chunker_name="RecursiveChunker", chunk_size=chunk_size
            )
            parts = [str(s).strip() for s in split if str(s).strip()]
            if parts:
                return parts
        except Exception:
            pass
        # Last-resort character windows.
        window = max(1, chunk_size) * _FALLBACK_CHARS_PER_TOKEN
        return [text[i : i + window] for i in range(0, len(text), window)] or [text]

    def _default_embed_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        from symai import Symbol

        def embed_one_batch(batch: List[str]) -> List[List[float]]:
            res = Symbol(batch).embed()
            values = res.value if hasattr(res, "value") else res
            return [self._normalize_vector(v) for v in values]

        def embed_safe(batch: List[str]) -> List[List[float]]:
            try:
                return embed_one_batch(batch)
            except Exception:
                if len(batch) <= 1:
                    raise
                mid = len(batch) // 2
                return embed_safe(batch[:mid]) + embed_safe(batch[mid:])

        vectors: List[List[float]] = []
        for batch in self._iter_embedding_batches(texts):
            vectors.extend(embed_safe(batch))
        return vectors if len(vectors) == len(texts) else None

    def _iter_embedding_batches(
        self, texts: List[str], *, max_items: int = 16, max_tokens: int = 12000
    ) -> List[List[str]]:
        batches: List[List[str]] = []
        batch: List[str] = []
        batch_tokens = 0
        for text in texts:
            t = self._default_count_tokens(text)
            if batch and (len(batch) >= max_items or batch_tokens + t > max_tokens):
                batches.append(batch)
                batch = []
                batch_tokens = 0
            batch.append(text)
            batch_tokens += t
        if batch:
            batches.append(batch)
        return batches

    # ------------------------------------------------------------------
    # Pure helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_vector(emb: Any) -> List[float]:
        if isinstance(emb, list) and emb and isinstance(emb[0], list):
            emb = emb[0]
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        if not isinstance(emb, list):
            emb = list(emb)
        return [float(x) for x in emb]

    @staticmethod
    def _sanitize_embedding_text(text: Any) -> str:
        raw = str(text or "")
        if not raw:
            return ""
        cleaned = raw.replace("\x00", " ")
        cleaned = "".join(ch if (ch >= " " or ch in "\n\r\t") else " " for ch in cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _quantile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        qq = min(1.0, max(0.0, q))
        idx = int(round(qq * (len(values) - 1)))
        return values[idx]

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        if n == 0:
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for i in range(n):
            av = float(a[i])
            bv = float(b[i])
            dot += av * bv
            na += av * av
            nb += bv * bv
        denom = math.sqrt(max(na, 1e-12)) * math.sqrt(max(nb, 1e-12))
        if denom <= 0:
            return 0.0
        return dot / denom


def chunk_text(
    text: str,
    *,
    max_chunk_size: int = 2000,
    min_chunk_size: int = 400,
    semantic: bool = True,
    embed_batch: Optional[Callable[[List[str]], Optional[List[List[float]]]]] = None,
    tokenizer_name: str = "gpt2",
) -> List[str]:
    """Convenience: hybrid-chunk ``text`` with sensible defaults.

    With ``semantic=True`` and no ``embed_batch``, uses the configured symai
    embedding engine for boundary detection. Returns chunks no larger than
    ``max_chunk_size`` tokens.
    """
    return HybridChunker(
        max_chunk_size=max_chunk_size,
        min_chunk_size=min_chunk_size,
        semantic=semantic,
        embed_batch=embed_batch,
        tokenizer_name=tokenizer_name,
    ).chunk(text)
