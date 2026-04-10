"""HierarchicalSummary v2.

This module intentionally mirrors the public interface of `src.hierarchical`
while implementing the newer pipeline design (field-wise merge, semantic
chunking mode, and usage/telemetry tracking).
"""

import asyncio
import contextvars
import hashlib
import inspect
import json
import math
import os
import re
import tempfile
import time
import urllib.request
from collections import Counter, OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Collection, Dict, List, Optional, Tuple, Union, get_args, get_origin

import nest_asyncio
from loguru import logger
from pydantic import Field, create_model, field_validator
from pydantic.fields import PydanticUndefined
from symai import Symbol
from symai.components import ChonkieChunker, DynamicEngine, FileReader, Function
from symai.core_ext import bind
from symai.models import LLMDataModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)
from tiktoken import Encoding
from tokenizers import Tokenizer

from .functions import SchemaValidationError, ValidatedFunction
from .hierarchical_v2_chunking import is_substantive_chunk, split_structural_segments
from .hierarchical_v2_io import normalize_reader_content
from .hierarchical_v2_reduce import (
    dedupe_unbounded_summary_text,
    deduplicate_string_items,
    normalize_text_key,
)
from .memoization import get_memoization_manager
from .types import TYPE_SPECIFIC_PROMPTS, DocumentType


class AssetMetadata(LLMDataModel):
    document_type: Optional[str] = Field(
        default=None,
        description="Best-effort detected document/content type for the asset.",
    )
    title: Optional[str] = Field(
        default=None,
        description="Best-effort asset title when the source exposes one.",
    )
    authors: Optional[List[str]] = Field(
        default=None,
        description="Best-effort author list for authored documents when available.",
    )
    speakers: Optional[List[str]] = Field(
        default=None,
        description="Best-effort speaker list for interview/talk/podcast-like assets when available.",
    )
    publisher_or_collection: Optional[str] = Field(
        default=None,
        description="Best-effort publisher, outlet, or source collection when available.",
    )
    publication_year: Optional[str] = Field(
        default=None,
        description="Best-effort publication year when available.",
    )

    @field_validator("publication_year", mode="before")
    @classmethod
    def coerce_year_to_str(cls, v):
        if isinstance(v, (int, float)):
            return str(int(v))
        return v


class Summary(LLMDataModel):
    summary: str = Field(
        description="An extremely comprehensive summary of the document. Do not start with 'This document is about...' or similar phrases."
    )
    facts: List[str] = Field(
        description="Important facts and subjects extracted from the document."
    )
    quotes: Optional[List[str]] = Field(
        default=None,
        description="Significant quotes extracted from the document verbatim if there are any.",
    )
    asset_metadata: Optional[AssetMetadata] = Field(
        default=None,
        description=(
            "Optional asset-level metadata such as detected document type, title, authors, speakers, "
            "publisher/collection, and publication year."
        ),
    )
    type: Optional[str] = None

    def validate():
        # TODO: validate that quotes are verbatim from the document
        pass


def gather(chunks: List[LLMDataModel]) -> Dict[str, Any]:
    """Compatibility helper used by older call sites.

    The implementation keeps old behavior (concat strings / extend lists), but the
    current hierarchical pipeline does not depend on this function anymore.
    """
    res_dict: Dict[str, Any] = {}
    for chunk in chunks:
        for field_name, field_type in type(chunk).model_fields.items():
            if getattr(field_type, "exclude", False):
                continue
            value = getattr(chunk, field_name, None)
            if isinstance(value, list):
                if field_name not in res_dict:
                    res_dict[field_name] = []
                res_dict[field_name].extend(value)
            elif isinstance(value, str):
                if field_name not in res_dict:
                    res_dict[field_name] = ""
                if value.strip():
                    res_dict[field_name] += value + "\n"
            elif value is not None and field_name not in res_dict:
                res_dict[field_name] = value
    return res_dict


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        current_loop = asyncio.get_event_loop()
        if current_loop.is_closed():
            raise RuntimeError("Event loop is closed.")
        return current_loop
    except RuntimeError:
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)
        return new_loop


def _sanitize_model_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _safe_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_safe_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v) for k, v in value.items()}
    if isinstance(value, LLMDataModel):
        return value.model_dump(mode="json")
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump(mode="json")
        except Exception:
            pass
    return str(value)


def get_current_tokenizer(tokenizer_name: str = "gpt2") -> Optional[Any]:
    """Best-effort tokenizer resolver used for fast token counting.

    Tries tiktoken first and falls back to HuggingFace `tokenizers`.
    """
    manager = get_memoization_manager()
    cache = manager.get_cache("tokenizers")
    cache_key = f"tokenizer::{tokenizer_name}"
    tokenizer = cache.get(cache_key)
    if tokenizer is not None:
        return tokenizer

    resolved = None

    try:
        import tiktoken

        resolved = tiktoken.get_encoding(tokenizer_name)
    except Exception:
        resolved = None

    if resolved is None:
        try:
            resolved = Tokenizer.from_pretrained(tokenizer_name)
        except Exception:
            resolved = None

    if resolved is not None:
        cache.put(cache_key, resolved)

    return resolved


@dataclass(frozen=True)
class DocTypeProfile:
    """Per-document-type tuning profile for chunking and extraction behavior."""

    name: str
    semantic_target_base_chunks: int = 5
    semantic_target_boosts: Tuple[Tuple[int, int], ...] = (
        (9000, 7),
        (12000, 8),
        (16000, 9),
    )
    semantic_target_cap: int = 12
    list_field_bonus_threshold: int = 6
    list_field_bonus: int = 1

    semantic_min_chunk_multiplier: float = 2.50
    semantic_rechunk_multiplier: float = 2.20
    max_rechunk_passes: int = 5

    # (soft_q, soft_fill, hard_q, hard_fill)
    boundary_unbounded: Tuple[float, float, float, float] = (0.25, 0.60, 0.08, 0.40)
    boundary_bounded: Tuple[float, float, float, float] = (0.30, 0.50, 0.14, 0.28)
    list_field_soft_q_bonus: float = 0.01
    list_field_soft_fill_penalty: float = 0.01

    split_by_lines: bool = False
    prompt_addendum: str = ""


DOC_TYPE_PROFILE_PRESETS: Dict[str, DocTypeProfile] = {
    "default": DocTypeProfile(name="default"),
    "conversational": DocTypeProfile(
        name="conversational",
        semantic_target_base_chunks=7,
        semantic_target_boosts=((6000, 8), (9000, 9), (12000, 10), (16000, 11)),
        semantic_target_cap=14,
        semantic_min_chunk_multiplier=1.75,
        semantic_rechunk_multiplier=1.45,
        max_rechunk_passes=6,
        boundary_unbounded=(0.40, 0.44, 0.24, 0.22),
        boundary_bounded=(0.44, 0.40, 0.26, 0.20),
        split_by_lines=True,
        prompt_addendum=dedent(
            """
            - Preserve speaker-specific points and keep statements from different speakers separate.
            - Avoid collapsing repeated topics when framing, intent, or stance differs.
            - Keep chronology and conversational context (what was asked vs answered) explicit.
            """
        ).strip(),
    ),
    "research": DocTypeProfile(
        name="research",
        semantic_target_base_chunks=6,
        semantic_target_boosts=((9000, 7), (12000, 8), (16000, 9), (22000, 10)),
        semantic_target_cap=12,
        semantic_min_chunk_multiplier=2.40,
        semantic_rechunk_multiplier=2.10,
        max_rechunk_passes=5,
        boundary_unbounded=(0.25, 0.58, 0.08, 0.38),
        boundary_bounded=(0.30, 0.50, 0.14, 0.28),
        prompt_addendum=dedent(
            """
            - Keep methods, data characteristics, outcomes, and limitations distinct.
            - Prefer keeping multiple related findings separate when metrics or subgroups differ.
            - Preserve study setup details (cohorts, time windows, endpoints, and controls).
            """
        ).strip(),
    ),
    "slides": DocTypeProfile(
        name="slides",
        semantic_target_base_chunks=6,
        semantic_target_boosts=((7000, 7), (10000, 8), (14000, 9)),
        semantic_target_cap=12,
        semantic_min_chunk_multiplier=2.00,
        semantic_rechunk_multiplier=1.75,
        max_rechunk_passes=5,
        boundary_unbounded=(0.34, 0.48, 0.18, 0.26),
        boundary_bounded=(0.38, 0.44, 0.22, 0.24),
        split_by_lines=True,
        prompt_addendum=dedent(
            """
            - Preserve slide-level bullet points separately where they carry distinct claims.
            - Keep figures, numbers, and headline takeaways explicit and not blended.
            """
        ).strip(),
    ),
}

# Extend by adding a preset above and mapping the document type here.
DOC_TYPE_PROFILE_BY_DOCUMENT_TYPE: Dict[DocumentType, str] = {
    DocumentType.INTERVIEW: "conversational",
    DocumentType.TALK: "conversational",
    DocumentType.KEYNOTE: "conversational",
    DocumentType.PODCAST: "conversational",
    DocumentType.SCIENTIFIC_PAPER: "research",
    DocumentType.REVIEW_PAPER: "research",
    DocumentType.REPORT: "research",
    DocumentType.ARTICLE: "research",
    DocumentType.WIKI: "research",
    DocumentType.PRESENTATION_SLIDES: "slides",
}


class HierarchicalSummaryV2(ValidatedFunction):
    def __init__(
        self,
        file_link: str = None,
        content: str = None,
        document_name: str = None,
        document_lang: str = None,
        asset_name: str = None,
        data_model: LLMDataModel = Summary,
        min_num_chunks: int = 5,
        min_chunk_size: int = 250,
        max_chunk_size: int = 1000,
        max_output_tokens: Optional[int] = None,
        user_prompt: str = None,
        include_quotes: bool = False,
        tokenizer_name: str = "gpt2",
        chunker_name: str = "SemanticHybridChunker",
        seed: int = 42,
        enable_initial_compression: bool = True,
        plain_text_only: bool = False,
        engine: Optional[object] = None,
        document_level_fields: Collection[str] | None = None,
        field_guidance: Optional[str] = None,
        *args,
        **kwargs,
    ):
        assert (file_link and not content) or (content and not file_link)
        if document_name is None and asset_name is not None:
            document_name = asset_name
        if content is not None:
            assert document_name is not None
        assert issubclass(data_model, LLMDataModel)

        super().__init__(data_model=data_model, retry_count=3, *args, **kwargs)

        self.document_lang = document_lang
        self.file_link = file_link
        self.min_num_chunks = min_num_chunks
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.max_output_tokens = int(max_output_tokens) if max_output_tokens is not None and int(max_output_tokens) > 0 else None
        self.user_prompt = user_prompt
        self.include_quotes = include_quotes
        self.seed = seed
        self.tokenizer_name = tokenizer_name
        self.enable_initial_compression = enable_initial_compression
        self.plain_text_only = plain_text_only
        self.engine = engine
        self.field_guidance = field_guidance
        self.document_level_fields = self._normalize_document_level_fields(document_level_fields)
        self._chunk_data_model = self._derive_chunk_data_model()
        self._document_level_data_model = self._derive_document_level_data_model()

        # Token accounting metrics used by debug scripts.
        self._token_offset = 0
        self._total_tokens_processed = 0
        self._last_prompt_token_estimate = 0
        self._usage = self._init_usage_tracking()
        self._token_count_cache: OrderedDict[Any, int] = OrderedDict()
        self._token_count_cache_max = 4096

        if file_link is not None:
            if file_link.startswith("http"):
                file_content, file_name = self.download_file(file_link)
            else:
                file_content, file_name = self.read_file(file_link)
        else:
            file_name = document_name
            file_content = str(content)

        self.content_only = str(file_content)
        self.content = f"[[DOCUMENT::{file_name}]]: <<<\n{self.content_only}\n>>>\n"

        self._chunker: Optional[ChonkieChunker] = None
        self.chunker_type = chunker_name

        self.document_type = None
        self._usage["chunking"]["requested_strategy"] = self._requested_chunking_strategy()
        self._refresh_doc_profile_usage()

    @property
    def chunker(self) -> ChonkieChunker:
        if self._chunker is None:
            self._chunker = ChonkieChunker(tokenizer_name=self.tokenizer_name)
        return self._chunker

    def _init_usage_tracking(self) -> Dict[str, Any]:
        return {
            "runtime": {
                "started_at": time.time(),
                "ended_at": None,
                "total_seconds": 0.0,
                "steps": {},
            },
            "tokens": {
                "total_processed": 0,
                "offset": 0,
                "net_tokens": 0,
                "by_operation": {},
                "cache_hits": 0,
                "cache_misses": 0,
            },
            "embedding": {
                "enabled": False,
                "requests": 0,
                "items": 0,
                "dropped_items": 0,
                "input_tokens": 0,
                "vector_dimensions": 0,
                "seconds": 0.0,
                "failed_requests": 0,
            },
            "llm": {
                "calls_total": 0,
                "calls_by_stage": {
                    "map": 0,
                    "document_level": 0,
                    "merge": 0,
                    "type_detection": 0,
                    "language_detection": 0,
                },
                "estimated_input_tokens": 0,
                "estimated_output_tokens": 0,
            },
            "chunking": {
                "requested_strategy": None,
                "effective_strategy": None,
                "fallback_reason": None,
                "profile_name": None,
                "strategy": None,
                "segments": 0,
                "chunks": 0,
                "chunk_size_target": 0,
                "effective_chunk_size_target": 0,
                "target_min_chunks": 0,
                "post_pack_rechunk_passes": 0,
                "semantic_boundary_threshold_soft": None,
                "semantic_boundary_threshold_hard": None,
            },
        }

    @property
    def usage(self) -> Dict[str, Any]:
        return self._usage

    @property
    def usage_report(self) -> Dict[str, Any]:
        return self._usage

    @contextmanager
    def _track_step(self, step_name: str):
        logger.info(f"[summarize-usage] step={step_name} started")
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            step_stats = self._usage["runtime"]["steps"].setdefault(
                step_name, {"seconds": 0.0, "calls": 0}
            )
            step_stats["seconds"] += elapsed
            step_stats["calls"] += 1
            logger.info(f"[summarize-usage] step={step_name} seconds={elapsed:.3f}")

    def _record_llm_call(
        self,
        stage: str,
        *,
        estimated_input_tokens: int = 0,
        estimated_output_tokens: int = 0,
    ) -> None:
        llm_stats = self._usage["llm"]
        llm_stats["calls_total"] += 1
        llm_stats["calls_by_stage"][stage] = llm_stats["calls_by_stage"].get(stage, 0) + 1
        llm_stats["estimated_input_tokens"] += max(0, int(estimated_input_tokens))
        llm_stats["estimated_output_tokens"] += max(0, int(estimated_output_tokens))

    def _token_cache_key(self, text: str, mode: str) -> Any:
        # Avoid storing large raw strings as cache keys.
        if len(text) <= 512:
            return (mode, text)
        digest = hashlib.blake2b(text.encode("utf-8", errors="ignore"), digest_size=12).hexdigest()
        return (mode, len(text), digest)

    def _token_cache_get(self, key: Any) -> Optional[int]:
        if key in self._token_count_cache:
            value = self._token_count_cache.pop(key)
            self._token_count_cache[key] = value
            self._usage["tokens"]["cache_hits"] += 1
            return value
        self._usage["tokens"]["cache_misses"] += 1
        return None

    def _token_cache_put(self, key: Any, value: int) -> None:
        if key in self._token_count_cache:
            self._token_count_cache.pop(key)
        self._token_count_cache[key] = int(value)
        while len(self._token_count_cache) > self._token_count_cache_max:
            self._token_count_cache.popitem(last=False)

    def _estimate_tokens_quiet(self, data: Any) -> int:
        text = self._serialize_for_token_count(data)
        cache_key = self._token_cache_key(text, mode="quiet")
        cached = self._token_cache_get(cache_key)
        if cached is not None:
            return cached

        tokenizer = get_current_tokenizer(self.tokenizer_name)

        if tokenizer is None:
            count = max(1, int(len(text) / 4))
            self._token_cache_put(cache_key, count)
            return count

        try:
            if isinstance(tokenizer, Tokenizer):
                count = len(tokenizer.encode(text).ids)
            else:
                count = len(tokenizer.encode(text))
            self._token_cache_put(cache_key, count)
            return count
        except Exception:
            count = max(1, int(len(text) / 4))
            self._token_cache_put(cache_key, count)
            return count

    def _approximate_token_count_from_text(self, text: str) -> int:
        if not text:
            return 0

        chars = len(text)
        words = len(re.findall(r"\S+", text))
        punctuation = len(re.findall(r"[\[\]\{\}\(\):;,\"']", text))
        newline_count = text.count("\n")

        chars_per_token = 3.4 if punctuation > max(12, chars * 0.08) else 4.0
        char_estimate = chars / chars_per_token
        word_estimate = words * 1.18
        structure_bonus = min(chars * 0.03, punctuation * 0.10 + newline_count * 0.15)

        return max(1, int(round(max(char_estimate, word_estimate) + structure_bonus)))

    def _estimate_tokens_approx(self, data: Any) -> int:
        text = self._serialize_for_token_count(data)
        cache_key = self._token_cache_key(text, mode="approx")
        cached = self._token_cache_get(cache_key)
        if cached is not None:
            return cached

        count = self._approximate_token_count_from_text(text)
        self._token_cache_put(cache_key, count)
        return count

    def _estimate_tokens_near_threshold(
        self,
        data: Any,
        *,
        threshold: int,
        margin_ratio: float = 0.18,
        min_margin: int = 48,
        exact_mode: str = "quiet",
    ) -> int:
        approx = self._estimate_tokens_approx(data)
        threshold = max(1, int(threshold))
        margin = max(int(threshold * margin_ratio), min_margin)

        if approx < max(1, threshold - margin) or approx > threshold + margin:
            return approx

        if exact_mode == "fast":
            return self._fast_path_token_count(data)
        return self._estimate_tokens_quiet(data)

    def _estimate_tokens_for_chunk_boundary(self, data: Any, chunk_size: int) -> int:
        return self._estimate_tokens_near_threshold(
            data,
            threshold=max(64, int(chunk_size)),
            margin_ratio=0.20,
            min_margin=64,
            exact_mode="quiet",
        )

    def _log_usage_summary(self) -> None:
        self._usage["tokens"]["total_processed"] = self._total_tokens_processed
        self._usage["tokens"]["offset"] = self._token_offset
        self._usage["tokens"]["net_tokens"] = self._total_tokens_processed - self._token_offset
        self._usage["runtime"]["ended_at"] = time.time()
        started_at = self._usage["runtime"]["started_at"]
        self._usage["runtime"]["total_seconds"] = max(0.0, self._usage["runtime"]["ended_at"] - started_at)

        llm = self._usage["llm"]
        emb = self._usage["embedding"]
        logger.info(
            "[summarize-usage] total_seconds={:.3f} llm_calls={} llm_in_tokens={} llm_out_tokens={} "
            "emb_requests={} emb_items={} emb_input_tokens={}".format(
                self._usage["runtime"]["total_seconds"],
                llm["calls_total"],
                llm["estimated_input_tokens"],
                llm["estimated_output_tokens"],
                emb["requests"],
                emb["items"],
                emb["input_tokens"],
            )
        )

    def _normalize_reader_content(self, content: Any) -> str:
        return normalize_reader_content(content, _safe_jsonable)

    def read_file(self, file_link: str) -> Tuple[str, str]:
        logger.info(f"Reading file from {file_link}")
        with self._track_step("read_file"):
            if self.plain_text_only:
                with open(file_link, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            else:
                reader = FileReader()
                content = reader(file_link, backend="markitdown")

        file_name = os.path.basename(file_link)
        return self._normalize_reader_content(content), file_name

    def download_file(self, file_link: str) -> Tuple[str, str]:
        logger.info(f"Downloading file from {file_link}")
        with self._track_step("download_file"):
            with urllib.request.urlopen(file_link) as f:
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_file.write(f.read())
                    tmp_file.flush()
                    tmp_file_name = tmp_file.name

        content, file_name = self.read_file(tmp_file_name)
        os.remove(tmp_file_name)
        return content, file_name

    def _prompt_for_schema(self, schema: type[LLMDataModel]) -> str:
        type_specific_prompt = ""
        if self.document_type and self.document_type in TYPE_SPECIFIC_PROMPTS:
            type_specific_prompt = dedent(
                f"""[Type-Specific Instructions]
            For this {self.document_type.value}: {TYPE_SPECIFIC_PROMPTS[self.document_type]}"""
            )
        profile_specific_prompt = ""
        profile = self._active_doc_profile()
        if profile.prompt_addendum:
            profile_specific_prompt = dedent(
                f"""[Document Profile Guidance]
            Profile: {profile.name}
            {profile.prompt_addendum}"""
            )

        user_prompt = ""
        if self.user_prompt is not None:
            user_prompt = dedent(
                f"""[Goal-specific Instructions]
            This summary is intended for a specific audience or purpose, which is defined in <purpose> below.
            In addition to general summarization, prioritize information relevant to this purpose for all schema fields.

            <purpose>
            {self.user_prompt}
            </purpose>"""
            )
        asset_metadata_prompt = self._asset_metadata_prompt_guidance(schema)

        field_guidance_prompt = ""
        if self.field_guidance:
            field_guidance_prompt = dedent(
                f"""[Schema Field Guidance]
            {self.field_guidance}"""
            )

        prompt_text = dedent(
            f"""
            [[Document Processing Task]]

            [Main Objective]
            Create a highly detailed and reliable extraction of the provided content and return it as JSON.
            The content can be split into chunks, and each chunk may be processed separately.
            The final result must satisfy the provided JSON schema and stay faithful to the source.
            The type of content is specified in [[CONTENT TYPE]].

            {type_specific_prompt}
            {profile_specific_prompt}
            {asset_metadata_prompt}

            {user_prompt}

            {field_guidance_prompt}

            [Language Requirements]
            The output must be in the language specified in [[CONTENT LANGUAGE]], regardless of source language.

            [Key Requirements]
            - Capture all salient information relevant to the schema field descriptions.
            - Prefer high recall over abstraction at chunk level; do not prematurely compress nuanced points.
            - Keep field semantics distinct according to schema.
            - Preserve factual consistency between fields.
            - For quote-like fields, keep quotations verbatim whenever possible.
            - Keep atomic details explicit:
              dates/years, numbers/percentages, named entities, countries/regions, product and policy terms.
            - Preserve evaluative qualifiers and uncertainty statements (for example: better-than-expected outcomes,
              constrained forecasts, caveats, conditions).
            - If two statements are similar but differ in timeframe, scope, cause, metric, or sentiment, keep both.
            - For interview-like content, retain speaker-specific claims and framing rather than collapsing to generic prose.
        """
        )
        prompt_text += schema.instruct_llm()
        return prompt_text

    @property
    def prompt(self) -> str:
        return self._prompt_for_schema(self._map_data_model())

    @property
    def static_context(self) -> str:
        return dedent(
            """
            Extract information from the provided text and return valid JSON that follows the schema exactly.
            The output language must match the requested output language.
            Prefer preserving concrete details over summarizing them away.
            Keep distinct factual points as separate items when the schema supports lists.
        """
        )

    @bind(engine="neurosymbolic", property="compute_required_tokens")(lambda: 0)
    def _compute_required_tokens(self):
        pass

    @bind(engine="neurosymbolic", property="max_context_tokens")
    def _max_context_tokens(_):
        pass

    @bind(engine="neurosymbolic", property="max_response_tokens")
    def _max_response_tokens(_):
        pass

    @bind(engine="neurosymbolic", property="compute_remaining_tokens")(lambda: 0)
    def _compute_remaining_tokens(self):
        pass

    def _log_token_usage(self, tokens_added: int, operation: str = "processing") -> None:
        tokens_added = max(0, int(tokens_added))
        self._total_tokens_processed += tokens_added
        by_operation = self._usage["tokens"]["by_operation"]
        by_operation[operation] = by_operation.get(operation, 0) + tokens_added
        logger.debug(
            f"Token usage - {operation}: +{tokens_added} tokens (total={self._total_tokens_processed}, offset={self._token_offset})"
        )

    def get_token_stats(self) -> Dict[str, int]:
        return {
            "total_processed": self._total_tokens_processed,
            "offset": self._token_offset,
            "net_tokens": self._total_tokens_processed - self._token_offset,
            "cache_hits": int(self._usage["tokens"].get("cache_hits", 0)),
            "cache_misses": int(self._usage["tokens"].get("cache_misses", 0)),
        }

    def _serialize_for_token_count(self, data: Any) -> str:
        if data is None:
            return ""
        if isinstance(data, str):
            return data
        if isinstance(data, LLMDataModel):
            return data.model_dump_json(indent=None)
        if hasattr(data, "model_dump_json"):
            try:
                return data.model_dump_json(indent=None)
            except Exception:
                pass
        try:
            return json.dumps(_safe_jsonable(data), ensure_ascii=False)
        except Exception:
            return str(data)

    def _fast_path_token_count(self, data: Any) -> int:
        text = self._serialize_for_token_count(data)
        cache_key = self._token_cache_key(text, mode="fast")
        cached = self._token_cache_get(cache_key)
        if cached is not None:
            return cached

        tokenizer = get_current_tokenizer(self.tokenizer_name)

        if tokenizer is None:
            count = max(1, int(len(text) / 4))
            self._token_cache_put(cache_key, count)
            self._log_token_usage(count, operation="fallback_token_count")
            return count

        try:
            if isinstance(tokenizer, Tokenizer):
                count = len(tokenizer.encode(text).ids)
            else:
                # tiktoken encoding
                count = len(tokenizer.encode(text))
            self._token_cache_put(cache_key, count)
            self._log_token_usage(count, operation="fast_token_count")
            return count
        except Exception:
            count = max(1, int(len(text) / 4))
            self._token_cache_put(cache_key, count)
            self._log_token_usage(count, operation="fallback_token_count_error")
            return count

    def compute_required_tokens(self, data: Any, count_context: bool = True) -> int:
        if not count_context:
            return self._fast_path_token_count(data)

        preview_function = Function(
            prompt=self.prompt,
            static_context=self.static_context,
            dynamic_context=self.dynamic_context,
        )

        preview = preview_function(
            data,
            preview=True,
            response_format={"type": "json_object"},
            seed=self.seed,
        )

        count = self._compute_required_tokens(preview.prop.prepared_input)
        self._log_token_usage(count, operation="prompt_token_count")
        return count

    def split_words(self, text: str) -> List[str]:
        return re.split(r"(\W+)", text)

    def _is_substantive_chunk(self, text: str, *, min_alnum_chars: int = 40) -> bool:
        return is_substantive_chunk(text, min_alnum_chars=min_alnum_chars)

    def _filter_non_substantive_chunks(self, chunks: List[Any]) -> List[str]:
        normalized = [str(c) for c in chunks if str(c).strip()]
        if not normalized:
            return []
        substantive = [c for c in normalized if self._is_substantive_chunk(c)]
        removed = len(normalized) - len(substantive)
        if removed > 0:
            logger.debug(
                f"Filtered {removed} non-substantive chunk(s) (wrapper/purpose-only)."
            )
        return substantive if substantive else normalized

    def _select_detection_chunk(self, chunks: List[str]) -> str:
        if not chunks:
            return self.content_only if self.content_only else self.content
        substantive = [c for c in chunks if self._is_substantive_chunk(c, min_alnum_chars=20)]
        pool = substantive if substantive else chunks
        return max(pool, key=lambda c: self._estimate_tokens_approx(c))

    def _split_structural_segments(self, text: str) -> List[str]:
        segments = split_structural_segments(text)
        profile = self._active_doc_profile()
        if not profile.split_by_lines:
            return segments
        expanded: List[str] = []
        for segment in segments:
            lines = [line.strip() for line in re.split(r"\n+", segment) if line.strip()]
            if len(lines) > 1:
                expanded.extend(lines)
            else:
                expanded.append(segment)
        return expanded if expanded else segments

    def _active_doc_profile(self) -> DocTypeProfile:
        profile_key = DOC_TYPE_PROFILE_BY_DOCUMENT_TYPE.get(self.document_type, "default")
        return DOC_TYPE_PROFILE_PRESETS.get(profile_key, DOC_TYPE_PROFILE_PRESETS["default"])

    def _refresh_doc_profile_usage(self) -> None:
        self._usage["chunking"]["profile_name"] = self._active_doc_profile().name

    def _normalize_document_level_fields(
        self, document_level_fields: Collection[str] | None
    ) -> tuple[str, ...]:
        if not document_level_fields:
            return ()

        requested = tuple(dict.fromkeys(str(field) for field in document_level_fields))
        available = {
            name
            for name, field in self.data_model.model_fields.items()
            if not getattr(field, "exclude", False)
        }
        unknown = sorted(set(requested) - available)
        if unknown:
            raise ValueError(
                f"Unknown document_level_fields for {self.data_model.__name__}: {', '.join(unknown)}"
            )
        return requested

    def _derive_subset_model(
        self,
        *,
        include_fields: Collection[str] | None = None,
        exclude_fields: Collection[str] | None = None,
        model_name_suffix: str,
    ) -> type[LLMDataModel]:
        include_set = set(include_fields) if include_fields is not None else None
        exclude_set = set(exclude_fields or ())
        fields: Dict[str, tuple[Any, Any]] = {}

        for field_name, field_info in self.data_model.model_fields.items():
            if getattr(field_info, "exclude", False):
                continue
            if include_set is not None and field_name not in include_set:
                continue
            if field_name in exclude_set:
                continue
            fields[field_name] = (field_info.annotation, field_info)

        model_name = f"{self.data_model.__name__}{model_name_suffix}"
        return create_model(model_name, __base__=LLMDataModel, **fields)

    def _derive_chunk_data_model(self) -> type[LLMDataModel]:
        if not self.document_level_fields:
            return self.data_model
        return self._derive_subset_model(
            exclude_fields=self.document_level_fields,
            model_name_suffix="Chunk",
        )

    def _derive_document_level_data_model(self) -> type[LLMDataModel] | None:
        if not self.document_level_fields:
            return None
        return self._derive_subset_model(
            include_fields=self.document_level_fields,
            model_name_suffix="DocumentLevel",
        )

    def _map_data_model(self) -> type[LLMDataModel]:
        return self._chunk_data_model

    def _document_data_model(self) -> type[LLMDataModel] | None:
        return self._document_level_data_model

    def _schema_contains_field(self, schema: type[LLMDataModel], field_name: str) -> bool:
        field_info = schema.model_fields.get(field_name)
        return field_info is not None and not getattr(field_info, "exclude", False)

    def _asset_metadata_prompt_guidance(self, schema: type[LLMDataModel]) -> str:
        if not self._schema_contains_field(schema, "asset_metadata"):
            return ""
        return dedent(
            """
            [Asset Metadata Guidance]
            - If the schema includes `asset_metadata`, populate it with only source-supported document-level metadata.
            - Always set `asset_metadata.document_type` when the detected content type is clear.
            - For article/report/scientific paper/review/wiki/book content, prioritize title, authors, publisher_or_collection, and publication_year.
            - For interview/podcast/talk/keynote/presentation content, prioritize speakers and keep speaker-specific claims separated in facts/quotes.
            - Leave unknown metadata fields null instead of inventing values.
            """
        )

    def _pre_chunk_type_detection_input(self, max_chars: int = 12000) -> str:
        text = (self.content_only if self.content_only else self.content) or ""
        text = str(text).strip()
        if len(text) <= max_chars:
            return text
        head_chars = int(max_chars * 0.70)
        tail_chars = max(512, max_chars - head_chars)
        return text[:head_chars] + "\n\n...\n\n" + text[-tail_chars:]

    def _semantic_mode_enabled(self) -> bool:
        return "semantic" in str(self.chunker_type).lower()

    def _requested_chunking_strategy(self) -> str:
        return "semantic_hybrid" if self._semantic_mode_enabled() else "structural"

    def _chunker_name_for_fallback_split(self) -> str:
        if self._semantic_mode_enabled():
            return "RecursiveChunker"
        return self.chunker_type

    def _schema_list_field_count(self) -> int:
        count = 0
        for field_info in self._map_data_model().model_fields.values():
            if getattr(field_info, "exclude", False):
                continue
            if self._annotation_is_list(field_info.annotation):
                count += 1
        return count

    def _make_extraction_function(
        self, schema: type[LLMDataModel], *, prompt_text: str | None = None
    ) -> ValidatedFunction:
        resolved_prompt = prompt_text or self._prompt_for_schema(schema)
        return ValidatedFunction(
            data_model=schema,
            retry_count=self.retry_count,
            prompt=resolved_prompt,
            static_context=self.static_context,
            dynamic_context=self.dynamic_context,
        )

    def _merge_overlay(self, base: Dict[str, Any], overlay: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in overlay.items():
            if value is None:
                continue
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                self._merge_overlay(base[key], value)
            else:
                base[key] = value
        return base

    def _extract_document_level_fields(self, payload: str) -> Dict[str, Any]:
        schema = self._document_data_model()
        if schema is None:
            return {}

        prompt_text = self._prompt_for_schema(schema)
        fn = self._make_extraction_function(schema, prompt_text=prompt_text)
        res = self._call_validated_function(
            fn,
            payload,
            stage="document_level",
            prompt_text=prompt_text,
        )
        return res.model_dump(mode="json", exclude_none=True)

    def _semantic_target_map_chunks(self, total_tokens: int, prompt_tokens: int) -> int:
        if not self._semantic_mode_enabled():
            return max(1, self.min_num_chunks)

        profile = self._active_doc_profile()
        target = max(profile.semantic_target_base_chunks, self.min_num_chunks)
        for token_threshold, suggested_chunks in profile.semantic_target_boosts:
            if total_tokens >= token_threshold:
                target = max(target, suggested_chunks)

        list_fields = self._schema_list_field_count()
        if list_fields >= profile.list_field_bonus_threshold:
            target += profile.list_field_bonus

        target = min(target, profile.semantic_target_cap)

        min_by_context = max(2, int(self._max_context_tokens() / max(512, prompt_tokens + 256)))
        target = min(target, max(2, min_by_context))

        return max(self.min_num_chunks, target)

    def _semantic_min_chunk_tokens(self, prompt_tokens: int) -> int:
        # Keep content payload larger than instructions while allowing finer map granularity.
        profile = self._active_doc_profile()
        multiplier = profile.semantic_min_chunk_multiplier if self._semantic_mode_enabled() else 1.75
        return max(self.min_chunk_size, int(max(128, prompt_tokens) * multiplier))

    def _semantic_rechunk_floor(self, prompt_tokens: int) -> int:
        profile = self._active_doc_profile()
        multiplier = profile.semantic_rechunk_multiplier if self._semantic_mode_enabled() else 1.75
        return max(self.min_chunk_size, int(max(128, prompt_tokens) * multiplier))

    def _quantile(self, values: List[float], q: float) -> float:
        if not values:
            return 0.0
        qq = min(1.0, max(0.0, q))
        idx = int(round(qq * (len(values) - 1)))
        return values[idx]

    def _semantic_boundary_config(self) -> Tuple[float, float, float, float]:
        profile = self._active_doc_profile()
        list_fields = self._schema_list_field_count()
        if not self._has_output_limit():
            soft_q, soft_fill, hard_q, hard_fill = profile.boundary_unbounded
        else:
            soft_q, soft_fill, hard_q, hard_fill = profile.boundary_bounded

        if list_fields >= profile.list_field_bonus_threshold:
            soft_q += profile.list_field_soft_q_bonus
            soft_fill -= profile.list_field_soft_fill_penalty

        soft_q = min(0.50, max(0.25, soft_q))
        soft_fill = min(0.60, max(0.22, soft_fill))
        hard_q = min(soft_q - 0.05, max(0.08, hard_q))
        hard_fill = min(0.40, max(0.12, hard_fill))
        return soft_q, soft_fill, hard_q, hard_fill

    def _embedding_item_token_limit(self) -> int:
        """Conservative per-item token limit for embedding requests.

        Some providers still reject tokenized embedding inputs beyond ~2048
        tokens even when model-level limits are higher, so keep a clear buffer.
        """
        return 1800

    def _embedding_batch_token_limit(self) -> int:
        """Conservative total token budget per embedding request."""
        return 12000

    def _embedding_batch_item_limit(self) -> int:
        """Conservative item cap per embedding request."""
        return 16

    def _sanitize_embedding_text(self, text: Any) -> str:
        raw = str(text or "")
        if not raw:
            return ""
        cleaned = raw.replace("\x00", " ")
        cleaned = "".join(
            ch if (ch >= " " or ch in "\n\r\t") else " " for ch in cleaned
        )
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    def _split_text_for_embedding(self, text: str, token_limit: int) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []

        if self._estimate_tokens_quiet(text) <= token_limit:
            return [text]

        split_target = max(700, min(2200, int(token_limit * 0.55)))
        parts: List[str] = []

        try:
            split = self.chunker(
                data=Symbol(text),
                chunker_name=self._chunker_name_for_fallback_split(),
                chunk_size=split_target,
            )
            parts = [str(s).strip() for s in split if str(s).strip()]
        except Exception:
            parts = []

        if not parts:
            # Last-resort character windows if chunker split fails.
            chars_per_token = 4
            window_chars = max(1200, token_limit * chars_per_token)
            overlap_chars = int(window_chars * 0.12)
            i = 0
            while i < len(text):
                chunk = text[i : i + window_chars].strip()
                if chunk:
                    parts.append(chunk)
                if i + window_chars >= len(text):
                    break
                i += max(200, window_chars - overlap_chars)

        # Ensure every part fits the embedding token limit.
        safe_parts: List[str] = []
        for part in parts:
            if self._estimate_tokens_quiet(part) <= token_limit:
                safe_parts.append(part)
                continue

            # Secondary fallback split by sentence windows.
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", part) if s.strip()]
            if len(sentences) <= 1:
                safe_parts.append(part[: token_limit * 4].strip())
                continue

            current: List[str] = []
            current_tokens = 0
            for sentence in sentences:
                st = self._estimate_tokens_quiet(sentence)
                if current and current_tokens + st > token_limit:
                    safe_parts.append(" ".join(current).strip())
                    current = [sentence]
                    current_tokens = st
                else:
                    current.append(sentence)
                    current_tokens += st
            if current:
                safe_parts.append(" ".join(current).strip())

        return [p for p in safe_parts if p]

    def _prepare_segments_for_semantic_embedding(self, segments: List[str]) -> List[str]:
        if not segments:
            return []

        token_limit = self._embedding_item_token_limit()
        expanded: List[str] = []
        expanded_count = 0

        for segment in segments:
            segment = self._sanitize_embedding_text(segment)
            if not segment:
                continue
            tokens = self._estimate_tokens_quiet(segment)
            if tokens <= token_limit:
                expanded.append(segment)
                continue

            parts = self._split_text_for_embedding(segment, token_limit=token_limit)
            if not parts:
                expanded.append(segment)
                continue

            expanded.extend(self._sanitize_embedding_text(part) for part in parts)
            expanded_count += max(0, len(parts) - 1)

        self._usage["chunking"]["semantic_segments_expanded"] = expanded_count
        return [segment for segment in expanded if segment]

    def _normalize_embedding_vector(self, emb: Any) -> List[float]:
        # Some providers return [D], others [[D]].
        if isinstance(emb, list) and emb and isinstance(emb[0], list):
            emb = emb[0]
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        if not isinstance(emb, list):
            emb = list(emb)
        return [float(x) for x in emb]

    def _is_embedding_input_too_long_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "array length must be 2048 or less",
                "maximum context length",
                "context length exceeded",
                "too many tokens",
                "input is too long",
            )
        )

    def _average_embedding_vectors(self, vectors: List[List[float]]) -> List[float]:
        if not vectors:
            raise ValueError("Cannot average empty embedding vector list.")

        dims = len(vectors[0])
        if dims == 0:
            return []

        total = [0.0] * dims
        for vector in vectors:
            if len(vector) != dims:
                raise ValueError("Embedding vectors must have consistent dimensions.")
            for i, value in enumerate(vector):
                total[i] += float(value)

        averaged = [value / len(vectors) for value in total]
        norm = math.sqrt(sum(value * value for value in averaged))
        if norm > 1e-12:
            averaged = [value / norm for value in averaged]
        return averaged

    def _embed_oversize_text_with_aggregation(self, text: str) -> List[float]:
        split_limit = self._embedding_item_token_limit()
        parts = [
            self._sanitize_embedding_text(part)
            for part in self._split_text_for_embedding(text, token_limit=split_limit)
        ]
        parts = [part for part in parts if part]
        if len(parts) <= 1:
            raise ValueError("Unable to split oversized embedding input into smaller parts.")

        logger.debug(f"Retrying oversized embedding input by splitting into {len(parts)} parts.")

        vectors: List[List[float]] = []
        for batch in self._iter_embedding_batches(parts):
            vectors.extend(self._embed_symbolicai_batch(batch))

        if len(vectors) != len(parts):
            raise ValueError(
                f"Embedding response size mismatch while aggregating oversized input: "
                f"expected {len(parts)} vectors, got {len(vectors)}."
            )

        return self._average_embedding_vectors(vectors)

    def _iter_embedding_batches(self, texts: List[str]) -> List[List[str]]:
        batches: List[List[str]] = []
        batch: List[str] = []
        batch_tokens = 0
        max_batch_tokens = self._embedding_batch_token_limit()
        max_batch_items = self._embedding_batch_item_limit()

        for text in texts:
            text_tokens = self._estimate_tokens_approx(text)
            should_flush = (
                batch
                and (
                    len(batch) >= max_batch_items
                    or batch_tokens + text_tokens > max_batch_tokens
                )
            )
            if should_flush:
                batches.append(batch)
                batch = []
                batch_tokens = 0
            batch.append(text)
            batch_tokens += text_tokens

        if batch:
            batches.append(batch)

        return batches

    def _call_embedding_batch(self, texts: List[str]) -> List[List[float]]:
        emb_stats = self._usage["embedding"]
        emb_stats["requests"] += 1
        emb_start = time.perf_counter()
        try:
            result = Symbol(texts).embed()
            values = result.value if hasattr(result, "value") else result
            vectors = [self._normalize_embedding_vector(v) for v in values]
            elapsed = time.perf_counter() - emb_start
            emb_stats["seconds"] += elapsed
            emb_stats["vector_dimensions"] = len(vectors[0]) if vectors else 0
            return vectors
        except Exception:
            elapsed = time.perf_counter() - emb_start
            emb_stats["failed_requests"] += 1
            emb_stats["seconds"] += elapsed
            raise

    def _embed_symbolicai_batch(self, texts: List[str]) -> List[List[float]]:
        try:
            vectors = self._call_embedding_batch(texts)
        except Exception as exc:
            if len(texts) == 1 and self._is_embedding_input_too_long_error(exc):
                return [self._embed_oversize_text_with_aggregation(texts[0])]
            if len(texts) <= 1:
                raise
            midpoint = len(texts) // 2
            if midpoint <= 0:
                raise
            return self._embed_symbolicai_batch(texts[:midpoint]) + self._embed_symbolicai_batch(
                texts[midpoint:]
            )

        if len(vectors) != len(texts):
            raise ValueError(
                f"Embedding response size mismatch: expected {len(texts)} vectors, got {len(vectors)}."
            )
        return vectors

    def _embed_text_batch(self, texts: List[str]) -> Optional[List[List[float]]]:
        if not texts:
            return []

        self._usage["embedding"]["enabled"] = True
        emb_stats = self._usage["embedding"]
        sanitized_texts = [self._sanitize_embedding_text(text) for text in texts]
        dropped_items = sum(1 for text in sanitized_texts if not text)
        if dropped_items:
            emb_stats["dropped_items"] += dropped_items
        sanitized_texts = [text for text in sanitized_texts if text]
        if not sanitized_texts:
            return []

        emb_stats["items"] += len(sanitized_texts)
        emb_stats["input_tokens"] += sum(
            self._estimate_tokens_approx(text) for text in sanitized_texts
        )

        try:
            vectors: List[List[float]] = []
            for batch in self._iter_embedding_batches(sanitized_texts):
                vectors.extend(self._embed_symbolicai_batch(batch))
            emb_stats["vector_dimensions"] = len(vectors[0]) if vectors else 0
            logger.info(
                f"[summarize-usage] embeddings requests={emb_stats['requests']} "
                f"items={len(sanitized_texts)} dims={emb_stats['vector_dimensions']} "
                f"seconds={emb_stats['seconds']:.3f}"
            )
            return vectors
        except Exception as e:
            logger.warning(f"Semantic embedding failed, falling back to structural chunking: {e}")
            return None

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
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

    def _pack_segments_semantic(self, segments: List[str], chunk_size: int) -> List[str]:
        prepared_segments = self._prepare_segments_for_semantic_embedding(segments)
        vectors = self._embed_text_batch(prepared_segments)
        if not vectors or len(vectors) != len(prepared_segments):
            return []

        sims: List[float] = []
        for i in range(len(prepared_segments) - 1):
            sims.append(self._cosine_similarity(vectors[i], vectors[i + 1]))

        soft_q, soft_fill_ratio, hard_q, hard_fill_ratio = self._semantic_boundary_config()
        hard_threshold = 0.0
        soft_threshold = 0.0
        if sims:
            sorted_sims = sorted(sims)
            soft_threshold = self._quantile(sorted_sims, soft_q)
            hard_threshold = self._quantile(sorted_sims, hard_q)
        self._usage["chunking"]["semantic_boundary_threshold_soft"] = soft_threshold
        self._usage["chunking"]["semantic_boundary_threshold_hard"] = hard_threshold

        chunks: List[str] = []
        current_parts: List[str] = []
        current_tokens = 0

        for i, segment in enumerate(prepared_segments):
            segment_tokens = self._estimate_tokens_for_chunk_boundary(segment, chunk_size)

            if segment_tokens >= chunk_size:
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_tokens = 0
                try:
                    split = self.chunker(
                        data=Symbol(segment),
                        chunker_name=self._chunker_name_for_fallback_split(),
                        chunk_size=chunk_size,
                    )
                    chunks.extend([str(s) for s in split if str(s).strip()])
                except Exception:
                    chunks.append(segment)
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

    def _pack_segments_into_chunks(self, segments: List[str], chunk_size: int) -> List[str]:
        if not segments:
            return []

        chunks: List[str] = []
        current_parts: List[str] = []
        current_tokens = 0

        for segment in segments:
            segment_tokens = self._estimate_tokens_for_chunk_boundary(segment, chunk_size)
            if segment_tokens >= chunk_size:
                if current_parts:
                    chunks.append("\n\n".join(current_parts))
                    current_parts = []
                    current_tokens = 0

                try:
                    split = self.chunker(
                        data=Symbol(segment),
                        chunker_name=self._chunker_name_for_fallback_split(),
                        chunk_size=chunk_size,
                    )
                    chunks.extend([str(s) for s in split if str(s).strip()])
                except Exception:
                    chunks.append(segment)
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

    def chunk_by_token_count(self, text: str, chunk_size: int, include_context: bool = False) -> List[str]:
        del include_context
        with self._track_step("chunking"):
            if not self._usage["chunking"].get("requested_strategy"):
                self._usage["chunking"]["requested_strategy"] = self._requested_chunking_strategy()
            segments = self._split_structural_segments(text)

            self._usage["chunking"]["segments"] = len(segments)
            self._usage["chunking"]["chunk_size_target"] = chunk_size

            if len(segments) > 1:
                if self._semantic_mode_enabled():
                    self._usage["chunking"]["strategy"] = "semantic_hybrid"
                    self._usage["chunking"]["effective_strategy"] = "semantic_hybrid"
                    self._usage["chunking"]["fallback_reason"] = None
                    total_tokens = self._estimate_tokens_approx(text)
                    prompt_tokens = max(128, int(self._last_prompt_token_estimate or 0))
                    target_min_chunks = self._semantic_target_map_chunks(total_tokens, prompt_tokens)
                    self._usage["chunking"]["target_min_chunks"] = target_min_chunks

                    effective_chunk_size = chunk_size
                    rechunk_floor = self._semantic_rechunk_floor(prompt_tokens)
                    max_rechunk_passes = max(1, self._active_doc_profile().max_rechunk_passes)
                    chunks = self._pack_segments_semantic(segments, chunk_size=effective_chunk_size)
                    rechunk_passes = 0
                    while (
                        chunks
                        and len(chunks) < target_min_chunks
                        and effective_chunk_size > rechunk_floor
                        and rechunk_passes < max_rechunk_passes
                    ):
                        observed_ratio = len(chunks) / float(max(1, target_min_chunks))
                        if observed_ratio >= 0.85:
                            shrink = 0.90
                        elif observed_ratio >= 0.65:
                            shrink = 0.80
                        else:
                            shrink = 0.70

                        next_chunk_size = max(rechunk_floor, int(effective_chunk_size * shrink))
                        if next_chunk_size >= effective_chunk_size:
                            break
                        rechunk_passes += 1
                        effective_chunk_size = next_chunk_size
                        chunks = self._pack_segments_semantic(segments, chunk_size=effective_chunk_size)

                    self._usage["chunking"]["post_pack_rechunk_passes"] = rechunk_passes
                    self._usage["chunking"]["effective_chunk_size_target"] = effective_chunk_size
                    if chunks:
                        self._usage["chunking"]["chunks"] = len(chunks)
                        return chunks
                    self._usage["chunking"]["fallback_reason"] = "semantic_empty_or_failed"

                self._usage["chunking"]["strategy"] = "structural"
                self._usage["chunking"]["effective_strategy"] = "structural"
                chunks = self._pack_segments_into_chunks(segments, chunk_size=chunk_size)
                if chunks:
                    self._usage["chunking"]["chunks"] = len(chunks)
                    return chunks

            self._usage["chunking"]["strategy"] = "recursive_fallback"
            self._usage["chunking"]["effective_strategy"] = "recursive_fallback"
            chunks = self.chunker(
                data=Symbol(text),
                chunker_name=self._chunker_name_for_fallback_split(),
                chunk_size=chunk_size,
            )
            normalized = [str(c) for c in chunks if str(c).strip()]
            self._usage["chunking"]["chunks"] = len(normalized)
            return normalized

    async def summarize_chunks(self, chunks: List[str], **kwargs) -> Tuple[List[LLMDataModel], int]:
        schema = self._map_data_model()
        prompt_text = self._prompt_for_schema(schema)

        @retry(
            retry=retry_if_not_exception_type(SchemaValidationError),
            wait=wait_exponential_jitter(initial=0.25, max=60),
            stop=stop_after_attempt(5),
            before_sleep=before_sleep_log(logger, logger.level("INFO").no),
        )
        async def summarize_chunk(chunk: str, idx: int, total: int):
            logger.info(f"[summarize-progress] chunk {idx}/{total} started")
            t0 = time.perf_counter()
            loop = asyncio.get_event_loop()

            def worker():
                fn = self._make_extraction_function(schema, prompt_text=prompt_text)
                return self._call_validated_function(
                    fn,
                    chunk,
                    stage="map",
                    prompt_text=prompt_text,
                )

            ctx = contextvars.copy_context()
            result = await loop.run_in_executor(None, lambda: ctx.run(worker))
            elapsed = time.perf_counter() - t0
            logger.info(f"[summarize-progress] chunk {idx}/{total} completed seconds={elapsed:.1f}")
            return result

        tasks = [summarize_chunk(chunk, i + 1, len(chunks)) for i, chunk in enumerate(chunks)]
        results = await asyncio.gather(*tasks)
        return results, len(results)

    def _augment_with_user_prompt(self, text: str) -> str:
        # Purpose is injected in prompt instructions, not chunk payload.
        # Keeping chunk payload pure content avoids purpose-only chunks.
        return text

    def calculate_chunk_size(self, total_tokens: int) -> int:
        with self._track_step("chunk_size"):
            prompt_tokens = self.compute_required_tokens("", count_context=True)
            self._last_prompt_token_estimate = prompt_tokens
            max_tokens_per_chunk = int(self._max_context_tokens() - prompt_tokens * 0.8)
            # Ensure content payload is materially larger than instruction overhead.
            min_effective_chunk = self._semantic_min_chunk_tokens(prompt_tokens)

            target_chunks = max(1, self.min_num_chunks)
            if self._semantic_mode_enabled():
                target_chunks = self._semantic_target_map_chunks(total_tokens, prompt_tokens)
            self._usage["chunking"]["target_min_chunks"] = target_chunks

            # Keep hard ceiling coherent even when prompt overhead is high.
            hard_cap = int(min(self.max_chunk_size, max_tokens_per_chunk))
            if hard_cap <= 0:
                hard_cap = max(64, int(self.max_chunk_size))

            effective_floor = int(min(min_effective_chunk, hard_cap))
            if effective_floor < min_effective_chunk:
                logger.debug(
                    f"Clamped semantic min chunk from {min_effective_chunk} to {effective_floor} "
                    f"(max_chunk_size={self.max_chunk_size}, context_cap={max_tokens_per_chunk})."
                )

            chunk_size = total_tokens // target_chunks - prompt_tokens
            chunk_size = max(effective_floor, chunk_size)

            if chunk_size > hard_cap:
                denominator = max(1, hard_cap + prompt_tokens)
                required_chunks = max(target_chunks, int(math.ceil(total_tokens / denominator)))
                chunk_size = total_tokens // required_chunks - prompt_tokens
                chunk_size = max(effective_floor, chunk_size)

            return int(max(64, min(hard_cap, chunk_size)))

    def get_document_type(self, content: str) -> DocumentType:
        allowed_types = [doc_type.value for doc_type in DocumentType]

        class ContentType(LLMDataModel):
            type: str

            @field_validator("type")
            def validate_type(cls, v):
                assert v in allowed_types, f"Type must be one of: {', '.join(sorted(allowed_types))}"
                return v

        doc_type_func = ValidatedFunction(
            data_model=ContentType,
            retry_count=self.retry_count,
            prompt=(
                "What type of content is this text?\n"
                + f"Allowed types: {', '.join(sorted(allowed_types))}\n"
                + "The type must map exactly to one of the listed values."
            ),
            static_context=r"Return JSON: {'type': string}",
        )

        with self._track_step("detect_document_type"):
            if self.engine is not None:
                with DynamicEngine(model=self.engine.model, api_key=self.engine.api_key):
                    res = doc_type_func(
                        content,
                        preview=False,
                        response_format={"type": "json_object"},
                        seed=self.seed,
                    )
            else:
                res = doc_type_func(
                    content,
                    preview=False,
                    response_format={"type": "json_object"},
                    seed=self.seed,
                )
        self._record_llm_call(
            "type_detection",
            estimated_input_tokens=self._estimate_tokens_approx(content),
            estimated_output_tokens=self._estimate_tokens_approx(res.model_dump(mode="json")),
        )

        self.document_type = DocumentType(res.type)
        return self.document_type

    def get_document_language(self, content: str) -> str:
        class ContentLanguage(LLMDataModel):
            language: str

        if self.document_lang is not None:
            return self.document_lang

        doc_lang_func = ValidatedFunction(
            data_model=ContentLanguage,
            retry_count=self.retry_count,
            prompt=dedent(
                """Which language is this document in?
            - Follow ISO 639 naming.
            - Use this exact format: '[[language_name]] ([[country]]) [[language_code]]'"""
            ),
            static_context=r"Return JSON: {'language': string}",
        )

        with self._track_step("detect_document_language"):
            if self.engine is not None:
                with DynamicEngine(model=self.engine.model, api_key=self.engine.api_key):
                    res = doc_lang_func(
                        content,
                        preview=False,
                        response_format={"type": "json_object"},
                        seed=self.seed,
                    )
            else:
                res = doc_lang_func(
                    content,
                    preview=False,
                    response_format={"type": "json_object"},
                    seed=self.seed,
                )
        self._record_llm_call(
            "language_detection",
            estimated_input_tokens=self._estimate_tokens_approx(content),
            estimated_output_tokens=self._estimate_tokens_approx(res.model_dump(mode="json")),
        )

        return res.language

    def forward(self, **kwargs) -> Summary:
        self.clear()
        self._usage = self._init_usage_tracking()
        self._usage["chunking"]["requested_strategy"] = self._requested_chunking_strategy()
        self._refresh_doc_profile_usage()
        self._token_count_cache.clear()
        self._total_tokens_processed = 0
        self._token_offset = 0
        self._last_prompt_token_estimate = 0
        try:
            if self.engine is not None:
                with DynamicEngine(model=self.engine.model, api_key=self.engine.api_key):
                    return self._forward_with_engine(**kwargs)
            return self._forward_with_engine(**kwargs)
        finally:
            self._log_usage_summary()

    def _forward_with_engine(self, **kwargs) -> Summary:
        with self._track_step("total_pipeline"):
            with self._track_step("input_token_count"):
                total_tokens = self.compute_required_tokens_graceful(self.content, count_context=False)
                if total_tokens is None:
                    logger.warning("Total tokens could not be determined.")
                    total_tokens = 1

            # Detect type before chunking so profile-specific chunking can be applied.
            if self.document_type is None:
                doc_type_probe = self._pre_chunk_type_detection_input()
                doc_type = self.get_document_type(doc_type_probe)
            else:
                doc_type = self.document_type
            self._refresh_doc_profile_usage()
            logger.debug(
                f"Using document profile '{self._active_doc_profile().name}' for detected type '{doc_type.value}'."
            )

            chunk_size = self.calculate_chunk_size(total_tokens)
            data = self._augment_with_user_prompt(self.content)

            chunks = self.chunk_by_token_count(str(data), chunk_size)
            if not chunks:
                chunks = [str(data)]
            chunks = self._filter_non_substantive_chunks(chunks)
            if not chunks:
                chunks = [str(data)]

            detection_chunk = self._select_detection_chunk(chunks)
            doc_lang = self.get_document_language(detection_chunk)
            self.adapt("[[DOCUMENT TYPE]]\n" + doc_type.value)
            self.adapt("[[DOCUMENT LANGUAGE]]\n" + doc_lang)
            document_level_payload = self._pre_chunk_type_detection_input()
            document_level_fields: Dict[str, Any] = {}

            if self._document_data_model() is not None:
                logger.debug("Extracting document-level fields...")
                with self._track_step("document_level_stage"):
                    document_level_fields = self._extract_document_level_fields(
                        document_level_payload
                    )

            nest_asyncio.apply()
            loop = always_get_an_event_loop()

            logger.debug(f"Processing {len(chunks)} chunks (map stage)...")
            with self._track_step("map_stage"):
                chunk_results, _orig_chunk_count = loop.run_until_complete(self.summarize_chunks(chunks, **kwargs))

            logger.debug("Merging chunk outputs by field (reduce stage)...")
            with self._track_step("reduce_stage"):
                merged = self._merge_fields(chunk_results, language=doc_lang)
                if document_level_fields:
                    merged = self._merge_overlay(merged, document_level_fields)
                res = self.data_model(**merged)

            token_count = self.compute_required_tokens_graceful(res, count_context=False) or 0
            logger.debug(f"Merged token count before budget enforcement: {token_count}")

            if self._has_output_limit() and self.enable_initial_compression and token_count > self.max_output_tokens:
                with self._track_step("budget_enforcement"):
                    self._enforce_budget(res)
                    token_count = self.compute_required_tokens_graceful(res, count_context=False) or token_count

            if self._has_output_limit() and token_count > self.max_output_tokens:
                logger.warning(
                    f"Output still exceeds max_output_tokens ({token_count}>{self.max_output_tokens}); applying deterministic trim"
                )
                with self._track_step("deterministic_trim"):
                    self._deterministic_trim(res)

            if hasattr(res, "type"):
                res.type = doc_type

            self._sanitize_contradictory_summary_intro(res)

            final_tokens = self.compute_required_tokens_graceful(res, count_context=False)
            if final_tokens is not None:
                logger.debug(f"Compression ratio: {total_tokens} -> {final_tokens} ({final_tokens/max(1,total_tokens):.2f})")

            return res

    def compute_required_tokens_graceful(self, data: Any, count_context: bool = True):
        try:
            return self.compute_required_tokens(data, count_context=count_context)
        except NotImplementedError:
            logger.debug("compute_required_tokens is not implemented for this engine, returning None")
            return

    # ------------------------
    # Field-aware reduce stage
    # ------------------------
    def _unwrap_annotation(self, annotation: Any) -> Any:
        if annotation is None:
            return Any

        origin = get_origin(annotation)
        args = get_args(annotation)

        # Annotated[T, ...]
        if origin is not None and str(origin).endswith("Annotated") and args:
            return self._unwrap_annotation(args[0])

        # Optional[T] / Union[T, None]
        if origin in (Union, getattr(__import__("types"), "UnionType", object)):
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                return self._unwrap_annotation(non_none[0])

        return annotation

    def _annotation_is_str(self, annotation: Any) -> bool:
        ann = self._unwrap_annotation(annotation)
        return ann is str

    def _annotation_is_list(self, annotation: Any) -> bool:
        ann = self._unwrap_annotation(annotation)
        origin = get_origin(ann)
        return origin in (list, List)

    def _annotation_is_list_of_str(self, annotation: Any) -> bool:
        ann = self._unwrap_annotation(annotation)
        origin = get_origin(ann)
        if origin not in (list, List):
            return False
        args = get_args(ann)
        if not args:
            return False
        return self._unwrap_annotation(args[0]) is str

    def _list_item_annotation(self, annotation: Any) -> Any:
        ann = self._unwrap_annotation(annotation)
        origin = get_origin(ann)
        if origin not in (list, List):
            return Any
        args = get_args(ann)
        if not args:
            return Any
        return self._unwrap_annotation(args[0])

    def _annotation_is_structured_model(self, annotation: Any) -> bool:
        ann = self._unwrap_annotation(annotation)
        return inspect.isclass(ann) and issubclass(ann, LLMDataModel)

    def _annotation_is_list_of_structured_models(self, annotation: Any) -> bool:
        if not self._annotation_is_list(annotation):
            return False
        return self._annotation_is_structured_model(self._list_item_annotation(annotation))

    def _default_value_for_field(self, field_info) -> Any:
        if field_info.default is not PydanticUndefined:
            return field_info.default
        if field_info.default_factory is not None:
            return field_info.default_factory()

        ann = self._unwrap_annotation(field_info.annotation)
        origin = get_origin(ann)

        if ann is str:
            return ""
        if origin in (list, List):
            return []
        if origin in (dict, Dict):
            return {}

        # If it is Optional[...], return None.
        raw_origin = get_origin(field_info.annotation)
        raw_args = get_args(field_info.annotation)
        if raw_origin in (Union, getattr(__import__("types"), "UnionType", object)) and any(
            a is type(None) for a in raw_args
        ):
            return None

        return None

    def _field_value_token_count(self, value: Any) -> int:
        return self._fast_path_token_count(_safe_jsonable(value))

    def _has_output_limit(self) -> bool:
        return self.max_output_tokens is not None and self.max_output_tokens > 0

    def _is_summary_like_field(self, field_name: str) -> bool:
        name = field_name.lower()
        return any(k in name for k in ("summary", "overview", "narrative"))

    def _has_extracted_evidence(self, result: LLMDataModel) -> bool:
        for field_name, field_info in type(result).model_fields.items():
            if getattr(field_info, "exclude", False) or field_name == "summary":
                continue
            value = getattr(result, field_name, None)
            if isinstance(value, list):
                if any(isinstance(v, str) and v.strip() for v in value):
                    return True
                if any(v not in (None, "", [], {}) for v in value):
                    return True
            elif isinstance(value, str):
                if len(value.strip()) >= 80:
                    return True
            elif value not in (None, "", [], {}):
                return True
        return False

    def _sanitize_contradictory_summary_intro(self, result: LLMDataModel) -> None:
        summary = getattr(result, "summary", None)
        if not isinstance(summary, str) or not summary.strip():
            return
        if not self._has_extracted_evidence(result):
            return

        text = summary.strip()
        patterns = [
            r"^\s*No(?:\s+accessible)?\s+content\s+was\s+provided[\s\S]{0,700}?(?:can(?:not|'t)\s+be\s+(?:extracted|summarized|performed)\.)\s*",
            r"^\s*No(?:\s+accessible)?\s+content\s+was\s+provided[^.]*\.\s*(?:As a result,[^.]*\.\s*)?",
        ]
        for pat in patterns:
            cleaned = re.sub(pat, "", text, count=1, flags=re.IGNORECASE).strip()
            if cleaned != text and cleaned:
                setattr(result, "summary", cleaned)
                logger.debug(
                    "Removed contradictory 'no content provided' preamble from summary."
                )
                return

    def _detail_list_min_items(self, field_name: str) -> int:
        # Prefer min_items from Pydantic Field json_schema_extra metadata
        field_info = self.data_model.model_fields.get(field_name)
        if field_info is not None:
            extra = getattr(field_info, "json_schema_extra", None) or {}
            if isinstance(extra, dict) and "min_items" in extra:
                return int(extra["min_items"])

        # Fallback to hardcoded defaults for schemas without metadata
        name = field_name.lower()
        if "fact" in name:
            return 8
        if "quote" in name:
            return 3
        if "insight" in name:
            return 4
        if "event" in name:
            return 4
        if "story" in name or "anecdote" in name:
            return 3
        if "case" in name:
            return 3
        if "vocab" in name:
            return 16
        if "exercise" in name:
            return 3
        return 2

    def _compute_field_budgets(self, merged_candidates: Dict[str, List[Any]]) -> Dict[str, int]:
        field_infos = self.data_model.model_fields

        weights: Dict[str, float] = {}
        list_like_fields: List[str] = []
        summary_like_fields: List[str] = []

        # Unlimited mode: keep recall-first budgets derived from available content.
        if not self._has_output_limit():
            budgets: Dict[str, int] = {}
            for field_name, field_info in field_infos.items():
                if getattr(field_info, "exclude", False):
                    continue
                field_candidates = merged_candidates.get(field_name, [])
                if not field_candidates:
                    continue
                candidate_tokens = self._estimate_tokens_approx(field_candidates)
                ann = self._unwrap_annotation(field_info.annotation)
                multiplier = 1.35
                if ann is str and self._is_summary_like_field(field_name):
                    multiplier = 1.15
                elif self._annotation_is_list(field_info.annotation):
                    multiplier = 1.5
                budgets[field_name] = max(256, int(candidate_tokens * multiplier))
            return budgets

        for field_name, field_info in field_infos.items():
            if getattr(field_info, "exclude", False):
                continue

            ann = self._unwrap_annotation(field_info.annotation)
            weight = 1.0

            # Check for explicit budget_weight in Pydantic Field json_schema_extra
            extra = getattr(field_info, "json_schema_extra", None) or {}
            explicit_weight = extra.get("budget_weight") if isinstance(extra, dict) else None

            if ann is str:
                weight = 1.1
                if self._is_summary_like_field(field_name):
                    # Keep summary concise so list fields can retain more details.
                    weight = 0.95
                    summary_like_fields.append(field_name)
            elif self._annotation_is_list_of_str(field_info.annotation) or self._annotation_is_list(field_info.annotation):
                if self._annotation_is_list_of_str(field_info.annotation):
                    weight = 2.4
                    list_like_fields.append(field_name)
                else:
                    weight = 1.4

                if explicit_weight is not None:
                    weight = float(explicit_weight)
                elif self._annotation_is_list_of_str(field_info.annotation):
                    # Fallback to hardcoded name-based weights for schemas without metadata
                    lname = field_name.lower()
                    if any(k in lname for k in ("fact", "insight", "event", "story", "anecdote", "case", "vocab", "exercise")):
                        weight = 2.8
                    if "quote" in field_name.lower():
                        weight = 2.1

            # Fields with more candidates usually need more space.
            candidate_count = max(1, len(merged_candidates.get(field_name, [])))
            weights[field_name] = weight * min(4.0, 1.0 + (candidate_count / 8.0))

        total_weight = max(1e-6, sum(weights.values()))
        budgets: Dict[str, int] = {}

        for field_name, weight in weights.items():
            # Keep a per-field floor to avoid dropping fields to near-empty outputs.
            budgets[field_name] = max(64, int(self.max_output_tokens * (weight / total_weight)))

        # Enforce list floors for higher recall.
        for field_name in list_like_fields:
            candidates = merged_candidates.get(field_name, [])
            if not candidates:
                continue
            min_items = self._detail_list_min_items(field_name)
            # Approximate 18 tokens/item for compact factual bullets.
            list_floor = max(88, min(260, int(min_items * 18)))
            budgets[field_name] = max(budgets.get(field_name, 64), list_floor)

        # Cap summary-like fields so they do not consume most of the output budget.
        summary_cap = max(140, int(self.max_output_tokens * 0.28))
        for field_name in summary_like_fields:
            budgets[field_name] = min(budgets.get(field_name, summary_cap), summary_cap)

        # If floors/caps push totals over budget, reduce non-list fields first.
        total_budget = sum(budgets.values())
        if total_budget > self.max_output_tokens:
            overflow = total_budget - self.max_output_tokens
            reduce_order: List[str] = []
            reduce_order.extend(summary_like_fields)
            reduce_order.extend(
                f
                for f in budgets.keys()
                if f not in reduce_order and f not in list_like_fields
            )
            reduce_order.extend(f for f in list_like_fields if f not in reduce_order)

            for field_name in reduce_order:
                if overflow <= 0:
                    break
                floor = 64
                if field_name in list_like_fields:
                    floor = min(96, budgets[field_name])
                room = max(0, budgets[field_name] - floor)
                if room <= 0:
                    continue
                delta = min(room, overflow)
                budgets[field_name] -= delta
                overflow -= delta

        return budgets

    def _normalize_text_key(self, text: str) -> str:
        return normalize_text_key(text)

    def _deduplicate_string_items(self, items: List[str], *, keep_quotes_verbatim: bool) -> List[str]:
        return deduplicate_string_items(
            items,
            keep_quotes_verbatim=keep_quotes_verbatim,
        )

    def _dedupe_unbounded_summary_text(self, text: str) -> str:
        return dedupe_unbounded_summary_text(text)

    def _call_validated_function(
        self,
        fn: ValidatedFunction,
        payload: str,
        *,
        max_completion_tokens: Optional[int] = None,
        stage: str = "merge",
        prompt_text: str = "",
    ):
        kwargs = {
            "preview": False,
            "response_format": {"type": "json_object"},
            "seed": self.seed,
        }
        if max_completion_tokens is not None:
            kwargs["max_completion_tokens"] = max_completion_tokens

        with self._track_step(f"validated_function_{stage}"):
            if self.engine is not None:
                with DynamicEngine(model=self.engine.model, api_key=self.engine.api_key):
                    res = fn(payload, **kwargs)
            else:
                res = fn(payload, **kwargs)
        self._record_llm_call(
            stage,
            estimated_input_tokens=self._estimate_tokens_approx(payload) + self._estimate_tokens_approx(prompt_text),
            estimated_output_tokens=self._estimate_tokens_approx(res.model_dump(mode="json")),
        )
        return res

    def _build_single_field_model(self, field_name: str, annotation: Any, description: str):
        model_name = f"FieldMerge_{_sanitize_model_name(field_name)}"
        return create_model(
            model_name,
            __base__=LLMDataModel,
            value=(annotation, Field(description=description or f"Merged value for field '{field_name}'")),
        )

    def _merge_prompt_with_schema(self, base_prompt: str, model: type[LLMDataModel]) -> str:
        return dedent(
            f"""
            {base_prompt.strip()}

            Exact target schema:
            {model.instruct_llm()}

            Additional schema rules:
            - Use exactly the field names shown in the schema.
            - Do not rename keys or invent synonyms.
            - Preserve nested object structure exactly as defined.
            - If a nested value is unavailable, use `null` only where the schema allows it.
            - For list outputs, every item must conform to the item schema exactly.
            """
        ).strip()

    def _structured_item_priority(self, item: Any) -> Tuple[float, int, int]:
        payload = _safe_jsonable(item)
        if isinstance(payload, dict):
            confidence = payload.get("confidence")
            try:
                confidence_score = float(confidence) if confidence is not None else -1.0
            except (TypeError, ValueError):
                confidence_score = -1.0

            populated_fields = 0
            for value in payload.values():
                if isinstance(value, dict):
                    populated_fields += sum(
                        1 for nested in value.values() if nested not in (None, "", [], {})
                    )
                elif value not in (None, "", [], {}):
                    populated_fields += 1

            primary_text = ""
            for key in ("fact", "quote", "text", "summary", "title", "name"):
                candidate = payload.get(key)
                if isinstance(candidate, str) and candidate.strip():
                    primary_text = candidate.strip()
                    break
            return confidence_score, populated_fields, min(len(primary_text), 240)

        text = str(payload).strip()
        return -1.0, 0, min(len(text), 240)

    def _merge_structured_list_field(
        self,
        *,
        field_name: str,
        items: List[Any],
        flattened: List[Any],
        target_tokens: int,
    ) -> List[Any]:
        if not items:
            return []

        if not self._has_output_limit():
            return items

        frequencies = Counter(
            json.dumps(_safe_jsonable(item), sort_keys=True, ensure_ascii=False)
            for item in flattened
        )
        unique_keys = [
            json.dumps(_safe_jsonable(item), sort_keys=True, ensure_ascii=False)
            for item in items
        ]
        order = {key: idx for idx, key in enumerate(unique_keys)}

        ranked_keys = sorted(
            unique_keys,
            key=lambda key: (
                -frequencies[key],
                -self._structured_item_priority(items[order[key]])[0],
                -self._structured_item_priority(items[order[key]])[1],
                -self._structured_item_priority(items[order[key]])[2],
                order[key],
            ),
        )

        selected_keys = set()
        running = 0
        keep_min = min(self._detail_list_min_items(field_name), len(items))

        for key in ranked_keys:
            item = items[order[key]]
            item_tokens = self._estimate_tokens_approx(_safe_jsonable(item))
            if len(selected_keys) >= keep_min and selected_keys and running + item_tokens > target_tokens:
                continue
            selected_keys.add(key)
            running += item_tokens

        if not selected_keys:
            selected_keys.add(ranked_keys[0])

        return [item for item, key in zip(items, unique_keys) if key in selected_keys]

    def _llm_merge_string_batch(
        self,
        *,
        field_name: str,
        field_description: str,
        candidates: List[str],
        target_tokens: int,
        language: Optional[str],
    ) -> str:
        if not candidates:
            return ""

        model = self._build_single_field_model(
            field_name=field_name,
            annotation=str,
            description=field_description or f"Merged content for {field_name}",
        )

        purpose = self.user_prompt or ""
        prompt = dedent(
            f"""
            Merge candidate values for one schema field.

            Field name: {field_name}
            Field description: {field_description or 'n/a'}
            Target language: {language or 'same as document'}
            Purpose: {purpose}

            Requirements:
            - Preserve concrete details from all candidates.
            - Remove repetitions and contradictions.
            - Keep style and content aligned to the field description.
            - Stay concise enough to fit approximately {max(64, target_tokens)} tokens.
            """
        )

        fn = ValidatedFunction(
            data_model=model,
            retry_count=self.retry_count,
            prompt=prompt,
            static_context="Return JSON with field `value`. JSON only.",
        )

        payload = json.dumps(
            {
                "field": field_name,
                "description": field_description,
                "purpose": purpose,
                "candidates": candidates,
            },
            ensure_ascii=False,
        )

        try:
            max_completion_tokens = max(128, min(2048, int(target_tokens * 1.4)))
            res = self._call_validated_function(
                fn,
                payload,
                max_completion_tokens=max_completion_tokens,
                stage="merge",
                prompt_text=prompt,
            )
            value = str(getattr(res, "value", "")).strip()
            return value if value else "\n\n".join(candidates)
        except Exception as e:
            logger.warning(f"String merge failed for field '{field_name}': {e}")
            return "\n\n".join(candidates)

    def _llm_merge_generic_field(
        self,
        *,
        field_name: str,
        field_description: str,
        annotation: Any,
        candidates: List[Any],
        language: Optional[str],
        target_tokens: int,
    ) -> Any:
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]

        model = self._build_single_field_model(
            field_name=field_name,
            annotation=annotation,
            description=field_description or f"Merged value for {field_name}",
        )

        prompt = self._merge_prompt_with_schema(
            dedent(
            f"""
            Merge multiple candidate values for one schema field.

            Field name: {field_name}
            Field description: {field_description or 'n/a'}
            Target language: {language or 'same as document'}

            Requirements:
            - Preserve important details.
            - Resolve duplicates and inconsistencies.
            - Output exactly one merged field value.
            - Respect the target type and schema constraints.
            - Keep the value concise enough for ~{max(64, target_tokens)} tokens.
            """
            ),
            model,
        )

        fn = ValidatedFunction(
            data_model=model,
            retry_count=self.retry_count,
            prompt=prompt,
            static_context="Return JSON with field `value`. JSON only.",
        )

        payload = json.dumps(
            {
                "field": field_name,
                "description": field_description,
                "purpose": self.user_prompt,
                "candidates": [_safe_jsonable(c) for c in candidates],
            },
            ensure_ascii=False,
        )

        try:
            max_completion_tokens = max(128, min(2048, int(target_tokens * 1.4)))
            res = self._call_validated_function(
                fn,
                payload,
                max_completion_tokens=max_completion_tokens,
                stage="merge",
                prompt_text=prompt,
            )
            return getattr(res, "value", candidates[0])
        except Exception as e:
            logger.warning(f"Generic field merge failed for '{field_name}': {e}")
            return candidates[0]

    def _merge_string_field(
        self,
        *,
        field_name: str,
        field_description: str,
        candidates: List[Any],
        target_tokens: int,
        language: Optional[str],
    ) -> str:
        clean = [str(c).strip() for c in candidates if isinstance(c, str) and str(c).strip()]
        if not clean:
            return ""

        # Deduplicate exact normalized strings while preserving order.
        seen = set()
        ordered = []
        for item in clean:
            key = self._normalize_text_key(item)
            if key not in seen:
                seen.add(key)
                ordered.append(item)

        if len(ordered) == 1:
            return ordered[0]

        if not self._has_output_limit():
            merged_unbounded = "\n\n".join(ordered)
            if self._is_summary_like_field(field_name):
                return self._dedupe_unbounded_summary_text(merged_unbounded)
            return merged_unbounded

        effective_target = target_tokens
        if self._is_summary_like_field(field_name):
            # Push summaries to stay tighter so list fields can carry richer detail.
            effective_target = min(target_tokens, max(160, int(self.max_output_tokens * 0.26)))

        total_tokens = self._estimate_tokens_near_threshold(
            "\n\n".join(ordered),
            threshold=max(64, int(effective_target * 1.1)),
            margin_ratio=0.22,
            min_margin=96,
            exact_mode="fast",
        )
        if total_tokens <= int(effective_target * 1.1):
            return "\n\n".join(ordered)

        work = ordered
        batch_size = 6
        while len(work) > 1:
            merged_batches: List[str] = []
            for i in range(0, len(work), batch_size):
                batch = work[i : i + batch_size]
                merged = self._llm_merge_string_batch(
                    field_name=field_name,
                    field_description=field_description,
                    candidates=batch,
                    target_tokens=max(64, int(effective_target * 0.9)),
                    language=language,
                )
                merged_batches.append(merged)
            work = merged_batches
            batch_size = min(12, batch_size + 2)

        merged = work[0]
        merged_tokens = self._estimate_tokens_near_threshold(
            merged,
            threshold=effective_target,
            margin_ratio=0.20,
            min_margin=96,
            exact_mode="fast",
        )
        if merged_tokens > effective_target:
            merged = self._llm_merge_string_batch(
                field_name=field_name,
                field_description=field_description,
                candidates=[merged],
                target_tokens=effective_target,
                language=language,
            )
        return merged

    def _merge_list_of_str_field(
        self,
        *,
        field_name: str,
        candidates: List[Any],
        target_tokens: int,
    ) -> List[str]:
        flattened: List[str] = []
        for c in candidates:
            if isinstance(c, list):
                flattened.extend([str(v) for v in c if isinstance(v, str)])
            elif isinstance(c, str):
                flattened.append(c)

        keep_quotes_verbatim = "quote" in field_name.lower()
        deduped = self._deduplicate_string_items(flattened, keep_quotes_verbatim=keep_quotes_verbatim)
        if not deduped:
            return []

        if not self._has_output_limit():
            return deduped

        # Ranking mixes repeated signals with richer entries.
        freq = Counter(self._normalize_text_key(x) for x in flattened if isinstance(x, str) and x.strip())
        order = {self._normalize_text_key(x): idx for idx, x in enumerate(deduped)}

        ranked = sorted(
            deduped,
            key=lambda x: (
                -freq[self._normalize_text_key(x)],
                -min(len(x), 220),
                order[self._normalize_text_key(x)],
            ),
        )

        kept: List[str] = []
        running = 0
        min_items = min(self._detail_list_min_items(field_name), len(ranked))
        for item in ranked:
            t = self._estimate_tokens_approx(item)
            if len(kept) >= min_items and kept and running + t > target_tokens:
                continue
            kept.append(item)
            running += t

        # If all entries are larger than budget, keep at least one.
        if not kept:
            kept = [ranked[0]]

        return kept

    def _merge_generic_list_field(
        self,
        *,
        field_name: str,
        field_description: str,
        annotation: Any,
        candidates: List[Any],
        language: Optional[str],
        target_tokens: int,
    ) -> List[Any]:
        flattened: List[Any] = []
        for c in candidates:
            if isinstance(c, list):
                flattened.extend(c)
            else:
                flattened.append(c)

        # Stable unique by JSON form.
        unique: List[Any] = []
        seen = set()
        for item in flattened:
            key = json.dumps(_safe_jsonable(item), sort_keys=True, ensure_ascii=False)
            if key not in seen:
                seen.add(key)
                unique.append(item)

        if not unique:
            return []

        if not self._has_output_limit():
            return unique

        joined_tokens = self._estimate_tokens_near_threshold(
            unique,
            threshold=max(64, int(target_tokens * 1.1)),
            margin_ratio=0.20,
            min_margin=96,
            exact_mode="fast",
        )
        if joined_tokens <= int(target_tokens * 1.1):
            return unique

        if self._annotation_is_list_of_structured_models(annotation):
            return self._merge_structured_list_field(
                field_name=field_name,
                items=unique,
                flattened=flattened,
                target_tokens=target_tokens,
            )

        merged = self._llm_merge_generic_field(
            field_name=field_name,
            field_description=field_description,
            annotation=annotation,
            candidates=unique,
            language=language,
            target_tokens=target_tokens,
        )

        if isinstance(merged, list):
            return merged

        return unique

    def _merge_fields(self, chunk_results: List[LLMDataModel], language: Optional[str]) -> Dict[str, Any]:
        field_infos = self.data_model.model_fields
        candidates: Dict[str, List[Any]] = defaultdict(list)

        for result in chunk_results:
            for field_name, field_info in type(result).model_fields.items():
                if getattr(field_info, "exclude", False):
                    continue
                value = getattr(result, field_name, None)
                if value is None:
                    continue
                candidates[field_name].append(value)

        budgets = self._compute_field_budgets(candidates)
        merged: Dict[str, Any] = {}

        for field_name, field_info in field_infos.items():
            if getattr(field_info, "exclude", False):
                continue

            field_candidates = candidates.get(field_name, [])
            if not field_candidates:
                merged[field_name] = self._default_value_for_field(field_info)
                continue

            ann = self._unwrap_annotation(field_info.annotation)
            field_description = field_info.description or f"Field '{field_name}'"
            if self._has_output_limit():
                fallback_target = max(64, self.max_output_tokens // max(1, len(field_infos)))
            else:
                fallback_target = max(256, self._field_value_token_count(field_candidates))
            target_tokens = budgets.get(field_name, fallback_target)

            if ann is str:
                merged[field_name] = self._merge_string_field(
                    field_name=field_name,
                    field_description=field_description,
                    candidates=field_candidates,
                    target_tokens=target_tokens,
                    language=language,
                )
            elif self._annotation_is_list_of_str(field_info.annotation):
                merged[field_name] = self._merge_list_of_str_field(
                    field_name=field_name,
                    candidates=field_candidates,
                    target_tokens=target_tokens,
                )
            elif self._annotation_is_list(field_info.annotation):
                merged[field_name] = self._merge_generic_list_field(
                    field_name=field_name,
                    field_description=field_description,
                    annotation=field_info.annotation,
                    candidates=field_candidates,
                    language=language,
                    target_tokens=target_tokens,
                )
            else:
                merged[field_name] = self._llm_merge_generic_field(
                    field_name=field_name,
                    field_description=field_description,
                    annotation=field_info.annotation,
                    candidates=field_candidates,
                    language=language,
                    target_tokens=target_tokens,
                )

        return merged

    # ------------------------
    # Budget control helpers
    # ------------------------
    def _enforce_budget(self, result: LLMDataModel) -> None:
        if not self._has_output_limit():
            return
        max_rounds = 12
        for _ in range(max_rounds):
            current = self.compute_required_tokens_graceful(result, count_context=False)
            if current is None or current <= self.max_output_tokens:
                return

            # 1) Trim summary-like strings first to preserve extracted lists.
            trimmed = False
            for field_name, field_info in type(result).model_fields.items():
                if getattr(field_info, "exclude", False) or not self._is_summary_like_field(field_name):
                    continue
                value = getattr(result, field_name, None)
                if isinstance(value, str) and len(value) > 220:
                    shortened = value[: int(len(value) * 0.82)].rstrip()
                    if len(shortened) < len(value):
                        shortened += "..."
                    setattr(result, field_name, shortened)
                    trimmed = True
                    break
            if trimmed:
                continue

            # 2) Trim other long strings.
            for field_name, field_info in type(result).model_fields.items():
                if getattr(field_info, "exclude", False):
                    continue
                value = getattr(result, field_name, None)
                if isinstance(value, str) and len(value) > 320:
                    shortened = value[: int(len(value) * 0.88)].rstrip()
                    if len(shortened) < len(value):
                        shortened += "..."
                    setattr(result, field_name, shortened)
                    trimmed = True
                    break
            if trimmed:
                continue

            # 3) Trim list fields conservatively.
            largest_list_field = None
            largest_list_tokens = 0
            for field_name, field_info in type(result).model_fields.items():
                if getattr(field_info, "exclude", False):
                    continue
                value = getattr(result, field_name, None)
                if not isinstance(value, list) or len(value) <= 1:
                    continue
                tokens = self._field_value_token_count(value)
                if tokens > largest_list_tokens:
                    largest_list_tokens = tokens
                    largest_list_field = field_name

            if largest_list_field is not None:
                value = getattr(result, largest_list_field)
                keep_min = min(self._detail_list_min_items(largest_list_field), len(value))
                keep_ratio = 0.92
                if current > int(self.max_output_tokens * 1.25):
                    keep_ratio = 0.82
                keep = max(keep_min, int(len(value) * keep_ratio))
                if keep < len(value):
                    setattr(result, largest_list_field, value[:keep])
                    continue

            return

    def _deterministic_trim(self, result: LLMDataModel) -> None:
        if not self._has_output_limit():
            return
        # Final bounded fallback when token estimator still reports over-limit.
        for _ in range(20):
            current = self.compute_required_tokens_graceful(result, count_context=False)
            if current is None or current <= self.max_output_tokens:
                return

            changed = False
            # Prioritize summary-like strings.
            for field_name, field_info in type(result).model_fields.items():
                if getattr(field_info, "exclude", False) or not self._is_summary_like_field(field_name):
                    continue
                value = getattr(result, field_name, None)
                if isinstance(value, str) and len(value) > 220:
                    setattr(result, field_name, value[: int(len(value) * 0.88)].rstrip() + "...")
                    changed = True
                    break
            if changed:
                continue

            for field_name, field_info in type(result).model_fields.items():
                if getattr(field_info, "exclude", False):
                    continue
                value = getattr(result, field_name, None)

                if isinstance(value, str) and len(value) > 256:
                    setattr(result, field_name, value[: int(len(value) * 0.9)].rstrip() + "...")
                    changed = True
                    break

                if isinstance(value, list) and len(value) > 1:
                    keep_min = min(self._detail_list_min_items(field_name), len(value))
                    if len(value) > keep_min:
                        setattr(result, field_name, value[:-1])
                        changed = True
                        break

            if not changed:
                break

        # Hard fallback: strictly enforce token budget, even if it means reducing
        # list fields below their preferred minimum.
        for _ in range(60):
            current = self.compute_required_tokens_graceful(result, count_context=False)
            if current is None or current <= self.max_output_tokens:
                return

            # Prefer trimming summary-like strings first.
            changed = False
            for field_name, field_info in type(result).model_fields.items():
                if getattr(field_info, "exclude", False) or not self._is_summary_like_field(field_name):
                    continue
                value = getattr(result, field_name, None)
                if isinstance(value, str) and len(value) > 100:
                    setattr(result, field_name, value[: int(len(value) * 0.9)].rstrip() + "...")
                    changed = True
                    break
            if changed:
                continue

            # Then trim the largest list by one element.
            largest_list_field = None
            largest_list_tokens = 0
            for field_name, field_info in type(result).model_fields.items():
                if getattr(field_info, "exclude", False):
                    continue
                value = getattr(result, field_name, None)
                if isinstance(value, list) and len(value) > 1:
                    t = self._field_value_token_count(value)
                    if t > largest_list_tokens:
                        largest_list_tokens = t
                        largest_list_field = field_name
            if largest_list_field is not None:
                value = getattr(result, largest_list_field)
                setattr(result, largest_list_field, value[:-1])
                continue

            # Last resort: trim any long string.
            for field_name, field_info in type(result).model_fields.items():
                if getattr(field_info, "exclude", False):
                    continue
                value = getattr(result, field_name, None)
                if isinstance(value, str) and len(value) > 80:
                    setattr(result, field_name, value[: int(len(value) * 0.9)].rstrip() + "...")
                    changed = True
                    break
            if not changed:
                return


HierarchicalSummary = HierarchicalSummaryV2
