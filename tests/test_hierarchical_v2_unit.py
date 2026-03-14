import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from symai.components import ChonkieChunker
from symai.models import LLMDataModel

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import DocumentType, HierarchicalSummary
from src.hierarchical_v2 import HierarchicalSummaryV2, get_current_tokenizer
from src.hierarchical_v2_reduce import deduplicate_string_items, dedupe_unbounded_summary_text


class UnitSchema(LLMDataModel):
    summary: str = Field(description="Summary text")
    facts: List[str] = Field(default_factory=list, description="Facts")
    type: Optional[str] = None


class NestedProvenance(LLMDataModel):
    location_hint: Optional[str] = Field(default=None, description="Location hint")


class NestedAttribution(LLMDataModel):
    work_or_publication: Optional[str] = Field(default=None, description="Source title")


class NestedFact(LLMDataModel):
    fact: str = Field(description="Fact text")
    provenance: Optional[NestedProvenance] = Field(default=None, description="Fact provenance")
    attribution: Optional[NestedAttribution] = Field(default=None, description="Fact attribution")
    confidence: Optional[float] = Field(default=None, description="Fact confidence")


class NestedSchema(LLMDataModel):
    summary: str = Field(description="Summary text")
    facts: List[NestedFact] = Field(default_factory=list, description="Structured facts")


class NestedAssetMetadata(LLMDataModel):
    document_type: Optional[str] = Field(default=None, description="Detected document type")
    title: Optional[str] = Field(default=None, description="Asset title")
    authors: Optional[List[str]] = Field(default=None, description="Authors")
    speakers: Optional[List[str]] = Field(default=None, description="Speakers")
    publisher_or_collection: Optional[str] = Field(default=None, description="Publisher or collection")
    publication_year: Optional[str] = Field(default=None, description="Publication year")


class RichSchema(LLMDataModel):
    summary: str = Field(description="Summary text")
    facts: List[NestedFact] = Field(default_factory=list, description="Structured facts")
    asset_metadata: Optional[NestedAssetMetadata] = Field(
        default=None,
        description="Asset-level metadata",
    )


def _make_summarizer(*, chunker_name: str = "SemanticHybridChunker") -> HierarchicalSummaryV2:
    return HierarchicalSummaryV2(
        content="alpha paragraph\n\nsecond paragraph\n\nthird paragraph",
        document_name="unit.txt",
        data_model=UnitSchema,
        min_num_chunks=2,
        min_chunk_size=64,
        max_chunk_size=256,
        max_output_tokens=None,
        user_prompt="extract useful material",
        include_quotes=False,
        tokenizer_name="gpt2",
        chunker_name=chunker_name,
    )


def test_normalize_reader_content_avoids_list_repr() -> None:
    class DummyContent:
        def __init__(self, value):
            self.value = value

        def __str__(self) -> str:
            return repr(self.value)

    s = _make_summarizer()
    normalized = s._normalize_reader_content(DummyContent(["first", "second"]))

    assert normalized == "first\n\nsecond"
    assert not normalized.startswith("[")


def test_chunking_usage_tracks_requested_and_effective_strategy(monkeypatch) -> None:
    s = _make_summarizer(chunker_name="SemanticHybridChunker")

    monkeypatch.setattr(s, "_split_structural_segments", lambda _text: ["segment a", "segment b"])
    monkeypatch.setattr(s, "_pack_segments_semantic", lambda _segments, chunk_size: [])
    monkeypatch.setattr(s, "_pack_segments_into_chunks", lambda _segments, chunk_size: ["chunk a", "chunk b"])

    chunks = s.chunk_by_token_count("ignored", chunk_size=128)

    assert chunks == ["chunk a", "chunk b"]
    assert s.usage_report["chunking"]["requested_strategy"] == "semantic_hybrid"
    assert s.usage_report["chunking"]["effective_strategy"] == "structural"
    assert s.usage_report["chunking"]["fallback_reason"] == "semantic_empty_or_failed"


def test_sanitize_contradictory_summary_intro_when_evidence_exists() -> None:
    s = _make_summarizer()

    res = UnitSchema(
        summary=(
            'No content was provided from the referenced document "x", so no extraction can be performed. '
            "Title: Chocolate intake and cognition. Key finding: higher intake associated with better cognitive scores."
        ),
        facts=["Participants with higher chocolate intake scored better on global cognition."],
    )

    s._sanitize_contradictory_summary_intro(res)

    assert res.summary.startswith("Title: Chocolate intake and cognition.")
    assert "No content was provided" not in res.summary


def test_sanitize_contradictory_summary_intro_keeps_text_without_evidence() -> None:
    s = _make_summarizer()

    original = "No content was provided from the referenced document, so no extraction can be performed."
    res = UnitSchema(summary=original, facts=[])

    s._sanitize_contradictory_summary_intro(res)

    assert res.summary == original


def test_unbounded_summary_merge_deduplicates_repeated_paragraphs() -> None:
    s = _make_summarizer()
    p1 = "Chocolate intake is associated with better cognitive performance in cross-sectional analysis."
    p2 = "The study adjusted for demographics, cardiovascular risk factors, and dietary variables."

    merged = s._merge_string_field(
        field_name="summary",
        field_description="summary",
        candidates=[p1, p1, p2, p1],
        target_tokens=10000,
        language="English",
    )

    assert merged.count(p1) == 1
    assert p2 in merged


def test_token_count_cache_registers_hits() -> None:
    s = _make_summarizer()
    before_hits = s.usage_report["tokens"]["cache_hits"]
    before_misses = s.usage_report["tokens"]["cache_misses"]

    c1 = s._estimate_tokens_quiet("alpha beta gamma")
    c2 = s._estimate_tokens_quiet("alpha beta gamma")

    assert c1 == c2
    assert s.usage_report["tokens"]["cache_misses"] >= before_misses + 1
    assert s.usage_report["tokens"]["cache_hits"] >= before_hits + 1


def test_get_current_tokenizer_prefers_tiktoken(monkeypatch) -> None:
    tokenizer_name = "unit_fake_encoding"

    class DummyEncoding:
        def encode(self, text: str):
            return [1] * max(1, len(text.split()))

    class DummyTikToken:
        @staticmethod
        def get_encoding(name: str):
            assert name == tokenizer_name
            return DummyEncoding()

    def fail_from_pretrained(_name: str):
        raise AssertionError("Tokenizer.from_pretrained should not be called when tiktoken works")

    monkeypatch.setitem(sys.modules, "tiktoken", DummyTikToken())
    monkeypatch.setattr("src.hierarchical_v2.Tokenizer.from_pretrained", fail_from_pretrained)

    tokenizer = get_current_tokenizer(tokenizer_name)

    assert isinstance(tokenizer, DummyEncoding)


def test_threshold_counting_skips_exact_when_far_from_boundary(monkeypatch) -> None:
    s = _make_summarizer()

    monkeypatch.setattr(s, "_estimate_tokens_approx", lambda _data: 120)
    monkeypatch.setattr(s, "_fast_path_token_count", lambda _data: (_ for _ in ()).throw(AssertionError("unexpected")))
    monkeypatch.setattr(s, "_estimate_tokens_quiet", lambda _data: (_ for _ in ()).throw(AssertionError("unexpected")))

    count = s._estimate_tokens_near_threshold("alpha beta gamma", threshold=1000, exact_mode="fast")

    assert count == 120


def test_default_entry_point_points_to_v2() -> None:
    assert HierarchicalSummary is HierarchicalSummaryV2


def test_v2_uses_symbolicai_chonkie_chunker() -> None:
    s = _make_summarizer()

    assert isinstance(s.chunker, ChonkieChunker)
    assert type(s.chunker).__module__ == "symai.components"


def test_read_file_uses_markitdown_backend(monkeypatch) -> None:
    seen = {}

    class DummyReader:
        def __call__(self, file_link: str, **kwargs):
            seen["file_link"] = file_link
            seen["kwargs"] = kwargs
            return "normalized content"

    monkeypatch.setattr("src.hierarchical_v2.FileReader", lambda: DummyReader())

    s = _make_summarizer()
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        content, file_name = s.read_file(tmp.name)

    assert content == "normalized content"
    assert file_name.endswith(".pdf")
    assert seen["kwargs"] == {"backend": "markitdown"}


def test_fast_dedupe_collapses_reordered_near_duplicate_items() -> None:
    items = [
        "Chocolate intake improved memory in older adults during the study.",
        "In older adults, chocolate intake improved memory during the study.",
        "Participants with higher flavanol intake performed better on attention tasks.",
    ]

    deduped = deduplicate_string_items(items, keep_quotes_verbatim=False)

    assert len(deduped) == 2
    assert any("attention tasks" in item for item in deduped)


def test_fast_dedupe_keeps_distinct_numeric_facts() -> None:
    items = [
        "The intervention lasted 8 weeks and improved recall.",
        "The intervention lasted 12 weeks and improved recall.",
    ]

    deduped = deduplicate_string_items(items, keep_quotes_verbatim=False)

    assert deduped == items


def test_summary_dedupe_removes_near_duplicate_paragraphs_without_difflib() -> None:
    text = (
        "Chocolate intake was associated with better global cognition in older adults.\n\n"
        "In older adults, chocolate intake was associated with better global cognition.\n\n"
        "The study adjusted for demographics and cardiovascular risk."
    )

    deduped = dedupe_unbounded_summary_text(text)
    paragraphs = [p for p in deduped.split("\n\n") if p.strip()]

    assert len(paragraphs) == 2
    assert any("cardiovascular risk" in p for p in paragraphs)


def test_nested_fact_evidence_survives_budget_enforcement(monkeypatch) -> None:
    s = HierarchicalSummaryV2(
        content="alpha paragraph\n\nsecond paragraph\n\nthird paragraph",
        document_name="nested.txt",
        data_model=NestedSchema,
        min_num_chunks=2,
        min_chunk_size=64,
        max_chunk_size=256,
        max_output_tokens=40,
        user_prompt="extract useful material",
        include_quotes=False,
        tokenizer_name="gpt2",
        chunker_name="SemanticHybridChunker",
    )

    result = NestedSchema(
        summary=(
            "A very long summary that should be shortened by budget enforcement while facts stay intact. "
            * 5
        ).strip(),
        facts=[
            NestedFact(
                fact="Structured fact",
                provenance=NestedProvenance(location_hint="Section: insight; citation [1]"),
                attribution=NestedAttribution(work_or_publication="Trusted Title"),
                confidence=0.82,
            )
        ],
    )

    token_counts = iter([120, 20])
    monkeypatch.setattr(
        s,
        "compute_required_tokens_graceful",
        lambda *_args, **_kwargs: next(token_counts),
    )

    s._enforce_budget(result)

    assert result.summary.endswith("...")
    assert len(result.facts) == 1
    assert result.facts[0].fact == "Structured fact"
    assert result.facts[0].provenance is not None
    assert result.facts[0].provenance.location_hint == "Section: insight; citation [1]"
    assert result.facts[0].attribution is not None
    assert result.facts[0].attribution.work_or_publication == "Trusted Title"
    assert result.facts[0].confidence == 0.82


def test_embed_text_batch_sanitizes_blank_inputs_and_micro_batches(monkeypatch) -> None:
    s = _make_summarizer()
    calls = []

    class FakeResult:
        def __init__(self, texts):
            self.value = [[float(len(text))] for text in texts]

    class FakeSymbol:
        def __init__(self, texts):
            self._texts = list(texts)

        def embed(self):
            calls.append(list(self._texts))
            return FakeResult(self._texts)

    monkeypatch.setattr("src.hierarchical_v2.Symbol", FakeSymbol)
    monkeypatch.setattr(s, "_embedding_batch_token_limit", lambda: 10)
    monkeypatch.setattr(s, "_embedding_batch_item_limit", lambda: 8)
    monkeypatch.setattr(
        s,
        "_estimate_tokens_approx",
        lambda text: {"alpha": 5, "beta": 5, "gamma": 5}[text],
    )

    vectors = s._embed_text_batch(["alpha", "   ", "beta\x00", "\n\t", "gamma"])

    assert vectors == [[5.0], [4.0], [5.0]]
    assert calls == [["alpha", "beta"], ["gamma"]]
    assert s.usage_report["embedding"]["items"] == 3
    assert s.usage_report["embedding"]["dropped_items"] == 2
    assert s.usage_report["embedding"]["requests"] == 2


def test_embed_text_batch_recursively_splits_failed_batches(monkeypatch) -> None:
    s = _make_summarizer()
    calls = []

    class FakeResult:
        def __init__(self, texts):
            self.value = [[float(len(text))] for text in texts]

    class FakeSymbol:
        def __init__(self, texts):
            self._texts = list(texts)

        def embed(self):
            calls.append(list(self._texts))
            if len(self._texts) > 1:
                raise ValueError("batch rejected")
            return FakeResult(self._texts)

    monkeypatch.setattr("src.hierarchical_v2.Symbol", FakeSymbol)
    monkeypatch.setattr(s, "_embedding_batch_token_limit", lambda: 100)
    monkeypatch.setattr(s, "_embedding_batch_item_limit", lambda: 10)
    monkeypatch.setattr(s, "_estimate_tokens_approx", lambda _text: 1)

    vectors = s._embed_text_batch(["alpha", "beta", "gamma"])

    assert vectors == [[5.0], [4.0], [5.0]]
    assert calls == [
        ["alpha", "beta", "gamma"],
        ["alpha"],
        ["beta", "gamma"],
        ["beta"],
        ["gamma"],
    ]
    assert s.usage_report["embedding"]["requests"] == 5
    assert s.usage_report["embedding"]["failed_requests"] == 2


def test_prepare_segments_for_semantic_embedding_uses_strict_counts(monkeypatch) -> None:
    s = _make_summarizer()

    monkeypatch.setattr(s, "_embedding_item_token_limit", lambda: 1800)
    monkeypatch.setattr(
        s,
        "_estimate_tokens_quiet",
        lambda text: {
            "short": 120,
            "too long": 2200,
            "part a": 900,
            "part b": 950,
        }[text],
    )
    monkeypatch.setattr(
        s,
        "_split_text_for_embedding",
        lambda text, token_limit: ["part a", "part b"] if text == "too long" else [text],
    )

    prepared = s._prepare_segments_for_semantic_embedding(["short", "too long"])

    assert prepared == ["short", "part a", "part b"]
    assert s.usage_report["chunking"]["semantic_segments_expanded"] == 1


def test_embed_text_batch_recovers_single_oversized_item(monkeypatch) -> None:
    s = _make_summarizer()
    calls = []

    class FakeResult:
        def __init__(self, vectors):
            self.value = vectors

    class FakeSymbol:
        def __init__(self, texts):
            self._texts = list(texts)

        def embed(self):
            calls.append(list(self._texts))
            if self._texts == ["too long"]:
                raise ValueError("Invalid 'input': array length must be 2048 or less.")
            mapping = {
                "part a": [1.0, 0.0],
                "part b": [0.0, 1.0],
            }
            return FakeResult([mapping[text] for text in self._texts])

    monkeypatch.setattr("src.hierarchical_v2.Symbol", FakeSymbol)
    monkeypatch.setattr(s, "_split_text_for_embedding", lambda text, token_limit: ["part a", "part b"])
    monkeypatch.setattr(s, "_estimate_tokens_approx", lambda _text: 1)

    vectors = s._embed_text_batch(["too long"])

    assert len(vectors) == 1
    assert round(vectors[0][0], 6) == round(2**-0.5, 6)
    assert round(vectors[0][1], 6) == round(2**-0.5, 6)
    assert calls == [["too long"], ["part a", "part b"]]
    assert s.usage_report["embedding"]["requests"] == 2
    assert s.usage_report["embedding"]["failed_requests"] == 1


def test_merge_generic_list_field_structured_models_avoids_llm_retry_path(monkeypatch) -> None:
    s = HierarchicalSummaryV2(
        content="alpha paragraph\n\nsecond paragraph\n\nthird paragraph",
        document_name="nested.txt",
        data_model=NestedSchema,
        min_num_chunks=2,
        min_chunk_size=64,
        max_chunk_size=256,
        max_output_tokens=120,
        user_prompt="extract useful material",
        include_quotes=False,
        tokenizer_name="gpt2",
        chunker_name="SemanticHybridChunker",
    )

    facts = [
        NestedFact(
            fact=f"Structured fact {i}",
            provenance=NestedProvenance(location_hint=f"Section {i}"),
            attribution=NestedAttribution(work_or_publication=f"Source {i}"),
            confidence=1.0 - (i * 0.01),
        )
        for i in range(10)
    ]

    monkeypatch.setattr(
        s,
        "_llm_merge_generic_field",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("LLM merge should not run")),
    )
    monkeypatch.setattr(
        s,
        "_estimate_tokens_near_threshold",
        lambda value, **_kwargs: 500 if isinstance(value, list) else 0,
    )
    monkeypatch.setattr(
        s,
        "_estimate_tokens_approx",
        lambda value: 20 if isinstance(value, dict) else 20,
    )

    merged = s._merge_generic_list_field(
        field_name="facts",
        field_description="Structured facts",
        annotation=NestedSchema.model_fields["facts"].annotation,
        candidates=[facts],
        language="English",
        target_tokens=160,
    )

    assert len(merged) == 8
    assert all(isinstance(item, NestedFact) for item in merged)
    assert [item.fact for item in merged] == [f"Structured fact {i}" for i in range(8)]
    assert merged[0].provenance is not None
    assert merged[0].provenance.location_hint == "Section 0"
    assert merged[0].attribution is not None
    assert merged[0].attribution.work_or_publication == "Source 0"


def test_llm_merge_generic_field_prompt_includes_exact_schema(monkeypatch) -> None:
    s = _make_summarizer()
    captured = {}

    def fake_call(fn, payload, **kwargs):
        del payload
        captured["prompt_text"] = kwargs["prompt_text"]
        return fn.data_model(
            value=[
                NestedFact(
                    fact="Merged fact",
                    provenance=NestedProvenance(location_hint="Section merged"),
                    attribution=NestedAttribution(work_or_publication="Merged source"),
                    confidence=0.91,
                )
            ]
        )

    monkeypatch.setattr(s, "_call_validated_function", fake_call)

    merged = s._llm_merge_generic_field(
        field_name="facts",
        field_description="Structured facts",
        annotation=NestedSchema.model_fields["facts"].annotation,
        candidates=[
            [
                NestedFact(
                    fact="Candidate A",
                    provenance=NestedProvenance(location_hint="Section A"),
                    attribution=NestedAttribution(work_or_publication="Source A"),
                    confidence=0.8,
                )
            ],
            [
                NestedFact(
                    fact="Candidate B",
                    provenance=NestedProvenance(location_hint="Section B"),
                    attribution=NestedAttribution(work_or_publication="Source B"),
                    confidence=0.9,
                )
            ],
        ],
        language="English",
        target_tokens=120,
    )

    assert isinstance(merged, list)
    assert merged[0].fact == "Merged fact"
    assert "Exact target schema:" in captured["prompt_text"]
    assert '"fact" (string, required)' in captured["prompt_text"]
    assert '"provenance" (nested object' in captured["prompt_text"]


def test_prompt_includes_asset_metadata_guidance_for_scientific_paper() -> None:
    s = HierarchicalSummaryV2(
        content="Paper content",
        document_name="paper.txt",
        data_model=RichSchema,
        min_num_chunks=2,
        min_chunk_size=64,
        max_chunk_size=256,
        max_output_tokens=None,
        user_prompt="extract useful material",
        include_quotes=False,
        tokenizer_name="gpt2",
        chunker_name="SemanticHybridChunker",
    )
    s.document_type = DocumentType.SCIENTIFIC_PAPER

    prompt = s.prompt

    assert "[Asset Metadata Guidance]" in prompt
    assert "prioritize title, authors, publisher_or_collection, and publication_year" in prompt
    assert "Extract the title, authors, and publication details." in prompt
    assert '"asset_metadata"' in prompt


def test_prompt_includes_asset_metadata_guidance_for_interview() -> None:
    s = HierarchicalSummaryV2(
        content="Interview transcript",
        document_name="interview.txt",
        data_model=RichSchema,
        min_num_chunks=2,
        min_chunk_size=64,
        max_chunk_size=256,
        max_output_tokens=None,
        user_prompt="extract useful material",
        include_quotes=False,
        tokenizer_name="gpt2",
        chunker_name="SemanticHybridChunker",
    )
    s.document_type = DocumentType.INTERVIEW

    prompt = s.prompt

    assert "[Asset Metadata Guidance]" in prompt
    assert "prioritize speakers and keep speaker-specific claims separated" in prompt
    assert "Identify and distinguish between different speakers." in prompt


def test_asset_metadata_survives_budget_enforcement(monkeypatch) -> None:
    s = HierarchicalSummaryV2(
        content="alpha paragraph\n\nsecond paragraph\n\nthird paragraph",
        document_name="rich.txt",
        data_model=RichSchema,
        min_num_chunks=2,
        min_chunk_size=64,
        max_chunk_size=256,
        max_output_tokens=40,
        user_prompt="extract useful material",
        include_quotes=False,
        tokenizer_name="gpt2",
        chunker_name="SemanticHybridChunker",
    )

    result = RichSchema(
        summary=("A very long summary that should be shortened while metadata survives. " * 6).strip(),
        facts=[
            NestedFact(
                fact="Structured fact",
                provenance=NestedProvenance(location_hint="Section: insight; citation [1]"),
                attribution=NestedAttribution(work_or_publication="Trusted Title"),
                confidence=0.82,
            )
        ],
        asset_metadata=NestedAssetMetadata(
            document_type="interview",
            title="Listening Interview",
            authors=["Editor A"],
            speakers=["Speaker A", "Speaker B"],
            publisher_or_collection="Interview Series",
            publication_year="2024",
        ),
    )

    token_counts = iter([120, 20])
    monkeypatch.setattr(
        s,
        "compute_required_tokens_graceful",
        lambda *_args, **_kwargs: next(token_counts),
    )

    s._enforce_budget(result)

    assert result.summary.endswith("...")
    assert result.asset_metadata is not None
    assert result.asset_metadata.document_type == "interview"
    assert result.asset_metadata.title == "Listening Interview"
    assert result.asset_metadata.authors == ["Editor A"]
    assert result.asset_metadata.speakers == ["Speaker A", "Speaker B"]
    assert result.asset_metadata.publisher_or_collection == "Interview Series"
    assert result.asset_metadata.publication_year == "2024"
