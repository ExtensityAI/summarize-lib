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

from src import HierarchicalSummary
from src.hierarchical_v2 import HierarchicalSummaryV2, get_current_tokenizer
from src.hierarchical_v2_reduce import deduplicate_string_items, dedupe_unbounded_summary_text


class UnitSchema(LLMDataModel):
    summary: str = Field(description="Summary text")
    facts: List[str] = Field(default_factory=list, description="Facts")
    type: Optional[str] = None


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
