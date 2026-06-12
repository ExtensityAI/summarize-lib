"""Unit tests for the standalone HybridChunker (no network/embeddings)."""

from summarize_lib import HybridChunker


def _count(text, chunk_size):  # char/4 token estimate, deterministic
    return max(1, len(text) // 4)


def _rec(text, chunk_size):  # simple char-window recursive split
    w = max(1, chunk_size) * 4
    return [text[i : i + w] for i in range(0, len(text), w)] or [text]


def _chunker(**kw):
    kw.setdefault("count_tokens", _count)
    kw.setdefault("recursive_split", _rec)
    return HybridChunker(**kw)


def test_structural_packing_respects_cap_and_merges_small_segments():
    hc = _chunker(max_chunk_size=50, min_chunk_size=10, semantic=False)
    text = "# Heading\n\n" + ("Alpha. " * 8) + "\n\n" + ("Beta. " * 8)
    chunks = hc.chunk(text)
    assert chunks
    # no chunk materially exceeds the cap (allow the recursive-split tolerance)
    assert all(_count(c, 50) <= int(50 * 1.3) for c in chunks)
    # small adjacent paragraphs merged rather than 1 chunk each
    assert len(chunks) < text.count("\n\n") + 5


def test_oversized_single_segment_is_recursively_split():
    hc = _chunker(max_chunk_size=20, semantic=False)
    text = "word " * 200  # one long paragraph, no seams
    chunks = hc.chunk(text)
    assert len(chunks) > 1
    assert all(_count(c, 20) <= int(20 * 1.3) for c in chunks)


def test_empty_text_returns_empty():
    assert _chunker().chunk("") == []
    assert _chunker().chunk("   \n  ") == []


def test_semantic_falls_back_to_structural_when_embedder_returns_none():
    hc = _chunker(max_chunk_size=40, semantic=True, embed_batch=lambda texts: None,
                  prepare_segments=lambda s: s)
    text = "# A\n\n" + ("one. " * 6) + "\n\n" + ("two. " * 6)
    chunks = hc.chunk(text)
    assert chunks  # structural fallback still produces chunks
    assert all(_count(c, 40) <= int(40 * 1.3) for c in chunks)


def test_semantic_packs_at_similarity_boundary():
    # Two clusters; small cap so a semantic boundary actually triggers a flush.
    def fake_embed(texts):
        out = []
        for t in texts:
            out.append([1.0, 0.0] if "Alpha" in t else [0.0, 1.0])
        return out

    hc = _chunker(max_chunk_size=12, min_chunk_size=2, semantic=True,
                  embed_batch=fake_embed, prepare_segments=lambda s: s,
                  boundary_config=(0.5, 0.0, 0.5, 0.0))
    segs = ["Alpha one.", "Alpha two.", "Beta one.", "Beta two."]
    chunks = hc.pack_semantic(segs, 12)
    # the Alpha/Beta transition (low similarity) should split the run
    assert len(chunks) >= 2
    assert any("Alpha" in c and "Beta" not in c for c in chunks)


def test_stats_dict_is_populated_in_semantic_mode():
    stats = {}
    hc = _chunker(max_chunk_size=12, semantic=True,
                  embed_batch=lambda texts: [[1.0, 0.0] for _ in texts],
                  prepare_segments=lambda s: s)
    hc.pack_semantic(["a a a.", "b b b.", "c c c."], 12, stats=stats)
    assert "semantic_boundary_threshold_soft" in stats
    assert "semantic_boundary_threshold_hard" in stats
