"""Unit tests for Destination RAG tool, chunking, and in-memory vector store."""

import pytest
from pathlib import Path
from vectorstore.ingest import (
    parse_destination_file,
    load_raw_destination_files,
    KNOWN_SECTIONS,
    DocumentChunk,
)
from vectorstore.store import InMemoryVectorStore, cosine_similarity
from tools.destination_rag import search_destination_guide


def test_cosine_similarity():
    """Verify cosine similarity calculation."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v2), 0.001) == 1.0

    v_orth = [0.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v_orth), 0.001) == 0.0

    v_zero = [0.0, 0.0, 0.0]
    assert cosine_similarity(v1, v_zero) == 0.0


def test_parse_destination_file(tmp_path: Path):
    """Verify section-based parsing and metadata extraction."""
    sample_text = (
        "DESTINATION GUIDE: Kyoto, Japan\n\n"
        "VISA & ENTRY\n"
        "Tourist visa exemption applies for up to 90 days.\n\n"
        "BEST TIME TO VISIT\n"
        "Spring for sakura blossoms and autumn for red maples.\n\n"
        "LOCAL CUSTOMS\n"
        "Bow when greeting and do not tip at restaurants.\n\n"
        "PACKING TIPS\n"
        "Wear comfortable shoes for walking around stone temples.\n\n"
        "SAFETY & HEALTH\n"
        "Very safe city with potable tap water.\n"
    )
    guide_file = tmp_path / "kyoto.txt"
    guide_file.write_text(sample_text, encoding="utf-8")

    chunks = parse_destination_file(guide_file)
    assert len(chunks) == 5

    sections_found = [c.section for c in chunks]
    assert sections_found == KNOWN_SECTIONS

    for chunk in chunks:
        assert chunk.city == "Kyoto"
        assert chunk.country == "Japan"
        assert len(chunk.content) > 0
        assert f"[KYOTO - {chunk.section}]" in chunk.formatted_text


def test_load_all_raw_destination_files():
    """Verify all raw destination guide files in repository are properly parsed."""
    chunks = load_raw_destination_files()
    assert len(chunks) >= 20  # 4 cities * 5 sections = 20 chunks

    cities = {c.city.lower() for c in chunks}
    assert "bangkok" in cities
    assert "barcelona" in cities
    assert "reykjavik" in cities
    assert "tokyo" in cities


def test_vector_store_similarity_search_with_mock_embedder():
    """Verify vector search with a deterministic mock embedder."""
    def mock_embedder(texts: list[str]) -> list[list[float]]:
        # Deterministic 16-dim vector based on character frequencies
        results = []
        for text in texts:
            vec = [0.0] * 16
            for ch in text.lower():
                vec[ord(ch) % 16] += 1.0
            norm = sum(x * x for x in vec) ** 0.5
            results.append([x / norm for x in vec] if norm > 0 else vec)
        return results

    store = InMemoryVectorStore(embedder=mock_embedder)
    docs = [
        DocumentChunk(
            chunk_id="tokyo_visa",
            city="Tokyo",
            country="Japan",
            section="VISA & ENTRY",
            content="Japan provides 90-day visa exemptions for tourists.",
        ),
        DocumentChunk(
            chunk_id="tokyo_packing",
            city="Tokyo",
            country="Japan",
            section="PACKING TIPS",
            content="Bring warm layers and comfortable walking shoes for Tokyo in winter.",
        ),
        DocumentChunk(
            chunk_id="bangkok_packing",
            city="Bangkok",
            country="Thailand",
            section="PACKING TIPS",
            content="Pack lightweight breathable cotton clothes for tropical heat.",
        ),
    ]
    store.add_documents(docs)

    # Search for Tokyo packing
    results = store.similarity_search("What should I pack for Tokyo?", k=2, city="Tokyo")
    assert len(results) >= 1
    assert results[0].city == "Tokyo"
    assert results[0].section == "PACKING TIPS"


def test_search_destination_guide_empty_query():
    """Verify input validation on empty query."""
    res = search_destination_guide.invoke({"query": ""})
    assert len(res) == 1
    assert "Error: Search query cannot be empty" in res[0]


def test_search_destination_guide_tool_invocation():
    """Verify search_destination_guide LangChain tool execution."""
    res = search_destination_guide.invoke({"query": "Tokyo visa requirements"})
    assert isinstance(res, list)
    assert len(res) > 0
    assert any("TOKYO" in chunk for chunk in res)
