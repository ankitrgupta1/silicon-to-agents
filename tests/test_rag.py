from pathlib import Path

import pytest

from s2a.rag import (
    REPO_EVAL_SET,
    REPO_PARAPHRASE_SET,
    BM25,
    Chunk,
    Retriever,
    build_rag_messages,
    chunk_markdown,
    load_markdown_corpus,
    recall_at_k,
    reciprocal_rank_fusion,
    tokenize,
)

ROOT = Path(__file__).resolve().parent.parent

CARS_AND_CAKES = [
    "my car engine keeps stalling at red lights",
    "the automobile and car industry sells millions of cars",
    "an automobile is another word for a car",
    "car and automobile insurance prices rose this year",
    "the recipe calls for two cups of flour and sugar",
    "bake the cake for forty minutes until golden",
    "fresh bread needs flour water yeast and salt",
    "whisk the eggs with sugar before adding flour",
]


def tiny_corpus():
    return [Chunk(f"doc{i}", "", t) for i, t in enumerate(CARS_AND_CAKES)]


def test_tokenize_drops_stopwords():
    assert tokenize("What is the KV cache?") == ["kv", "cache"]


def test_chunking_follows_headings_and_drops_front_matter():
    md = "---\nid: x\n---\n# Title\nintro\n## Part A\nalpha beta\n## Part B\ngamma"
    chunks = chunk_markdown(md, "x.md")
    assert [(c.heading, c.text) for c in chunks] == [("Title", "intro"), ("Part A", "alpha beta"), ("Part B", "gamma")]


def test_long_sections_split_with_overlap():
    md = "## S\n" + " ".join(f"w{i}" for i in range(250))
    chunks = chunk_markdown(md, "x.md", max_words=100, overlap=20)
    assert [len(c.text.split()) for c in chunks] == [100, 100, 90]
    assert chunks[1].text.split()[0] == "w80"          # starts 20 words before the first chunk ended


def test_bm25_prefers_rare_matching_words():
    bm = BM25([["kv", "cache", "memory"], ["memory", "memory"], ["tokenizer"]])
    s = bm.scores(["kv", "memory"])
    assert s[0] > s[1] > s[2] == 0


def test_dense_search_finds_synonyms_keyword_search_misses():
    r = Retriever(tiny_corpus(), dim=2)
    q = tokenize("automobile")
    assert r.bm25.scores(q)[0] == 0                        # doc0 never says "automobile"
    assert r.lsa.scores(q)[0] > max(r.lsa.scores(q)[4:])   # but dense ranks it above every cooking doc


def test_reciprocal_rank_fusion():
    assert reciprocal_rank_fusion([[1, 2, 3], [3, 1, 2]])[0] == 1
    assert set(reciprocal_rank_fusion([[1], [2]])) == {1, 2}


def test_search_methods_and_errors():
    r = Retriever(tiny_corpus(), dim=2)
    for m in ("bm25", "dense", "hybrid"):
        assert len(r.search("flour sugar", k=3, method=m)) == 3
    with pytest.raises(ValueError):
        r.search("x", method="magic")


def test_prompt_numbers_sources_and_sets_rules():
    msgs = build_rag_messages("What stalls?", [Chunk("doc0", "Cars", "my car stalls")])
    assert "[1] (doc0 › Cars)" in msgs[1]["content"] and "don't know" in msgs[0]["content"]


def test_retrieval_over_this_repository():
    retriever = Retriever(load_markdown_corpus(ROOT))
    for _, src in REPO_EVAL_SET:
        assert (ROOT / src).exists(), src
    assert recall_at_k(retriever, REPO_EVAL_SET, k=5, method="hybrid") >= 0.9
    assert recall_at_k(retriever, REPO_PARAPHRASE_SET, k=5, method="hybrid") >= 0.8
