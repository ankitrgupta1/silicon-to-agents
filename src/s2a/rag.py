"""Retrieval-augmented generation, minus the generator.

Everything needed to find the right passages for a question and hand them to
a model: chunking, keyword search (BM25), a small "dense" search (LSA: TF-IDF
compressed with the SVD, a stand-in for neural embeddings), hybrid search with
reciprocal rank fusion, recall@k evaluation, and building the final prompt.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STOPWORDS = set("""
a an and are as at be but by can do does for from has have how i if in into is it its of on or
so than that the their them then there these this to was what when where which who why will with
you your not no
""".split())

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, minus very common words."""
    return [w for w in _WORD.findall(text.lower()) if w not in STOPWORDS]


@dataclass(frozen=True)
class Chunk:
    source: str      # file the chunk came from
    heading: str     # nearest heading above it
    text: str

    def __str__(self) -> str:
        return f"[{self.source} › {self.heading}] {self.text[:80]}…"


def chunk_markdown(text: str, source: str, max_words: int = 150, overlap: int = 30) -> list[Chunk]:
    """Split a Markdown page into chunks under headings, each at most `max_words` words.

    Front matter is dropped. Long sections are split into overlapping windows
    so a sentence cut at a boundary still appears whole in one chunk.
    """
    if not 0 <= overlap < max_words:
        raise ValueError("need 0 <= overlap < max_words")
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.S)
    chunks: list[Chunk] = []
    heading, buf = "(top)", []

    def flush():
        words = " ".join(buf).split()
        step = max_words - overlap
        for start in range(0, max(len(words) - overlap, 1), step):
            piece = " ".join(words[start:start + max_words])
            if piece.strip():
                chunks.append(Chunk(source, heading, piece))

    for line in text.splitlines():
        m = re.match(r"^#{1,4}\s+(.*)", line)
        if m:
            flush()
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    flush()
    return chunks


def load_markdown_corpus(root: str | Path, exclude: tuple[str, ...] = ("_templates", "pathways")) -> list[Chunk]:
    """Chunk every topic page under `root` (one level of folders, like this repo)."""
    root = Path(root)
    chunks = []
    for path in sorted(root.glob("*/*.md")):
        if path.parent.name not in exclude and not path.parent.name.startswith((".", "_")):
            chunks += chunk_markdown(path.read_text(encoding="utf-8"), str(path.relative_to(root)))
    return chunks


class BM25:
    """Classic keyword ranking: rare query words that appear often in a short chunk score high."""

    def __init__(self, docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs, self.k1, self.b = docs, k1, b
        self.tf = [Counter(d) for d in docs]
        self.avg_len = sum(map(len, docs)) / max(len(docs), 1)
        df = Counter(w for d in docs for w in set(d))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        out = np.zeros(len(self.docs))
        for i, (tf, d) in enumerate(zip(self.tf, self.docs)):
            norm = self.k1 * (1 - self.b + self.b * len(d) / self.avg_len)
            for w in query:
                if w in tf:
                    out[i] += self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + norm)
        return out


class LSA:
    """Tiny dense retrieval: TF-IDF vectors compressed to `dim` dimensions with the SVD.

    Chunks that use related words end up near each other even without exact
    overlap. Neural embedding models do this far better; the retrieval
    mechanics (embed, then rank by cosine similarity) are the same.
    """

    def __init__(self, docs: list[list[str]], dim: int = 64) -> None:
        self.vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in d}))}
        n = len(docs)
        df = np.zeros(len(self.vocab))
        for d in docs:
            for w in set(d):
                df[self.vocab[w]] += 1
        self.idf = np.log((1 + n) / (1 + df)) + 1
        X = np.stack([self._tfidf(d) for d in docs])
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.dim = min(dim, len(S))
        self.proj = Vt[: self.dim].T                     # (vocab, dim)
        self.doc_vecs = self._unit(X @ self.proj)

    def _tfidf(self, words: list[str]) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for w, c in Counter(words).items():
            if w in self.vocab:
                v[self.vocab[w]] = (1 + math.log(c)) * self.idf[self.vocab[w]]
        n = np.linalg.norm(v)
        return v / n if n else v

    @staticmethod
    def _unit(M: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(M, axis=-1, keepdims=True)
        return M / np.where(n == 0, 1, n)

    def embed(self, words: list[str]) -> np.ndarray:
        return self._unit(self._tfidf(words) @ self.proj)

    def scores(self, query: list[str]) -> np.ndarray:
        return self.doc_vecs @ self.embed(query)          # cosine similarity


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Merge ranked lists: each item scores sum(1 / (k + rank)). Robust to different score scales."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            score[item] = score.get(item, 0.0) + 1 / (k + rank)
    return sorted(score, key=lambda i: -score[i])


class Retriever:
    def __init__(self, chunks: list[Chunk], dim: int = 64) -> None:
        self.chunks = chunks
        docs = [tokenize(f"{c.heading} {c.text}") for c in chunks]
        self.bm25 = BM25(docs)
        self.lsa = LSA(docs, dim)

    def search(self, query: str, k: int = 5, method: str = "hybrid") -> list[Chunk]:
        q = tokenize(query)
        if method == "bm25":
            order = list(np.argsort(-self.bm25.scores(q), kind="stable"))
        elif method == "dense":
            order = list(np.argsort(-self.lsa.scores(q), kind="stable"))
        elif method == "hybrid":
            order = reciprocal_rank_fusion([
                list(np.argsort(-self.bm25.scores(q), kind="stable")),
                list(np.argsort(-self.lsa.scores(q), kind="stable")),
            ])
        else:
            raise ValueError("method must be 'bm25', 'dense' or 'hybrid'")
        return [self.chunks[i] for i in order[:k]]


def recall_at_k(retriever: Retriever, eval_set: list[tuple[str, str | tuple[str, ...]]], k: int = 5,
                method: str = "hybrid") -> float:
    """Share of questions with a relevant source in the top k results.

    Each question lists one relevant source, or a tuple of sources that all count.
    """
    hits = 0
    for q, relevant in eval_set:
        ok = {relevant} if isinstance(relevant, str) else set(relevant)
        hits += any(c.source in ok for c in retriever.search(q, k, method))
    return hits / len(eval_set)


def build_rag_messages(question: str, chunks: list[Chunk]) -> list[dict]:
    """The prompt a RAG app sends: numbered sources, then the question, with grounding rules."""
    sources = "\n\n".join(f"[{i}] ({c.source} › {c.heading})\n{c.text}" for i, c in enumerate(chunks, 1))
    system = ("Answer using only the numbered sources. Cite them like [1]. "
              "If the sources don't contain the answer, say you don't know.")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"<sources>\n{sources}\n</sources>\n\nQuestion: {question}"},
    ]


# Questions about this repository and the page that answers each one.
REPO_EVAL_SET: list[tuple[str, str]] = [
    ("how much GPU memory does the KV cache need per token", "06-inference/kv-cache.md"),
    ("why divide attention scores by the square root of the key dimension", "03-foundations/attention.md"),
    ("how does byte pair encoding decide which pairs to merge", "03-foundations/tokenization.md"),
    ("rotary position embeddings make scores depend on relative distance", "03-foundations/embeddings-positions.md"),
    ("what is perplexity and how is it related to the loss", "03-foundations/lm-objective.md"),
    ("where do the parameters of Llama 3 8B live", "04-architectures/transformer.md"),
    ("held-out loss rising while training loss falls", "05-training/tiny-gpt.md"),
    ("why do gradients add up when a value is used twice", "prereqs/backprop.md"),
    ("rules for broadcasting tensor shapes", "prereqs/pytorch.md"),
    ("how many trainable parameters does LoRA add", "prereqs/linear-algebra.md"),
    ("KL penalty that keeps an RLHF model close to the original", "prereqs/probability.md"),
    ("guarantee the model outputs valid JSON", "07-applications/prompting.md"),
    ("prefill versus decode memory bandwidth", "06-inference/kv-cache.md"),
    ("SwiGLU gated MLP", "04-architectures/transformer.md"),
    ("glitch tokens like SolidGoldMagikarp", "03-foundations/tokenization.md"),
    ("reuse a cached system prompt across requests", "07-applications/prompting.md"),
    ("confident and wrong predictions cost the most", "prereqs/probability.md"),
    ("extend a model trained on short sequences to long contexts", "03-foundations/embeddings-positions.md"),
    ("weight tying between input embedding and output layer", "03-foundations/embeddings-positions.md"),
    ("vectorized tensor operations are faster than python loops", "prereqs/pytorch.md"),
]


# The same kind of questions, worded without the pages' key terms.
REPO_PARAPHRASE_SET: list[tuple[str, str | tuple[str, ...]]] = [
    ("remembering earlier words so the model does not redo work for each new word", "06-inference/kv-cache.md"),
    ("how does the network know which word came first", "03-foundations/embeddings-positions.md"),
    ("splitting text into pieces before the model reads it", "03-foundations/tokenization.md"),
    ("how surprised the network is on average", ("prereqs/probability.md", "03-foundations/lm-objective.md")),
    ("nudging every weight downhill a little at a time", ("05-training/tiny-gpt.md", "prereqs/backprop.md")),
    ("why stacking matrices without anything in between is pointless", "prereqs/linear-algebra.md"),
    ("text hidden in an email telling the bot to do something else", "07-applications/prompting.md"),
    ("each word looks back at earlier words and copies what is relevant", "04-architectures/transformer.md"),
    ("fine tuning cheaply by learning two thin matrices", "prereqs/linear-algebra.md"),
    ("making sampling more random or more predictable", ("prereqs/probability.md", "03-foundations/lm-objective.md")),
]
