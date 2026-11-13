---
id: applications.rag
title: RAG Basics
layer: 07-applications
pathways: [fundamentals, builder]
level: intro
prereqs: [applications.prompting, prereqs.linear-algebra]
status: evolving
last_reviewed: 2026-11-12
review_every_days: 120
---

# RAG Basics

## TL;DR

**Retrieval-augmented generation (RAG)** answers questions from your own documents: split them into chunks, find the chunks most relevant to the question, and put them in the prompt with instructions to answer only from them and cite them. Retrieval quality decides most of the result, so the core skills are chunking well, combining **keyword** and **dense** search (**hybrid**), and **measuring** retrieval with a small set of real questions.

## Intuition

A model only knows what was in its training data, frozen at some date, and nothing about your private documents. You could fine-tune it on them, but that's slow, costly, and poor at teaching precise facts. RAG takes the open-book-exam approach: when a question comes in, look up the relevant pages first and hand them to the model along with the question.

Two consequences follow. The model can now cite sources, which makes answers checkable. And when the right page isn't retrieved, even the best model can't answer, so most of the engineering effort goes into the lookup.

## How it works

```text
Offline:  documents -> chunks -> index (keyword index and/or vectors)
Online:   question -> search -> top-k chunks -> prompt -> model -> answer with citations
```

### 1. Chunking

Split documents into passages small enough to be specific and large enough to make sense on their own, typically a few hundred words.

- **Follow the document's structure.** Split at headings and paragraphs, not mid-sentence.
- **Overlap** neighboring chunks a little, so an idea cut at a boundary appears whole somewhere.
- **Keep metadata:** source file, heading, date. It enables citations and filtering.

### 2. Keyword search (BM25)

Score each chunk by the query words it contains. Rare words count more (they carry more information), repeated words have diminishing returns, and long chunks are slightly penalized. BM25 is fast, needs no model, and excels at exact terms: names, error codes, product numbers. It can't match a word that isn't there: a question about "automobiles" won't find a chunk that only says "car".

### 3. Dense search (embeddings)

An **embedding model** turns each chunk and the question into vectors so that texts with similar meaning land near each other. Ranking by [cosine similarity](../prereqs/linear-algebra.md) then finds relevant chunks even with no words in common. Store the vectors in a vector index (often a vector database) for fast nearest-neighbor search over millions of chunks.

The lab uses **LSA** (TF-IDF vectors compressed with the SVD) as a tiny stand-in for a neural embedding model. The retrieval mechanics are identical; real embedding models are far better at meaning.

### 4. Hybrid search and reranking

Keyword and dense search fail in different ways, so combining them is the robust default. **Reciprocal rank fusion** merges the two rankings by giving each chunk $\sum 1/(60 + \text{rank})$ across lists, which avoids comparing incompatible score scales. A **reranker** can then rescore the top few dozen candidates with a slower, more accurate model that reads the question and chunk together.

### 5. The prompt

Number the retrieved chunks, put them before the question, and give explicit rules: answer only from the sources, cite them, and say "I don't know" when they don't contain the answer (see [Prompting](prompting.md)).

### 6. Evaluate

Write down 20–50 real questions and which document answers each. Measure **recall@k**: the share of questions where a right chunk is in the top k. Then separately check the answers themselves: are they correct, and supported by the cited sources? Change one thing at a time and re-measure.

## Code

[`s2a.rag`](../src/s2a/rag.py) implements chunking, BM25, LSA dense search, hybrid fusion, recall@k and prompt building, with tests that run retrieval over this repository's own pages. The [RAG lab](../labs/rag.ipynb) builds a search engine for this knowledge base, no API key needed.

```python
from s2a.rag import Retriever, load_markdown_corpus, build_rag_messages

retriever = Retriever(load_markdown_corpus("."))          # this repo's topic pages
hits = retriever.search("how much GPU memory does the KV cache need per token", k=3)
messages = build_rag_messages("How much memory does the KV cache use?", hits)
```

## Numbers that matter

**This repo as a corpus** (when this page was written, including this page): 24 pages split into 235 chunks of at most 150 words. The numbers below shift a little as pages are added; rerun the lab for current ones.

**Recall from the lab**, on two small question sets about this repo:

| Method | Keyword-style (20 questions) recall@1 / @5 | Paraphrased (10 questions) recall@1 / @5 |
|---|---|---|
| BM25 | 0.85 / 1.00 | 0.90 / 1.00 |
| Dense (LSA) | 0.90 / 1.00 | 0.80 / 1.00 |
| Hybrid | 0.90 / 1.00 | 1.00 / 1.00 |

- **Hybrid is the most reliable.** It matches or beats both single methods on each set, even though each one alone gets some questions wrong at rank 1.
- **On a small, well-organized corpus, keyword search is already strong.** Each page uses its topic's vocabulary, so even paraphrased questions share some words with the right page. Everything reaches recall@5 of 1.00; the differences are in what comes *first*. A neural embedding model would matter more on larger, messier collections.
- **Dense search can match meaning without shared words.** In the lab's toy example, "my car engine keeps stalling" scores 0 for the query "automobile" under BM25 but 1.0 under dense search, because "car" and "automobile" appear together in other documents.
- **Thirty questions is a small sample**: one question changes recall by 3–10 points. Treat these as a sanity check, not a benchmark.

## Current consensus (as of 2026-11)

- Hybrid retrieval (keyword plus embeddings) followed by a reranker is the common production pattern.
- Long context windows haven't made RAG obsolete: retrieval is still cheaper and faster than putting everything in context, works over collections far larger than any window, and gives clear citations. The two are often combined.
- **Agentic RAG**, where the model decides when and what to search, possibly over several rounds, is increasingly common for complex questions (see [08 Agents](../08-agents/agents.md)).
- Evaluation with a fixed question set is considered essential; retrieval changes that "look better" often aren't.

## What changed

- 2026-11: First version of this page.

## Open questions

- How should knowledge be split between retrieval, long context and fine-tuning as context windows grow and get cheaper?

## Resources

- Lewis et al., [*Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*](https://arxiv.org/abs/2005.11401) (2020) (B, P)
- Robertson & Zaragoza, [*The Probabilistic Relevance Framework: BM25 and Beyond*](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) (2009) (P)
- Karpukhin et al., [*Dense Passage Retrieval for Open-Domain Question Answering*](https://arxiv.org/abs/2004.04906) (2020) (P)
- Cormack, Clarke & Büttcher, [*Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (2009) (P)
- Anthropic, [*Introducing Contextual Retrieval*](https://www.anthropic.com/news/contextual-retrieval) (2024) (B)
- Liu et al., [*Lost in the Middle: How Language Models Use Long Contexts*](https://arxiv.org/abs/2307.03172) (2023) (B, P)
- Chip Huyen, *AI Engineering*, chapter on RAG and agents (B)

## Check yourself

Try each one before opening the answer.

**1. Users search your support docs for error codes like "E4471". Dense search alone often returns the wrong article. Why, and what fixes it?**

<details>
<summary>Answer</summary>

Embedding models are trained to capture meaning, and a rare code like "E4471" carries little meaning for them, so it can get blurred with similar-looking codes or ignored. Keyword search treats it as an exact, rare (and therefore highly weighted) term. Use hybrid search so the keyword side catches exact identifiers and the dense side catches paraphrases.

</details>

**2. Your RAG app gives a wrong answer. How do you tell whether retrieval or generation is at fault?**

<details>
<summary>Answer</summary>

Look at what was retrieved. If the chunk containing the right answer wasn't in the prompt, it's a retrieval problem: work on chunking, search or the number of chunks retrieved, and track recall@k on your question set. If the right chunk was there and the model still answered wrongly, it's a generation problem: work on the prompt, the order of sources, or the model. Logging the retrieved chunks with every answer makes this check routine.

</details>

**3. Why does a RAG prompt tell the model to say "I don't know" when the sources don't contain the answer?**

<details>
<summary>Answer</summary>

Retrieval always returns *something*, even for questions the documents don't cover. Without the rule, the model tends to answer anyway, either stretching weakly related sources or falling back on its own training data, and presenting the result as if it were grounded. An explicit "I don't know" is far more useful to users than a confident, uncited guess.

</details>
