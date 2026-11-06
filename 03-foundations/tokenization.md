---
id: foundations.tokenization
title: Tokenization
layer: 03-foundations
pathways: [fundamentals, builder, practitioner]
level: intro
prereqs: []
status: stable
last_reviewed: 2026-11-06
review_every_days: 180
---

# Tokenization

## TL;DR

Models don't read characters or words. They read **tokens**: chunks of text from a fixed vocabulary, usually learned by **byte-pair encoding (BPE)**. Tokenization decides how much a prompt costs, how much fits in context, and explains odd failures like miscounting letters or worse performance in some languages.

## Intuition

A model needs a fixed list of symbols, each mapped to an ID and then a vector. There are three options:

- **Characters or bytes:** a tiny vocabulary, but sequences are long, and attention cost grows with length.
- **Whole words:** short sequences, but the vocabulary explodes, and new words are unknown.
- **Subwords:** common words become one token, rare words split into pieces. This is the compromise almost every model uses.

BPE builds that subword vocabulary from data: start with single bytes, find the pair that appears next to each other most often, glue it into a new token, and repeat until the vocabulary is the size you want. Frequent words end up as single tokens; rare words fall back to smaller pieces, and in the worst case to raw bytes, so **any** text can be encoded.

## How it works

**Training** (done once, on a large corpus):

1. Convert text to UTF-8 bytes. The starting vocabulary is the 256 possible byte values.
2. **Pre-tokenize:** split text into chunks (words, numbers, punctuation, usually with the leading space attached) so merges never cross those boundaries. That's why `" cat"` with a leading space is a different token from `"cat"`.
3. Count every adjacent pair of tokens, merge the most frequent pair into a new token, and record the merge.
4. Repeat until the vocabulary reaches the target size.

**Encoding** new text replays the learned merges in the order they were learned. **Decoding** looks up each token's bytes and joins them.

Two other algorithms are common. **WordPiece** (BERT) picks merges by likelihood rather than raw frequency. **Unigram** (in the SentencePiece library) starts with a large vocabulary and prunes it. Most modern LLMs use byte-level BPE.

**Special tokens** are reserved IDs outside normal text, such as end-of-text and the markers that separate system, user, assistant and tool messages in chat templates. Tool calling depends on them (see [08 Agents](../08-agents/agents.md)).

## Code

[`s2a.bpe`](../src/s2a/bpe.py) is a minimal byte-level BPE tokenizer, about 100 lines and tested in CI. Try it in the [tokenizer lab](../labs/tokenizer.ipynb).

```python
from s2a.bpe import BPETokenizer

corpus = "the cat sat on the mat. the cat ate the rat. " * 20
tok = BPETokenizer().train(corpus, vocab_size=300)

tok.tokens("the cat sat")          # ['the', ' cat', ' sat']
tok.tokens("cats")                 # ['c', 'at', 's']: unseen word, split into pieces
tok.decode(tok.encode("naïve 東京"))  # round-trips any text, via raw bytes
```

## Numbers that matter

| Tokenizer | Vocabulary size |
|---|---|
| GPT-2 | 50,257 |
| Llama 2 (SentencePiece BPE) | 32,000 |
| GPT-4 (`cl100k_base`) | about 100,000 |
| Llama 3 | 128,256 |

- **Rule of thumb:** with modern English tokenizers, one token is about 4 characters, or about ¾ of a word. Code, numbers and most non-English text use more tokens per word.
- **Vocabulary size is a trade-off.** A bigger vocabulary means shorter sequences, so cheaper attention and a smaller [KV cache](../06-inference/kv-cache.md) per sentence, but bigger embedding and output layers. Moving from 32k to 128k is one reason newer models encode the same text in fewer tokens.
- **Fairness:** the same sentence can take several times more tokens in some languages than in English (Petrov et al., 2023). Those users pay more per sentence and fit less in the context window.

## Why tokenization explains odd behavior

- **Spelling and counting letters:** the model sees `strawberry` as a few tokens, not ten letters, so counting the r's means recalling something it never directly sees.
- **Arithmetic:** numbers split into irregular chunks (`12345` may become `123` + `45`). Some tokenizers split digits one by one, or in groups of three, to help.
- **Leading spaces and formatting:** `"Hello"` and `" Hello"` are different tokens. Trailing spaces in prompts can change outputs.
- **Glitch tokens:** tokens that were common in the tokenizer's training data but rare in the model's (the famous `SolidGoldMagikarp`) get badly trained embeddings and can cause bizarre outputs.

## Current consensus (as of 2026-11)

- Byte-level BPE with regex pre-tokenization is the default for LLMs, with vocabularies of roughly 100k–250k tokens, larger for strongly multilingual models.
- Tokenizers are trained once and frozen with the model; changing one later means retraining or careful vocabulary surgery.
- **Tokenizer-free** models that work on bytes directly, grouping them dynamically (for example Meta's Byte Latent Transformer), are an active research direction but not yet mainstream.

## What changed

- 2026-11: First version of this page.

## Open questions

- Will byte-level, tokenizer-free models catch up enough on efficiency to replace BPE?

## Resources

- Karpathy, [*Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE) (video) and [minbpe](https://github.com/karpathy/minbpe) (F, B, P)
- Sennrich, Haddow & Birch, [*Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909) (BPE for NLP, 2016) (P)
- Kudo & Richardson, [*SentencePiece*](https://arxiv.org/abs/1808.06226) (2018) and Kudo, [*Subword Regularization*](https://arxiv.org/abs/1804.10959) (Unigram, 2018) (P)
- Petrov et al., [*Language Model Tokenizers Introduce Unfairness Between Languages*](https://arxiv.org/abs/2305.15619) (2023) (B, P)
- Pagnoni et al., [*Byte Latent Transformer*](https://arxiv.org/abs/2412.09871) (2024) (P)
- OpenAI [tiktoken](https://github.com/openai/tiktoken), to see real GPT tokenizations (B)

## Check yourself

1. Why does byte-level BPE never produce an "unknown token"?
2. Train the lab tokenizer with and without pre-tokenization. What kinds of tokens appear without it, and why are they a problem?
3. Your app's prompts are mostly in Hindi. What does that mean for cost and context length compared with English, and what could you measure first?
