# Changelog

All notable changes to this knowledge base. Releases use calendar versioning (`vYYYY.MM`) and the [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

## Unreleased

### Added

- Topic page: [The language-modeling objective](03-foundations/lm-objective.md), with loss, perplexity and a bigram model (`s2a.lm`) and a [language-modeling lab](labs/lm-objective.ipynb). Completes the Foundations layer.
- Topic page: [Embeddings & positions](03-foundations/embeddings-positions.md), with RoPE and sinusoidal encodings (`s2a.positions`) and a [positions lab](labs/positions.ipynb).
- Interactive [KV-cache calculator](https://ankitrgupta1.github.io/silicon-to-agents/widgets/kv-cache.html) that runs in the browser.
- Hidden answers in the *Check yourself* sections, and in the topic template.
- Site build check on every pull request.
- Topic page: [Tokenization](03-foundations/tokenization.md), with a from-scratch byte-level BPE tokenizer (`s2a.bpe`) and a [tokenizer lab](labs/tokenizer.ipynb).
- Repository scaffold: nine stack layers, prerequisites and cross-cutting sections.
- Three learning pathways: Fundamentals, Builder, Practitioner.
- Topic template and AGENTS.md conventions.
- First topic page: [Attention](03-foundations/attention.md).
- KV-cache calculator (`s2a.kv_cache`) with tests and a Colab-ready notebook.
- Website built with Jupyter Book 2 and deployed to GitHub Pages.
- CI for tests, plus a weekly link check.
- Topic page: [KV cache](06-inference/kv-cache.md).
- Licenses: MIT for code, CC BY 4.0 for written content.

### Changed

- Layer overview pages are named after their layer (e.g. `00-hardware/hardware.md`) so site URLs are readable, like `/hardware`.
- GitHub Actions updated to their Node 24 versions.
