# AGENTS.md

Instructions for AI coding agents working in this repository.

## What this repo is

A living knowledge base about the AI stack, from hardware to agents. Markdown topic pages are the source of truth; reusable code lives in `src/s2a/`; notebooks import from that package instead of copying code.

## Conventions

- New topic pages start from `_templates/topic.md` and keep its front-matter fields. `id` values look like `layer.topic-name` and must be unique.
- When you change a topic's content, update `last_reviewed` and add a dated line under **What changed**.
- Add new pages to the `toc` in `myst.yml` so they appear on the website.
- Link between pages with relative links to `.md` files so links work on both GitHub and the site.
- Cite sources with links. Prefer papers, official docs and specs over aggregators. Don't state version numbers or hardware specs without a source and date.
- Keep prose plain and direct. Explain intuition before math.
- Record notable changes in `CHANGELOG.md` under `Unreleased`.

## Branches and pull requests

- Never commit directly to `main`. `main` is what the website publishes.
- Start every update on a new branch from the latest `main`, named by type: `content/<topic>` (new or updated pages), `code/<thing>`, `fix/<thing>`, `chore/<thing>` (tooling, CI, config).
- Keep one topic or change per branch, and open a pull request into `main`. CI must pass before merging.
- Merging to `main` deploys the site automatically.

## Code

- Python 3.10+, tested with `pytest`. Run `pip install -e ".[dev]" && pytest` before committing.
- Notebooks must run on CPU unless tagged otherwise, and must start with an *Open in Colab* badge.

## Don't

- Don't auto-merge AI-drafted digests or content; open a pull request for human review.
- Don't copy large passages from books, papers or articles; summarize and link.
