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

## Interactivity

Learning here should be hands-on wherever that builds intuition. Every topic page gets, in order of priority:

1. **Check yourself with hidden answers.** Two or three questions, each followed by a `<details><summary>Answer</summary> … </details>` block, with blank lines around the inner text so Markdown renders. Answers explain the reasoning, not just the result. This works both on GitHub and on the site.
2. **A lab or calculator notebook** when the topic has code: `labs/` for hands-on exploration, `calculators/` for sizing tools. Notebooks run on CPU, start with an *Open in Colab* badge, and import from `s2a` with a fallback `%pip install` from GitHub.
3. **An interactive widget** only for core ideas where moving a slider or stepping through something teaches more than text (aim for the 10–15 most important concepts, not every page).

Widget rules:

- One self-contained HTML file per widget in `widgets/`: inline CSS and JS, no external libraries or network requests, light and dark mode via `prefers-color-scheme`, mobile-first layout.
- Any math must match the tested Python in `src/s2a/`. Before committing, load the widget in a headless browser, check there are no JS errors, and compare its outputs with the Python functions.
- The deploy workflow copies `widgets/` into the site. Link to widgets with absolute URLs (`https://ankitrgupta1.github.io/silicon-to-agents/widgets/<name>.html`) so links work on GitHub and the site.
- Pages must still read well as plain text; interactivity adds to the explanation, never replaces it.
- GPU-heavy work (training, kernels, serving) stays in Colab notebooks or recorded results, since it can't run in a browser.

## Branches and pull requests

- Never commit directly to `main`. `main` is what the website publishes.
- Start every update on a new branch from the latest `main`, named by type: `content/<topic>` (new or updated pages), `code/<thing>`, `fix/<thing>`, `chore/<thing>` (tooling, CI, config).
- Keep one topic or change per branch, and open a pull request into `main`. CI tests and the site build check must pass before merging.
- Merging to `main` deploys the site automatically.

## Code

- Python 3.10+, tested with `pytest`. Run `pip install -e ".[dev]" && pytest` before committing.
- Notebooks must run on CPU unless tagged otherwise, and must start with an *Open in Colab* badge.
- `tests/test_notebooks.py` runs every notebook's code cells in CI. Notebooks that train must read their step count from the `S2A_STEPS` environment variable (with a sensible default) so the CI run stays short.
- PyTorch code lives in its own modules (for example `s2a.gpt`), so the rest of `s2a` imports without it. Tests for those modules start with `pytest.importorskip("torch")`. CI installs CPU PyTorch; if your environment has no PyTorch, rely on the CI results and say so in the pull request.
- Where a PyTorch module mirrors a NumPy one, keep a test that both give the same outputs from the same weights.

## Don't

- Don't auto-merge AI-drafted digests or content; open a pull request for human review.
- Don't copy large passages from books, papers or articles; summarize and link.
