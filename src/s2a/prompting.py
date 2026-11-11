"""What a prompt looks like to the model, and two mechanics apps rely on.

- Chat templates: a list of messages becomes one string with special
  markers, and the model simply continues it.
- Constrained decoding: at each step, mask out every token that would break
  the required format, so the output is always valid. This is how
  "structured output" / JSON modes can guarantee a parseable answer.
- Prompt-caching and context-budget arithmetic.

Everything runs without a real model: the decoding functions take a
`score_fn(prefix) -> scores over the vocabulary`, so any model (or a toy one)
can plug in.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

END = "<end>"


# --- chat templates ---------------------------------------------------------

def render_chatml(messages: list[dict], add_generation_prompt: bool = True) -> str:
    """Render messages in the ChatML layout used by several open model families.

    Each message becomes `<|im_start|>{role}\\n{content}<|im_end|>\\n`. With
    `add_generation_prompt`, an open assistant turn is appended: the model's
    job is just to continue the text from there.
    """
    out = []
    for m in messages:
        if m.get("role") not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unknown role: {m.get('role')!r}")
        out.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n")
    if add_generation_prompt:
        out.append("<|im_start|>assistant\n")
    return "".join(out)


# --- decoding ---------------------------------------------------------------

ScoreFn = Callable[[str], np.ndarray]


def greedy_decode(score_fn: ScoreFn, vocab: list[str], max_len: int = 40) -> str:
    """Unconstrained: always take the highest-scoring next token."""
    out = ""
    for _ in range(max_len):
        tok = vocab[int(np.argmax(score_fn(out)))]
        if tok == END:
            break
        out += tok
    return out


def allowed_next(prefix: str, options: list[str], vocab: list[str]) -> list[int]:
    """Indices of vocabulary tokens that keep `prefix` on track to one of `options`."""
    ok = []
    for i, tok in enumerate(vocab):
        if tok == END:
            if prefix in options:
                ok.append(i)
        elif any(o.startswith(prefix + tok) for o in options):
            ok.append(i)
    return ok


def constrained_decode(score_fn: ScoreFn, vocab: list[str], options: list[str]) -> str:
    """Greedy decoding that can only ever produce one of `options`.

    The model's scores are used as-is, except that tokens which would leave
    the set of valid outputs get a score of minus infinity.
    """
    if END not in vocab:
        raise ValueError(f"vocab must contain the end token {END!r}")
    if not options:
        raise ValueError("need at least one allowed output")
    out = ""
    while True:
        scores = np.asarray(score_fn(out), dtype=float)
        masked = np.full_like(scores, -np.inf)
        idx = allowed_next(out, options, vocab)
        masked[idx] = scores[idx]
        tok = vocab[int(np.argmax(masked))]
        if tok == END:
            return out
        out += tok


def toy_scorer(vocab: list[str], seed: int = 0, bias: dict[str, float] | None = None) -> ScoreFn:
    """A stand-in "model": fixed random scores that depend on the previous token.

    `bias` adds to the score of particular tokens, to mimic a model that
    leans toward some answer.
    """
    rng = np.random.default_rng(seed)
    table = rng.normal(size=(len(vocab) + 1, len(vocab)))
    extra = np.array([(bias or {}).get(t, 0.0) for t in vocab])
    index = {t: i for i, t in enumerate(vocab)}

    def score(prefix: str) -> np.ndarray:
        last = len(vocab)  # row for "start of output"
        for t in sorted(vocab, key=len, reverse=True):
            if t != END and prefix.endswith(t):
                last = index[t]
                break
        return table[last] + extra

    return score


# --- caching and budgets ----------------------------------------------------

def prefill_tokens(shared_prefix: int, new_per_request: int, requests: int, cached: bool) -> int:
    """Input tokens the model must process across `requests` calls that share a prefix.

    With prefix caching, the shared prefix is processed once and reused
    (assuming every later request is a cache hit).
    """
    if min(shared_prefix, new_per_request, requests) < 0:
        raise ValueError("token counts must be non-negative")
    if requests == 0:
        return 0
    if cached:
        return shared_prefix + new_per_request * requests
    return (shared_prefix + new_per_request) * requests


def remaining_context(window: int, *, system: int = 0, tools: int = 0, history: int = 0,
                      reserved_output: int = 0) -> int:
    """Tokens left for new content after everything else that must fit in the window."""
    left = window - system - tools - history - reserved_output
    if left < 0:
        raise ValueError(f"over budget by {-left:,} tokens")
    return left
