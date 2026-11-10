"""Smoke-test every notebook by running its code cells.

Notebooks that train read the S2A_STEPS environment variable, so this run
stays short. Notebooks that need PyTorch are skipped when it isn't installed.
"""

import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted(p for d in ("labs", "calculators") for p in (ROOT / d).glob("*.ipynb"))
MAGIC = re.compile(r"^(\s*)[%!].*$", re.MULTILINE)   # %pip / !pip lines only run in Jupyter


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(path, monkeypatch):
    nb = json.loads(path.read_text())
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    if any("torch" in c for c in code) and importlib.util.find_spec("torch") is None:
        pytest.skip("needs PyTorch")
    pytest.importorskip("matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    monkeypatch.setenv("S2A_STEPS", "20")
    monkeypatch.chdir(path.parent)
    first = nb["cells"][0]
    assert "colab.research.google.com" in "".join(first["source"]), "must start with an Open in Colab badge"
    ns: dict = {"__name__": "__main__"}
    for i, src in enumerate(code):
        try:
            exec(compile(MAGIC.sub(r"\1raise", src), f"{path.name}[cell {i}]", "exec"), ns)
        except Exception as e:  # pragma: no cover
            pytest.fail(f"{path.name} code cell {i} failed: {type(e).__name__}: {e}")
    assert os.environ["S2A_STEPS"] == "20"
