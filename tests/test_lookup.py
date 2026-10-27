"""S5 (02/10) : brouillon « prompt lookup » et simulation de l'acceptation (scripts/measure_lookup.py)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from measure_lookup import lookup_draft, simulate  # noqa: E402


def test_lookup_draft_finds_latest_ngram():
    seq = [1, 2, 3, 9, 1, 2, 3, 7, 8, 5, 2, 3]
    assert lookup_draft(seq, 2) == [7, 8]  # dernière occurrence de (2, 3) avant la fin
    assert lookup_draft([4, 5, 6], 3) == []


def test_simulate_counts_steps():
    prompt = [1, 2, 3, 4]
    gen = [1, 2, 3, 4, 1, 2]  # répète le prompt : le brouillon devine presque tout
    steps, toks, acc = simulate(prompt, gen, 4)
    assert toks == 6 and steps < 6 and acc == toks - steps
    steps, toks, acc = simulate([10, 11], [20, 21, 22], 4)  # rien à copier
    assert steps == 3 and acc == 0
