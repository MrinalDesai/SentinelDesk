"""
Explanation generator.

Reads a ScoreResult back into plain language: which team won, which words
drove it, how close the runner-up was, and — for edge cases — why the
deterministic path wasn't confident enough and what happens next. Because the
scorer records every term that fired, the explanation is fully grounded: every
claim points at a real matched term, nothing is invented.
"""

from __future__ import annotations

from .scorer import ScoreResult


def _join(terms: list[str]) -> str:
    terms = [t for t in terms if t]
    if not terms:
        return ""
    if len(terms) == 1:
        return f"'{terms[0]}'"
    return ", ".join(f"'{t}'" for t in terms[:-1]) + f" and '{terms[-1]}'"


def explain(result: ScoreResult) -> str:
    m = result.matched
    reasons: list[str] = []

    if m["dept"]:
        reasons.append(f"the ticket names the team directly ({_join(m['dept'])})")
    if m["unique"]:
        reasons.append(f"it uses {result.category}-specific terms {_join(m['unique'])}")
    if m["freq"]:
        reasons.append(f"common {result.category} terms {_join(m['freq'])}")
    if m["layman"]:
        reasons.append(f"casual phrasing {_join(m['layman'])} maps to {result.category}")

    if result.is_edge_case:
        lead = f"Uncertain — best guess is {result.category}"
        body = f"but {result.edge_reason}."
        nxt = " Sending to the edge-case resolver (root-cause check, history vote, then a human if still unclear)."
        if reasons:
            why = " The signal that did fire: " + "; ".join(reasons) + "."
        else:
            why = ""
        return f"{lead}, {body}{why}{nxt}"

    lead = f"Routed to {result.category} (confidence {result.confidence:.0%})."
    why = " Because " + "; ".join(reasons) + "." if reasons else ""
    runner = ""
    if result.runner_up and result.scores.get(result.runner_up, 0) > 0:
        runner = (
            f" Runner-up was {result.runner_up} "
            f"(score {result.scores[result.category]:g} vs "
            f"{result.scores[result.runner_up]:g})."
        )
    return f"{lead}{why}{runner}"
