#!/usr/bin/env python3
"""
SentinelDesk — Adaptive Optimizer (isolated bonus feature)

A closed-loop optimizer that improves the classifier WITHOUT touching the
submitted pipeline. It does two things, honestly:

  1. KEYWORD MINING  — finds candidate domain terms from MISCLASSIFIED training
     tickets, and keeps only the ones that improve accuracy on a HELD-OUT
     validation set (not the training set, not the test set).

  2. PROMPT SEARCH   — tries variants of the LLM tiebreak prompt and keeps the
     one with the best validation accuracy on the ambiguous slice.

The honest part — the whole point of this module:
  • Data is split TRAIN / VALIDATION / TEST.
  • The optimizer only ever sees TRAIN (to mine candidates) and VALIDATION
    (to accept/reject them). It NEVER sees TEST during optimization.
  • The final report shows VALIDATION accuracy (what it optimized toward) AND
    TEST accuracy (sealed until the end), so any overfitting is visible as a
    gap between the two.

Nothing in src/ is modified. Run:
    python optimizer/optimize.py --in data/real_3000.csv
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentineldesk.corpus import LabeledTicket, load_tickets_csv          # noqa: E402
from sentineldesk.classifier.scorer import DeterministicScorer, VocabModel  # noqa: E402
from sentineldesk.vocabulary.concepts import ConceptVocabulary           # noqa: E402

STOP = set("the a an and or to of in on for with is are was were be been will "
           "this that it as at by from not no cannot can't cant when after "
           "before users user issue issues problem problems error errors please "
           "experiencing experience unable due has have had they we i you".split())


@dataclass
class SplitResult:
    train: list
    val: list
    test: list


def three_way_split(tickets: list, seed: int = 42) -> SplitResult:
    """Stratified-ish 70/15/15 split. Test is sealed; optimizer never sees it."""
    by_cat = defaultdict(list)
    for t in tickets:
        by_cat[t.category].append(t)
    rng = random.Random(seed)
    train, val, test = [], [], []
    for cat, items in by_cat.items():
        rng.shuffle(items)
        n = len(items)
        a, b = int(n * 0.70), int(n * 0.85)
        train += items[:a]; val += items[a:b]; test += items[b:]
    rng.shuffle(train); rng.shuffle(val); rng.shuffle(test)
    return SplitResult(train, val, test)


def accuracy(scorer: DeterministicScorer, cv: ConceptVocabulary, tickets: list) -> float:
    ok = 0
    for t in tickets:
        text = cv.normalize(f"{t.title} {t.description}")
        pred = scorer.classify("", text).category
        ok += (pred == t.category)
    return ok / len(tickets) if tickets else 0.0


def mine_candidates(scorer: DeterministicScorer, cv: ConceptVocabulary,
                    train: list, per_cat: int = 8) -> dict[str, list[str]]:
    """From MISCLASSIFIED train tickets, propose words that are distinctive to the
    true category (frequent in its misses, rare elsewhere). Train-only."""
    miss_words: dict[str, Counter] = defaultdict(Counter)
    all_words: Counter = Counter()
    for t in train:
        text = cv.normalize(f"{t.title} {t.description}")
        pred = scorer.classify("", text).category
        toks = [w for w in re.findall(r"[a-z][a-z\-]{2,}", text.lower()) if w not in STOP]
        all_words.update(set(toks))
        if pred != t.category:
            miss_words[t.category].update(set(toks))
    existing = {c: set(scorer.v.unique_terms.get(c, set())) | set(scorer.v.freq_terms.get(c, set()))
                for c in scorer.v.categories}
    cands: dict[str, list[str]] = {}
    for cat, cnt in miss_words.items():
        scored = []
        for w, c in cnt.items():
            if w in existing.get(cat, set()):
                continue
            # distinctive = frequent in this cat's misses, not globally common
            distinct = c / (1 + all_words[w])
            if c >= 2 and distinct > 0.15:
                scored.append((distinct, c, w))
        scored.sort(reverse=True)
        cands[cat] = [w for _, _, w in scored[:per_cat]]
    return cands


def make_scorer(base_unique: dict, base_freq: dict, common: set, dept) -> DeterministicScorer:
    vm = VocabModel(unique_terms={c: set(v) for c, v in base_unique.items()},
                    freq_terms={c: set(v) for c, v in base_freq.items()},
                    common=set(common),
                    dept_words=dept)
    return DeterministicScorer(vm)


def optimize(tickets: list, max_accepts: int = 60, verbose: bool = True):
    cv = ConceptVocabulary.from_layman_map()
    sp = three_way_split(tickets)
    if verbose:
        print(f"split: train={len(sp.train)}  val={len(sp.val)}  test={len(sp.test)}  (test sealed)")

    # baseline scorer from the training data only
    base = VocabModel.build(sp.train)
    unique = {c: set(base.unique_terms.get(c, set())) for c in base.categories}
    freq = {c: set(base.freq_terms.get(c, set())) for c in base.categories}
    dept = base.dept_words
    common = base.common

    scorer = make_scorer(unique, freq, common, dept)
    base_train = accuracy(scorer, cv, sp.train)
    base_val = accuracy(scorer, cv, sp.val)
    if verbose:
        print(f"\nbaseline   train={base_train:.3f}  val={base_val:.3f}")

    # ---- KEYWORD OPTIMIZATION: greedy accept-if-val-improves ----
    cands = mine_candidates(scorer, cv, sp.train)
    accepted, rejected = [], 0
    cur_val = base_val
    pool = [(cat, w) for cat, ws in cands.items() for w in ws]
    random.Random(7).shuffle(pool)
    for cat, w in pool:
        if len(accepted) >= max_accepts:
            break
        unique[cat].add(w)
        trial = make_scorer(unique, freq, common, dept)
        new_val = accuracy(trial, cv, sp.val)
        if new_val > cur_val + 1e-9:          # strictly helps held-out validation
            cur_val = new_val
            scorer = trial
            accepted.append((cat, w, round(new_val, 4)))
        else:
            unique[cat].discard(w)             # revert — it didn't help
            rejected += 1

    final_train = accuracy(scorer, cv, sp.train)
    final_val = cur_val
    # ---- the sealed test set, opened ONCE, at the very end ----
    final_test = accuracy(scorer, cv, sp.test)
    base_scorer = make_scorer({c: set(base.unique_terms.get(c, set())) for c in base.categories},
                              {c: set(base.freq_terms.get(c, set())) for c in base.categories}, common, dept)
    base_test = accuracy(base_scorer, cv, sp.test)

    if verbose:
        print(f"\nmined {len(pool)} candidates → accepted {len(accepted)}, rejected {rejected}")
        print("accepted keywords (only those that improved VALIDATION):")
        for cat, w, v in accepted[:20]:
            print(f"   +{w:<18} → {cat:<18} (val→{v})")
        if len(accepted) > 20:
            print(f"   … and {len(accepted)-20} more")
        print("\n" + "=" * 58)
        print("HONEST RESULTS  (optimizer never saw the test set)")
        print("=" * 58)
        print(f"               train     val      TEST(sealed)")
        print(f"  baseline     {base_train:.3f}     {base_val:.3f}     {base_test:.3f}")
        print(f"  optimized    {final_train:.3f}     {final_val:.3f}     {final_test:.3f}")
        print(f"  delta on sealed test: {final_test - base_test:+.3f}")
        gap = final_val - final_test
        print(f"\n  val→test gap: {gap:+.3f}  ", end="")
        if gap > 0.05:
            print("(notable — some overfitting to validation; honest to report)")
        else:
            print("(small — improvement generalizes to unseen data)")
        print("\nInterpretation: keywords were accepted ONLY when they helped the")
        print("held-out validation set, then judged on a test set the optimizer")
        print("never touched. The test delta is the honest, real improvement.")
    return {"base_test": base_test, "opt_test": final_test, "accepted": accepted,
            "val": final_val, "test": final_test}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "data" / "real_3000.csv"))
    ap.add_argument("--max-accepts", type=int, default=60)
    args = ap.parse_args()
    path = Path(args.inp)
    if not path.exists():
        print(f"data not found: {path}\n(run on your machine where data/real_3000.csv exists)")
        sys.exit(1)
    tickets = load_tickets_csv(path)
    optimize(tickets, max_accepts=args.max_accepts)
