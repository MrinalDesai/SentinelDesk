r"""
Agentic retraining loop — a LangGraph-orchestrated, data-driven improvement pipeline.

The agents ASSESS the data with a battery of diagnostic tests, map each detected
problem to a concrete remedy (the problem -> test -> remedy table below), apply the
remedies, retrain, and only keep the new model if it validates better.

The graph takes DIFFERENT PATHS depending on what the tests find (a real LangGraph
conditional edge after `plan`):

    diagnose -> plan --(problems found)--> retrain -> ground -> validate -> judge
                     \--(data healthy)----------------------------------> judge

  - diagnose : run every test in CHECKS against the data; train a baseline SVM
  - plan     : collect the remedies for whichever tests tripped
  - (route)  : if any remedy is needed -> retrain path; else -> straight to judge
  - retrain  : apply remedies (SMOTE / stopwords / bigrams / dedupe) and retrain
  - ground   : RAG-style retrieval of nearest past tickets as evidence
  - validate : held-out metrics, candidate vs baseline
  - judge    : accept only if it improves; write candidate; or report "no change"

Isolated: reads data read-only, writes ONLY to improvement/out/, never touches
data/svm_model.pkl, src/, or any live route.

Run:  python improvement/agentic_retrain.py            (induced hard case)
      python improvement/agentic_retrain.py balanced   (healthy -> skip path)
Deps: scikit-learn, imbalanced-learn (SMOTE), langgraph
"""
from __future__ import annotations

import csv
import json
import pickle
import random
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from langgraph.graph import END, StateGraph

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
DATA = ROOT / "data" / "real_3000.csv"          # read-only source
SEED = 42

MAJORITY = ["Network", "Access Management"]
TEST_PER = 40
MAJ_TRAIN, MIN_TRAIN, BAL_TRAIN = 300, 6, 150

AGENT_FILES = ["01_diagnose", "02_plan", "03_retrain", "04_ground_rag", "05_validate", "06_decision"]

# ---- the problem -> test -> remedy table the agents work through ----------------
CHECKS = [
    {"id": "imbalance",  "problem": "Class imbalance",
     "test": "max/min class-size ratio", "threshold": ">= 3:1",
     "remedy": "SMOTE oversampling of minority classes"},
    {"id": "stopwords",  "problem": "Noisy generic tokens",
     "test": "stopword fraction of all tokens", "threshold": "> 0.50",
     "remedy": "Remove English stopwords"},
    {"id": "shorttext",  "problem": "Short / low-signal text",
     "test": "median tokens per ticket", "threshold": "< 12",
     "remedy": "Add bigrams (1-2 grams)"},
    {"id": "duplicates", "problem": "Duplicate tickets",
     "test": "exact-duplicate count", "threshold": "> 0",
     "remedy": "Deduplicate before training"},
    {"id": "rare",       "problem": "Class too rare to synthesize",
     "test": "smallest class size", "threshold": "< 5",
     "remedy": "Flag for data collection (cannot SMOTE safely)"},
]


class S(TypedDict, total=False):
    induce: bool
    Xtr: list; ytr: list; Xte: list; yte: list
    diagnosis: dict; plan: dict; baseline: dict
    candidate_model: Any; candidate: dict; rag: dict; decision: dict
    reasoning: list


def _write(name: str, obj: dict) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _toks(t: str) -> list:
    return re.findall(r"[a-z]+", t.lower())


def _load_split(induce: bool):
    rows = list(csv.DictReader(open(DATA, encoding="utf-8")))
    by: dict[str, list] = {}
    for r in rows:
        cat = r.get("category", "")
        if not cat:
            continue
        # induced hard case uses terse titles; healthy case uses full text
        text = (r.get("title", "") if induce
                else f"{r.get('title','')} {r.get('description','')}").strip()
        by.setdefault(cat, []).append(text)
    rng = random.Random(SEED)
    Xtr, ytr, Xte, yte = [], [], [], []
    for cat, texts in by.items():
        rng.shuffle(texts)
        Xte += texts[:TEST_PER]; yte += [cat] * TEST_PER
        keep = (MAJ_TRAIN if cat in MAJORITY else MIN_TRAIN) if induce else BAL_TRAIN
        tr = texts[TEST_PER:TEST_PER + keep]
        Xtr += tr; ytr += [cat] * len(tr)
    return Xtr, ytr, Xte, yte


def _metrics(model, Xte, yte) -> dict:
    pred = model.predict(Xte); labels = sorted(set(yte))
    rec = recall_score(yte, pred, labels=labels, average=None, zero_division=0)
    return {"accuracy": round(float(accuracy_score(yte, pred)), 3),
            "macro_f1": round(float(f1_score(yte, pred, average="macro", zero_division=0)), 3),
            "min_class_recall": round(float(min(rec)), 3),
            "per_class_recall": {c: round(float(v), 3) for c, v in zip(labels, rec)}}


def _run_checks(Xtr, ytr) -> list:
    counts = Counter(ytr)
    ratio = max(counts.values()) / max(min(counts.values()), 1)
    all_toks = [w for t in Xtr for w in _toks(t)]
    sw_frac = sum(1 for w in all_toks if w in ENGLISH_STOP_WORDS) / max(len(all_toks), 1)
    med_tokens = statistics.median(len(_toks(t)) for t in Xtr)
    dup_count = len(Xtr) - len(set(Xtr))
    min_class = min(counts.values())
    measured = {
        "imbalance":  (f"{ratio:.1f}:1", ratio >= 3),
        "stopwords":  (f"{sw_frac:.2f}", sw_frac > 0.50),
        "shorttext":  (f"{med_tokens:.0f} tokens", med_tokens < 12),
        "duplicates": (f"{dup_count}", dup_count > 0),
        "rare":       (f"{min_class}", min_class < 5),
    }
    return [{**c, "value": measured[c["id"]][0], "triggered": bool(measured[c["id"]][1])}
            for c in CHECKS]


# ---- agents (LangGraph nodes) -------------------------------------------------

def diagnose(state: S) -> dict:
    induce = state.get("induce", True)
    Xtr, ytr, Xte, yte = _load_split(induce)
    table = _run_checks(Xtr, ytr)
    base = make_pipeline(TfidfVectorizer(max_features=300), LinearSVC(random_state=SEED)).fit(Xtr, ytr)
    bm = _metrics(base, Xte, yte)
    tripped = [c["problem"] for c in table if c["triggered"]]
    diagnosis = {"mode": "induced imbalance (titles)" if induce else "balanced / healthy (full text)",
                 "n_train": len(Xtr), "n_test": len(Xte),
                 "class_counts": dict(Counter(ytr)),
                 "checks": table, "problems_found": tripped,
                 "baseline_metrics": bm}
    _write("01_diagnose.json", diagnosis)
    return {"Xtr": Xtr, "ytr": ytr, "Xte": Xte, "yte": yte,
            "diagnosis": diagnosis, "baseline": bm,
            "reasoning": [f"diagnose: ran {len(table)} tests; problems: {tripped or 'none'}; "
                          f"baseline macro-F1={bm['macro_f1']}, min-recall={bm['min_class_recall']}"]}


def plan(state: S) -> dict:
    remedies = [{"problem": c["problem"], "remedy": c["remedy"], "tool": c["id"]}
                for c in state["diagnosis"]["checks"] if c["triggered"]]
    plan = {"remedies": remedies, "tools": [r["tool"] for r in remedies]}
    _write("02_plan.json", plan)
    return {"plan": plan,
            "reasoning": state["reasoning"] + [
                f"plan: remedies -> {[r['remedy'] for r in remedies] or 'none needed'}"]}


def route_after_plan(state: S) -> str:
    # the real conditional edge: branch on whether any remedy is needed
    return "retrain" if state["plan"]["tools"] else "skip"


def retrain(state: S) -> dict:
    tools = set(state["plan"]["tools"])
    Xtr, ytr = state["Xtr"], state["ytr"]
    if "duplicates" in tools:
        seen = set(); X2 = []; y2 = []
        for x, y in zip(Xtr, ytr):
            if x not in seen:
                seen.add(x); X2.append(x); y2.append(y)
        Xtr, ytr = X2, y2
    vec = TfidfVectorizer(
        stop_words="english" if "stopwords" in tools else None,
        ngram_range=(1, 2) if "shorttext" in tools else (1, 1),
        max_features=600)
    steps = [("tfidf", vec)]
    smote_k = None
    if "imbalance" in tools and "rare" not in tools:
        smote_k = max(1, min(5, min(Counter(ytr).values()) - 1))
        steps.append(("smote", SMOTE(k_neighbors=smote_k, random_state=SEED)))
    steps.append(("svm", LinearSVC(random_state=SEED)))
    t0 = time.time()
    model = ImbPipeline(steps).fit(Xtr, ytr)
    info = {"remedies_applied": sorted(tools), "smote_k_neighbors": smote_k,
            "train_rows_used": len(Xtr), "fit_seconds": round(time.time() - t0, 2),
            "n_features": int(model.named_steps["tfidf"].idf_.shape[0])}
    _write("03_retrain.json", info)
    return {"candidate_model": model,
            "reasoning": state["reasoning"] + [
                f"retrain: applied {sorted(tools)}"
                + (f" (SMOTE k={smote_k})" if smote_k else "")
                + f", fit {info['fit_seconds']}s, {info['n_features']} features"]}


def ground(state: S) -> dict:
    vec = TfidfVectorizer(stop_words="english")
    Xv = vec.fit_transform(state["Xtr"])
    nn = NearestNeighbors(n_neighbors=3, metric="cosine").fit(Xv)
    rng = random.Random(SEED)
    examples = []
    for i in rng.sample(range(len(state["Xte"])), 3):
        q = state["Xte"][i]
        dist, idx = nn.kneighbors(vec.transform([q]))
        neigh = [{"text": state["Xtr"][j][:90], "domain": state["ytr"][j],
                  "similarity": round(1 - float(d), 3)} for d, j in zip(dist[0], idx[0])]
        votes = Counter(n["domain"] for n in neigh)
        examples.append({"query": q[:90], "true_domain": state["yte"][i],
                         "retrieved": neigh, "evidence_domain": votes.most_common(1)[0][0]})
    rag = {"method": "TF-IDF cosine kNN (k=3)", "examples": examples}
    _write("04_ground_rag.json", rag)
    return {"rag": rag,
            "reasoning": state["reasoning"] + [f"ground: retrieved evidence for {len(examples)} tickets"]}


def validate(state: S) -> dict:
    cm = _metrics(state["candidate_model"], state["Xte"], state["yte"]); bm = state["baseline"]
    deltas = {k: round(cm[k] - bm[k], 3) for k in ("accuracy", "macro_f1", "min_class_recall")}
    val = {"baseline": bm, "candidate": cm, "deltas": deltas}
    _write("05_validate.json", val)
    return {"candidate": val,
            "reasoning": state["reasoning"] + [
                f"validate: macro-F1 {bm['macro_f1']}->{cm['macro_f1']} (d{deltas['macro_f1']:+}), "
                f"min-recall {bm['min_class_recall']}->{cm['min_class_recall']} (d{deltas['min_class_recall']:+})"]}


def judge(state: S) -> dict:
    if "candidate" not in state:                       # arrived via the skip edge
        decision = {"accept": False, "path": "skip",
                    "reason": "no problems detected by the tests — model left unchanged",
                    "deltas": {"accuracy": 0.0, "macro_f1": 0.0, "min_class_recall": 0.0},
                    "model_written": None}
        _write("06_decision.json", decision)
        _write("report.json", {"diagnosis": state["diagnosis"], "plan": state["plan"],
                               "decision": decision,
                               "reasoning_trace": state["reasoning"] + ["judge: no remediation needed (skip path)"]})
        return {"decision": decision,
                "reasoning": state["reasoning"] + ["judge: no remediation needed (skip path)"]}
    d = state["candidate"]["deltas"]
    accept = d["macro_f1"] > 0 or d["min_class_recall"] > 0
    saved = None
    if accept:
        saved = str(OUT / "svm_candidate.pkl")
        with open(saved, "wb") as fh:
            pickle.dump(state["candidate_model"], fh)
    decision = {"accept": accept, "path": "retrain",
                "reason": ("remedies improved held-out macro-F1 / minority recall -> accept"
                           if accept else "remedies gave no held-out gain -> reject, keep baseline"),
                "deltas": d, "model_written": saved}
    _write("06_decision.json", decision)
    _write("report.json", {"diagnosis": state["diagnosis"], "plan": state["plan"],
                           "validation": state["candidate"], "decision": decision,
                           "reasoning_trace": state["reasoning"] + [f"judge: {'ACCEPT' if accept else 'REJECT'}"]})
    return {"decision": decision, "reasoning": state["reasoning"] + [f"judge: {'ACCEPT' if accept else 'REJECT'}"]}


def build_graph():
    g = StateGraph(S)
    for name, fn in [("diagnose", diagnose), ("plan", plan), ("retrain", retrain),
                     ("ground", ground), ("validate", validate), ("judge", judge)]:
        g.add_node(name, fn)
    g.set_entry_point("diagnose")
    g.add_edge("diagnose", "plan")
    g.add_conditional_edges("plan", route_after_plan, {"retrain": "retrain", "skip": "judge"})
    g.add_edge("retrain", "ground")
    g.add_edge("ground", "validate")
    g.add_edge("validate", "judge")
    g.add_edge("judge", END)
    return g.compile()


def run_and_collect(induce: bool = True) -> dict:
    for f in AGENT_FILES:                               # clear stale outputs (skip path writes fewer files)
        (OUT / f"{f}.json").unlink(missing_ok=True)
    graph = build_graph()
    try:
        (OUT / "graph.mmd").write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    except Exception:
        pass
    final = graph.invoke({"induce": induce})
    out: dict = {"reasoning": final.get("reasoning", []),
                 "path": final.get("decision", {}).get("path", "retrain")}
    for f in AGENT_FILES:
        p = OUT / f"{f}.json"
        if p.exists():
            out[f] = json.loads(p.read_text(encoding="utf-8"))
    return out


def main():
    induce = not (len(sys.argv) > 1 and sys.argv[1].lower().startswith("bal"))
    graph = build_graph()
    try:
        (OUT / "graph.mmd").write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
    except Exception:
        pass
    print("=" * 70)
    print(f"AGENTIC RETRAINING LOOP  (LangGraph)   mode={'induced imbalance' if induce else 'balanced/healthy'}")
    print("=" * 70)
    final = graph.invoke({"induce": induce})
    print("\nproblem -> test -> remedy:")
    for c in final["diagnosis"]["checks"]:
        mark = "TRIPPED" if c["triggered"] else "  ok   "
        print(f"  [{mark}] {c['problem']:<28} {c['test']:<26} {c['value']:>10}  -> {c['remedy'] if c['triggered'] else '-'}")
    print(f"\npath taken: {final['decision'].get('path','retrain').upper()}")
    print("reasoning trace:")
    for s in final["reasoning"]:
        print("  -", s)
    dec = final["decision"]
    print(f"\ndecision: {'ACCEPT' if dec['accept'] else 'REJECT / no-change'} | deltas {dec['deltas']}")
    print(f"outputs in: {OUT}")


if __name__ == "__main__":
    main()
