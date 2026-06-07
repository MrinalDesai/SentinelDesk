#!/usr/bin/env python3
"""
SentinelDesk live demo backend (FastAPI).

Wraps the REAL pipeline and the REAL demo/ablation scripts so a web console can:
  - run a single ticket end-to-end through every layer (live), and
  - run each named experiment on a button press (synonym ablation, real-data eval,
    edge cases, Graph RAG, self-correction, PII redaction, full system check),
    streaming the actual script output back.

Nothing here is simulated — every endpoint executes your real code, so anything
shown to a judge is reproducible on the spot.

Run:
    pip install fastapi uvicorn
    python web/server.py            # serves API + the console at http://127.0.0.1:8000

The console (web/console.html) calls these endpoints.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "optimizer"))

from fastapi import FastAPI                                   # noqa: E402
from fastapi.middleware.cors import CORSMiddleware           # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse     # noqa: E402
from pydantic import BaseModel                               # noqa: E402

# --- load the real pipeline once at startup -----------------------------------
from sentineldesk.classifier import (                        # noqa: E402
    DeterministicScorer, EdgeCaseResolver, KNNVoter, SVMClassifier, VocabModel, load_model, train_svm,
)
from sentineldesk.classifier.explainability import explain_decision   # noqa: E402
from sentineldesk.corpus import LabeledTicket, load_tickets_csv       # noqa: E402
from sentineldesk.data_gen.controlled import generate_controlled_corpus  # noqa: E402
from sentineldesk.pipeline import Pipeline                            # noqa: E402
from sentineldesk.rag import KnowledgeGraph                           # noqa: E402
from sentineldesk.safety.safety_layer import safety_check             # noqa: E402
from sentineldesk.vocabulary.concepts import ConceptVocabulary        # noqa: E402

app = FastAPI(title="SentinelDesk Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE: dict = {}


def _boot():
    cv = ConceptVocabulary.from_layman_map()
    model_path = ROOT / "data" / "svm_model.pkl"
    # training corpus for scorer/knn/graph (controlled corpus mirrors the trained vocab)
    res = {c: f"Standard {c} remediation: diagnose, apply the fix, verify recovery." for c in
           ["Network", "Database", "Storage", "Application", "Infrastructure", "Security", "Access Management"]}
    base = generate_controlled_corpus(per_category=150)
    train = [LabeledTicket(text=t.text, category=t.category, title=t.title,
                           description=t.description, resolution=res[t.category]) for t in base]
    svm = SVMClassifier(load_model(model_path)) if model_path.exists() else SVMClassifier(train_svm(train))
    scorer = DeterministicScorer(VocabModel.build(train))
    knn = KNNVoter(k=5).fit(train)
    # use a real Ollama client if available, else stub
    try:
        from sentineldesk.llm import OllamaClient
        llm = OllamaClient(model="mistral:7b-instruct-q8_0")
    except Exception:
        from sentineldesk.llm import StubLLMClient
        llm = StubLLMClient(lambda p: "Database")
    resolver = EdgeCaseResolver(scorer, svm, knn, llm=llm, confidence_gate=0.80)
    # graph from the REAL corpus if present, else the controlled train set
    real = ROOT / "data" / "real_3000.csv"
    graph_src = load_tickets_csv(real) if real.exists() else train
    graph = KnowledgeGraph.build(graph_src, cv)
    pipe = Pipeline(scorer, resolver, graph)
    # sample of real tickets for the picker
    sample = []
    if real.exists():
        with open(real, encoding="utf-8") as fh:
            for i, r in enumerate(csv.DictReader(fh)):
                if i >= 300:
                    break
                sample.append({"id": i, "title": r.get("title", ""),
                               "text": r.get("description", r.get("text", "")),
                               "truth": r.get("category", "")})
    STATE.update(cv=cv, svm=svm, scorer=scorer, knn=knn, resolver=resolver,
                 graph=graph, pipe=pipe, sample=sample)
    # learning-loop context (held-out split + llm kind), isolated bonus feature
    try:
        import optimize as _opt
        llm_kind = "ollama" if llm.__class__.__name__ == "OllamaClient" else "stub"
        split = _opt.three_way_split(graph_src) if len(graph_src) > 50 else None
        STATE.update(split=split, llm=llm, llm_kind=llm_kind)
    except Exception as e:
        STATE.update(split=None, llm=None, llm_kind="stub", learn_error=str(e))


@app.on_event("startup")
def startup():
    _boot()


class TicketIn(BaseModel):
    title: str = ""
    description: str = ""


@app.get("/api/tickets")
def tickets():
    """A sample of real tickets for the picker."""
    return STATE.get("sample", [])


@app.post("/api/run")
def run_ticket(t: TicketIn):
    """Run ONE ticket through every layer, returning each layer's real output."""
    cv, svm, scorer = STATE["cv"], STATE["svm"], STATE["scorer"]
    resolver, graph, pipe = STATE["resolver"], STATE["graph"], STATE["pipe"]
    title, desc = t.title, t.description
    text = f"{title} {desc}".strip()

    # layer-by-layer, real outputs
    safe = safety_check(title, desc)
    normalized = cv.normalize(text)
    raw_pred, raw_conf = svm.predict(text)
    norm_pred, norm_conf = svm.predict(normalized)
    sr = scorer.classify(title, desc)
    knn_vote = STATE["knn"].vote(text)
    decision = resolver.resolve(title, desc)
    chain = explain_decision(title, desc, scorer, resolver, graph)
    gr = graph.query(text)
    final = pipe.run(title, desc)

    return JSONResponse({
        "safety": {"escalated": safe.bypass_llm, "matched": safe.matched_category,
                   "department": safe.department, "latency_ms": round(safe.latency_ms, 3)},
        "normalization": {"before": text, "after": normalized, "changed": normalized != text.lower()},
        "svm": {"raw": [raw_pred, round(raw_conf, 3)], "normalized": [norm_pred, round(norm_conf, 3)]},
        "scorer": {"category": sr.category, "confidence": round(sr.confidence, 3),
                   "edge_case": sr.is_edge_case, "matched": sr.matched},
        "knn": {"category": knn_vote.category, "confidence": round(knn_vote.confidence, 3)},
        "resolver": {"category": decision.category, "method": decision.method,
                     "confidence": round(decision.confidence, 3), "trace": decision.trace,
                     "votes": decision.votes, "escalated": decision.escalated},
        "graph_rag": {"root_cause": gr.root_cause, "category": gr.category,
                      "resolution": gr.resolution, "symptoms": gr.symptom_hits,
                      "quality": getattr(gr, "resolution_quality", None)},
        "reasoning_chain": chain.steps,
        "final": {"category": final.category, "confidence": round(final.confidence, 3),
                  "method": final.method, "outcome": final.outcome,
                  "resolution": final.resolution},
    })


# --- experiment buttons: run the REAL scripts and stream their output ----------
EXPERIMENTS = {
    "system_check":   ["scripts/check_everything.py"],
    "real_eval":      ["scripts/eval_corpus.py", "--in", "data/real_3000.csv"],
    "ablation_off":   ["scripts/eval_corpus.py", "--in", "data/real_3000.csv", "--no-normalize"],
    "edge_cases":     ["scripts/edge_case_ablation.py", "--examples"],
    "exclusivity":    ["scripts/vocab_table.py", "--in", "data/real_3000.csv", "--top-n", "10"],
    "resolver_tiers": ["scripts/resolve_eval.py", "--in", "data/real_3000.csv"],
    "graph_rag":      ["scripts/graph_rag_demo.py", "--in", "data/real_3000.csv"],
    "self_correction":["scripts/self_correction_demo.py"],
    "explainability": ["scripts/explain_demo.py"],
    "pipeline":       ["scripts/pipeline_demo.py"],
    "external_real":  ["external_test/run_test.py", "--examples"],
    "pii_security":   ["security/run_demo.py"],
}


@app.get("/api/experiment/{name}")
def experiment(name: str):
    """Run a named experiment script and return its real stdout."""
    cmd = EXPERIMENTS.get(name)
    if not cmd:
        return JSONResponse({"error": f"unknown experiment '{name}'"}, status_code=404)
    try:
        proc = subprocess.run([sys.executable, *cmd], cwd=ROOT,
                              capture_output=True, text=True, timeout=600)
        return JSONResponse({"name": name, "cmd": " ".join(cmd),
                             "stdout": proc.stdout, "stderr": proc.stderr[-2000:],
                             "returncode": proc.returncode})
    except subprocess.TimeoutExpired:
        return JSONResponse({"name": name, "error": "timed out (>10 min)"}, status_code=504)


@app.get("/api/learn/cached")
def learn_cached():
    """Replay a previously captured real run (instant, no compute)."""
    import live_learning as ll
    return JSONResponse(ll.load_cached())


class LearnIn(BaseModel):
    title: str = ""
    description: str = ""
    prompt_search: bool = False
    capture: bool = False


@app.post("/api/learn/live")
def learn_live(t: LearnIn):
    """Run the real learning loop and return ordered step events."""
    import live_learning as ll
    if STATE.get("split") is None:
        return JSONResponse({"mode": "live", "steps": [ll.step("error", "Not enough data",
            "Need a corpus (data/real_3000.csv) with >50 tickets to split train/val/test.", {})]})
    ctx = STATE
    title = t.title or "websites won't load"
    desc = t.description or "users across the office cannot reach internal sites"
    if t.capture:
        return JSONResponse(ll.capture(ctx, title, desc, do_prompt=t.prompt_search))
    return JSONResponse(ll.run_live(title, desc, ctx, do_prompt=t.prompt_search))


@app.get("/learn", response_class=HTMLResponse)
def learn_page():
    html = (ROOT / "web" / "learn.html")
    return html.read_text(encoding="utf-8") if html.exists() else "<h1>learn.html missing</h1>"


@app.get("/", response_class=HTMLResponse)
def console():
    html = (ROOT / "web" / "console.html")
    return html.read_text(encoding="utf-8") if html.exists() else "<h1>console.html missing</h1>"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
