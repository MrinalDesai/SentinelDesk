# SentinelDesk

**An on-premise, agentic IT-support ticket system — classification, routing, grounded resolution, and honest escalation.**

SentinelDesk takes an IT-support ticket and decides which of seven technical domains it belongs to, routes it to the right team, suggests a resolution grounded in past ticket history, and escalates anything it isn't confident about. It runs fully on-premise: a local LLM (Mistral-7B via Ollama) is consulted on under 1% of tickets — the rest is decided by deterministic, interpretable components.

> **Thesis: Rules own safety. Retrieval owns grounding. The LLM owns reasoning.**

Domains: Infrastructure · Application · Security · Database · Storage · Network · Access Management

---

## Key Features

- **Deterministic safety gate** — high-stakes tickets (data breach, ransomware, outage) are caught by regex rules and escalated to the right team *before any model runs*.
- **Vocabulary-guided classification** — a linear SVM + a fully-auditable deterministic scorer over a two-layer term vocabulary, with synonym normalization for casual phrasing.
- **Confidence-gated agentic routing** — confident tickets route directly; ambiguous ones climb a voter ladder (scorer + SVM + kNN), then an LLM tiebreak, then human escalation.
- **Hybrid retrieval** — BM25 lexical ranking + BGE-M3 semantic search served from Qdrant + Graph-RAG resolution, fused via Reciprocal Rank Fusion.
- **Graph RAG** — symptom -> root cause -> quality-vetted resolution; escalates honestly when no clean fix exists.
- **Explainability** — every decision produces a grounded reasoning chain.
- **Self-correction** — learns new synonyms from human corrections, no retraining.
- **Security** — PII redaction, encryption at rest, and RBAC implemented; runs zero-cloud.

---

## Architecture

A ticket flows through five agents over one shared state, doing as little work as needed:

```
ticket -> [1 Safety gate] -> [2 Classify] -> [3 Route] -> [4 Resolve] -> [5 Judge] -> outcome
            (regex)           (SVM+scorer)    (ladder)     (Graph RAG)    (consistency)
```

Easy tickets resolve in milliseconds; only ambiguous ones pull in more machinery. The pipeline is also compiled as a LangGraph state machine, with a dependency-free fallback.

---

## Tech Stack

- **Language:** Python 3.11
- **ML / NLP:** scikit-learn (TF-IDF + linear SVM), NLTK, rank-bm25
- **LLM:** Mistral-7B-Instruct-Q8 via Ollama (local)
- **Semantic search:** sentence-transformers (BGE-M3) + Qdrant (embedded / in-memory; scales to a hosted cluster)
- **Graph RAG:** NetworkX
- **Orchestration:** LangGraph
- **Service / UI:** FastAPI + a browser console
- **Security:** cryptography (Fernet/AES)

---

## Installation

```bash
git clone https://github.com/MrinalDesai/SentinelDesk.git
cd SentinelDesk
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

# for the local LLM tiebreak layer:
ollama pull mistral:7b-instruct-q8_0
```

---

## Running

```bash
# Live console - run any ticket through every layer
python web/server.py            # -> http://127.0.0.1:8000

# Live-learning demo (held-out keyword/prompt optimization)
#                                 -> http://127.0.0.1:8000/learn

# Hybrid retrieval (BM25 + BGE-M3/Qdrant + Graph RAG, fused by RRF)
python hybrid/hybrid_retrieval.py --in data/real_3000.csv --fast

# Full system check (15/15)
python scripts/check_everything.py

# Accuracy on the real-style corpus
python scripts/eval_corpus.py --in data/real_3000.csv

# Security controls demo
python security/run_demo.py
```

---

## Project Structure

```
src/sentineldesk/   core pipeline (classifier, vocabulary, safety, rag, pipeline, learning)
scripts/            evaluation, training, and demo scripts
tests/              125 unit tests
web/                live console + server + showcase + live-learning tab
hybrid/             hybrid retrieval (BM25 + BGE-M3/Qdrant + Graph RAG + RRF)
semantic/           semantic edge-case fallback
optimizer/          adaptive keyword/prompt optimization (held-out evaluation)
security/           PII redaction, encryption, RBAC
external_test/      validation on real public (Zenodo) tickets
data/               corpora + trained model
```

---

## Validation

- **98%** accuracy on 2,996 real-style tickets (tiered routing -> 0.983).
- Tier split: confident SVM 92.8% @1.00 - voter ladder 6.8% @0.76 - LLM tiebreak 0.3% @1.00.
- **Real public tickets (Zenodo):** Network 99%, Access Management 75%, combined 89.2%.
- **Tests:** 125 unit - 15/15 functional system checks - 6 security tests.

Every figure is reproducible via the commands above.

---

## Limitations (stated honestly)

- Training and primary evaluation use **synthetic data** (LLM-generated from a seed lexicon), which is near-canonical; the 98% shows robustness to phrasing of in-domain concepts, and the Zenodo result is the real-generalization evidence on the readable subset.
- The system is **lexical-first by design** for interpretability; semantic search runs as a hybrid/fallback layer rather than the primary path.
- **Graph-RAG resolution coverage is ~67%** of root causes; the rest escalate to a human.
- Low-signal tickets can drift toward a Network/Access-Management bias; mitigated by the confidence gate (escalation), not eliminated.
- Security: PII redaction / encryption-at-rest / RBAC are implemented; Vault and mTLS are deployment-layer integrations; Presidio NER is on the roadmap.

---

## Author

Mrinal Desai - linkedin.com/in/mrinal-d-30093134

*Built for the AI-Code-Sarathi / NASSCOM Agentic AI Hackathon 2026.*
