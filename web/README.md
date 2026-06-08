# SentinelDesk

**An on-premise, agentic IT-support system — classification, routing, grounded resolution, and honest escalation.**

SentinelDesk takes an IT-support ticket and decides which of seven technical domains it belongs to, routes it to the right team, suggests a resolution grounded in past ticket history, and escalates anything it isn't confident about. It runs fully on-premise: a local LLM (Mistral-7B via Ollama) is consulted on under 1% of tickets — the rest is decided by deterministic, interpretable components.

> **Thesis: Deterministic where it counts, semantic where it helps, the LLM only when nothing else can decide.**

Escalation ladder: Easy → rules · Hard → ML · Edge cases → semantic · Rare ties → LLM · High risk → human

Domains: Infrastructure · Application · Security · Database · Storage · Network · Access Management

---

## Architecture

Five agents over a shared state (orchestrated as a LangGraph state machine), doing as little work as the difficulty requires:

```
ticket -> [1 Safety] -> [2 Classify] -> [3 Route] -> [4 Resolve] -> [5 Judge] -> outcome
            regex         SVM+scorer      ladder       hybrid          consistency
                                                       retrieval        guard
```

1. **Safety gate** — regex catches high-stakes tickets (breach, ransomware, outage) and escalates to the right team before any model runs.
2. **Classify** — synonym normalization, then a linear SVM + a deterministic scorer over a two-layer TF-IDF / n-gram vocabulary.
3. **Route** — confidence-gated: confident tickets route directly; ambiguous ones climb a voter ladder (scorer + SVM + kNN), then a semantic fallback, then an LLM tiebreak, then a human.
4. **Resolve** — hybrid retrieval: BM25 + BGE-M3 semantic search (Qdrant) + Graph RAG, fused by Reciprocal Rank Fusion.
5. **Judge** — auto-resolve only if confident and the routing domain matches the resolution's domain; else escalate.

The deterministic classification core (SVM + TF-IDF + scorer + kNN) is the primary path; hybrid retrieval and the semantic fallback are additive layers for grounding and edge cases.

---

## Key Features

- **Deterministic safety gate** — high-stakes tickets escalated by rule before any model runs.
- **Vocabulary-guided classification** — SVM + auditable scorer over frequent and unique TF-IDF / n-gram terms, with synonym normalization for casual phrasing.
- **Confidence-gated agentic routing** — a reason-act-decide loop that escalates instead of guessing.
- **Hybrid retrieval** — BM25 + BGE-M3/Qdrant semantic + Graph RAG, fused via RRF.
- **Semantic edge-case fallback** — embeddings used precisely where lexical methods break.
- **Self-correction** — learns a new synonym from a human's reroute (no retraining), guarded and audited.
- **Explainability** — every decision emits a grounded reasoning chain.
- **Audit log** — every routing decision recorded (timestamp, agent decisions, confidence).
- **Security** — PII redaction, encryption at rest, RBAC; runs zero-cloud.

---

## Demo (web)

```bash
python web/server.py        # -> http://127.0.0.1:8000
```

Seven connected tabs:

- `/` **Console** — run a ticket through every layer; an agent-flow strip shows which of the five agents fired or was skipped.
- `/explain` **Explain** — watch the query light up term by term, the scorer decide, and (on ties) the LLM tiebreak prompt assemble; includes a synonym-compare mode (classify with vs without normalization).
- `/hybrid` **Hybrid** — BM25, semantic (BGE-M3/Qdrant), and Graph RAG ranking side by side, fused by RRF.
- `/vocab` **Vocabulary** — searchable per-domain TF-IDF and n-gram terms.
- `/learn` **Live Learning** — the held-out keyword/prompt optimizer loop.
- `/audit` **Audit Log** — the live decision trail.
- `/correct` **Self-Correction** — a misrouted ticket, a human correction, and the same ticket routed correctly afterward (no retraining).

---

## Tech Stack

Python 3.11 · scikit-learn (TF-IDF + linear SVM) · NLTK · rank-bm25 · Mistral-7B-Instruct-Q8 via Ollama · sentence-transformers (BGE-M3) · Qdrant (embedded / in-memory; scales to a hosted cluster) · NetworkX (Graph RAG) · LangGraph · FastAPI · cryptography.

---

## Installation

```bash
git clone https://github.com/MrinalDesai/SentinelDesk.git
cd SentinelDesk
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
ollama pull mistral:7b-instruct-q8_0    # for the local LLM tiebreak layer
```

---

## Running

```bash
python web/server.py                                   # live demo (all tabs)
python hybrid/hybrid_retrieval.py --in data/real_3000.csv --fast   # hybrid retrieval
python scripts/check_everything.py                     # full system check (15/15)
python scripts/eval_corpus.py --in data/real_3000.csv  # accuracy on the corpus
python security/run_demo.py                            # security controls
```

---

## Project Structure

```
src/sentineldesk/   core pipeline (classifier, vocabulary, safety, rag, pipeline, learning)
scripts/            evaluation, training, demo scripts
tests/              125 unit tests
web/                console + explain + hybrid + vocab + learn + audit + correct tabs + server
hybrid/             hybrid retrieval (BM25 + BGE-M3/Qdrant + Graph RAG + RRF)
semantic/           semantic edge-case fallback
optimizer/          adaptive keyword/prompt optimization (held-out evaluation)
security/           PII redaction, encryption, RBAC
external_test/      validation on real public (Zenodo) tickets
data/               corpora + trained model
```

---

## Validation

- **98%** accuracy on 2,996 real-style tickets (tiered routing → 0.983).
- Tier split: confident SVM 92.8% @1.00 · voter ladder 6.8% @0.76 · LLM tiebreak 0.3% @1.00.
- Synonym ablation: dormant on canonical text (0.98 → 0.98), load-bearing on casual text (0.42 → 1.00).
- Real public tickets (Zenodo): Network 99%, Access Management 75%, combined 89.2%.
- Tests: 125 unit · 15/15 functional system checks · 6 security tests.

---

## Limitations (stated honestly)

- Training and primary evaluation use **synthetic data** (LLM-generated from a seed lexicon), which is near-canonical; the 98% shows robustness to phrasing of in-domain concepts, and the Zenodo result is the real-generalization evidence on the readable subset.
- The classification core is **lexical-first**; semantic search runs as a hybrid/fallback layer, not the primary classifier.
- **Graph-RAG resolution coverage is ~67%** of root causes; the rest escalate.
- Low-signal tickets can drift toward a Network/Access-Management bias, mitigated by the confidence gate (escalation).
- Security: PII redaction / encryption-at-rest / RBAC implemented; Vault and mTLS are deployment-layer; Presidio NER is on the roadmap.

---

## Author

Mrinal Desai — linkedin.com/in/mrinal-d-30093134

*Built for the AI-Code-Sarathi / NASSCOM Agentic AI Hackathon 2026.*
