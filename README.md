# SentinelDesk

**Agentic ITSM ticket triage — rules own safety, retrieval owns grounding, the LLM owns reasoning, and a deterministic ladder owns the routing.**

SentinelDesk takes an incoming IT support ticket (a title + free-text description, often in casual/layman language) and routes it to the correct team across seven ITSM domains. The high-volume decisions are made deterministically and cheaply; the LLM is used only where it genuinely adds value; and every decision is explainable and auditable.

**Domains:** Network · Access Management · Database · Application · Security · Infrastructure · Storage

---

## Design principle

> Rules own safety, retrieval owns grounding, the LLM owns reasoning.

The right tool is used for each job rather than calling an LLM for everything. Deterministic classifiers handle the bulk of routing in milliseconds, retrieval grounds decisions in real evidence, and the LLM is reserved for genuine tie-breaks. The result is fast, cheap, predictable, and auditable.

---

## How routing works

A ticket's title and description are combined into raw text and passed, as-is, to the classifiers. The router then walks a decision ladder:

1. **SVM confidence gate** — a linear SVM over TF-IDF features predicts the domain; if confident, the ticket resolves immediately. This single gate resolves **~92.8%** of tickets (accuracy ~1.00 on the held-out set).
2. **Voter ladder (scorer + SVM + kNN)** — three independent voters weigh in:
   - **DeterministicScorer** — keyword/evidence-signal scoring. It matches layman and synonym phrasings *as an evidence signal* — it does **not** rewrite or pre-classify the ticket.
   - **SVM** — the same linear classifier.
   - **kNN** — instance-based vote against the labelled corpus.
   When they agree, the ticket resolves (~6.8% of tickets).
3. **LLM tie-break** — only when the voters genuinely disagree does a local LLM (Mistral-7B) arbitrate. By design this fires rarely, inside the small escalation band.
4. **Escalate** — anything still unresolved (~0.3%) goes to a human.

**Net result:** ~99.6% of tickets resolved deterministically; overall routing accuracy **~0.980** on the held-out corpus.

> A separate semantic-fallback module exists in the codebase but is **not** a rung in this routing ladder.

---

## Retrieval & grounding (RAG)

Hybrid retrieval ranks evidence with three retrievers combined via reciprocal-rank fusion:

- **BM25** lexical retrieval
- **Semantic** retrieval using **BGE-M3** embeddings (vector search via **Qdrant**, in-memory)
- **Graph-RAG** retrieval

A graph-RAG **judge** supervises routing decisions, grounding them in retrieved evidence.

*Deployment-roadmap (stated honestly as not-yet-productionised): a persistent/relational store, containerisation, and a persistent vector index.*

---

## Vocabulary, synonyms & self-correction

- **Controlled vocabulary** — the canonical evidence terms per domain used by the scorer.
- **Layman/synonym matching** — casual phrasings are matched to domain evidence signals *inside the scorer* (an evidence signal, not a pre-routing rewrite).
- **Self-correction** — correct a misroute once and that phrasing is never routed wrong again: a vocabulary/dictionary update, not a model retrain. Guards reject generic phrases and collisions so a correction can't create a "magnet" class.
- **Live learning** — corrections feed back into the vocabulary in place.

---

## The Improve loop — agentic retraining (LangGraph)

A separate, self-improving loop that diagnoses the classifier's weaknesses, fixes them, and keeps the new model only if it genuinely validates better. This is the system's concrete **agentic + LangGraph** component — a real compiled `StateGraph` (langgraph), six agents over a shared state:

```
diagnose -> plan -> [retrain -> ground -> validate] -> judge -> SVM candidate
                 \__(data healthy)____________________> judge
```

The branch after `plan` is a **real LangGraph conditional edge** (`add_conditional_edges`): if any test trips, take the full retrain path; if the data is healthy, skip straight to the judge and change nothing.

**Agents:** `diagnose` (run tests + train baseline) → `plan` (collect remedies) → `retrain` (apply remedies, retrain candidate SVM) → `ground` (RAG retrieval of nearest tickets) → `validate` (held-out, candidate vs baseline) → `judge` (accept only if improved; write `svm_candidate.pkl`, else keep baseline). The Improve-loop judge is a **model-promotion gate** — separate from the graph-RAG judge in the live router.

**Data-driven problem → test → remedy table** (only tripped tests are fixed):

| Problem | Test | Trips | Remedy |
|---|---|---|---|
| Class imbalance | max/min class-size ratio | ≥ 3:1 | SMOTE oversampling |
| Noisy generic tokens | stopword fraction | > 0.50 | remove English stopwords |
| Short / low-signal text | median tokens per ticket | < 12 | add bigrams (1–2 grams) |
| Duplicate tickets | exact-duplicate count | > 0 | deduplicate |
| Class too rare to synthesize | smallest class size | < 5 | flag for data collection |

It targets the **SVM** — the only trained model in the deterministic tier (kNN is instance-based, the scorer is rule-based) and the highest-leverage one (it resolves ~92% of tickets on the fast path).

**Isolated by design:** reads data read-only, writes **only** to `improvement/out/`, never touches the production model (`data/svm_model.pkl`), the source tree, or any live route. It *proposes* a better SVM; promoting it is a deliberate, separate human step.

Run it standalone:
```bash
python improvement/agentic_retrain.py            # induced hard case  -> retrain path -> ACCEPT
python improvement/agentic_retrain.py balanced   # healthy data       -> skip path    -> no change
```

---

## Web app

A FastAPI app serving eleven self-contained tabs:

| Tab | What it shows |
|---|---|
| **Console** | Submit a ticket and watch it get triaged / routed |
| **Explain** | Decision explainability — why a ticket went where (SVM / scorer / kNN reasoning) |
| **Hybrid** | Hybrid retrieval: BM25 + semantic + Graph-RAG, fused |
| **Vocabulary** | The controlled vocabulary / evidence terms per domain |
| **Synonyms** | Layman/synonym matching demo — casual phrasing → signals |
| **Live Learning** | The system learning from human corrections in place |
| **Self-Correct** | Correct a misroute once; it never repeats that phrasing |
| **Data Gen** | Synthetic ticket generation |
| **External** | Evaluation against the real external (Zenodo) test set |
| **Audit** | Audit log / governance view |
| **Improve** | The agentic retraining loop (LangGraph) |

---

## Data

- **Synthetic training corpus** (`data/real_3000.csv`) — a synthetic, balanced corpus of 2,996 tickets (428 per domain × 7), used for training and held-out evaluation. *(The filename is historical; the data is synthetic.)*
- **Real external test set** (Zenodo, `external_test/`) — a genuinely real, independently-sourced set of 214 tickets used as an out-of-distribution check.

---

## Tech stack

| Layer | Tooling |
|---|---|
| Classifiers | scikit-learn (LinearSVC + TF-IDF, kNN), DeterministicScorer |
| Imbalance fix | imbalanced-learn (SMOTE) |
| Embeddings | BGE-M3 (sentence-transformers) |
| Vector store | Qdrant (in-memory) |
| LLM | Mistral-7B (Q8) via Ollama, local |
| Orchestration | LangGraph (the Improve loop) |
| Retrieval | BM25 + semantic + Graph-RAG, reciprocal-rank fusion |
| Security | Fernet / AES-128-CBC, RBAC, audit trail |
| Web | FastAPI + uvicorn |
| Runtime | Python 3.11 |

---

## Setup & run

```bash
# 1. install
python -m venv .venv
.venv\Scripts\activate          # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

# 2. local LLM (for the rare tie-break) — install Ollama separately, then:
ollama pull mistral:7b-instruct-q8_0

# 3. run the web app  (serves the Console at http://127.0.0.1:8000/)
python web/server.py

# 4. run the agentic improve loop (optional, also available from the Improve tab)
python improvement/agentic_retrain.py
```

---

## Testing

```bash
pytest
```

92 unit tests across 16 files cover the classifiers, scorer, resolver ladder, vocabulary/self-correction guards, and supporting modules.

---

## Repository layout

```
src/sentineldesk/
  classifier/      resolver (the ladder), scorer, svm, knn, explainability
  vocabulary/      controlled vocabulary + self-correction
  safety/          guards, PII, encryption
  rag/             graph_rag
  retrieval/       BM25 / semantic / fusion
  pipeline/        orchestrator, langgraph_app
  data_gen/        synthetic ticket generation
  learning/        live learning from corrections
  llm/             Mistral tie-break client
  models/          data contracts
improvement/
  agentic_retrain.py   the LangGraph improve loop  (writes only to improvement/out/)
web/                   FastAPI server + the self-contained tab HTML
tests/                 92 tests / 16 files
data/                  synthetic corpus + trained model
external_test/         real Zenodo set
```

---

## Honest framing

- The training corpus is **synthetic**; the Zenodo set is the genuinely real, out-of-distribution check.
- The LLM is **rare by design** — it only arbitrates the small disagreement band, which is why ~99.6% of routing is deterministic.
- Layman/synonym handling is an **evidence signal inside the scorer**, not a pre-routing rewrite.
- The live router runs the deterministic ladder on a fast serial path; the **Improve loop** is the part that is genuinely LangGraph-orchestrated end to end, with a real conditional edge that decides its own path.
- The Improve loop genuinely **declines to change a healthy model** — that restraint is a feature, not a limitation.
