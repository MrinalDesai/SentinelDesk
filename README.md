# SentinelDesk

Agentic ITSM ticket routing and resolution — NASSCOM Agentic AI Hackathon 2026.

## Build status (greenfield, in progress)

Bottom-up build. Each layer is testable before the next depends on it.

| Layer | Status | Notes |
|-------|--------|-------|
| Data contracts (Pydantic models) | done | `src/sentineldesk/models/` |
| Config (domains, priorities, thresholds) | done | `src/sentineldesk/config.py` |
| Stage 0 safety layer (regex) | done + tested | `src/sentineldesk/safety/` |
| Vocabulary builder (N-gram + TF-IDF + merge) | done + tested | `src/sentineldesk/vocabulary/` |
| LLM enrichment (Layer 4, Algo 4) | done | Ollama client + stub; run `scripts/build_vocabulary.py --enrich` |
| Synthetic data generation (batch) | done + tested | `scripts/generate_data.py` |
| Term-seeded generation (vocabulary-first) | done + tested | `data_gen/word_model.py`, `term_seeded.py` |
| Vocabulary database (single source of truth) | done + tested | `vocabulary/database.py`, `concepts.py`; `data/vocabulary_db.json` |
| End-to-end pipeline (gen+DB+SVM+ablation) | done | run `scripts/build_pipeline.py` |
| Shared LLM client (Ollama) | done | `src/sentineldesk/llm/` |
| SVM training (Algo 5) + CV report | done + tested | `src/sentineldesk/classifier/svm.py`; run `scripts/train_svm.py` |
| Deterministic scorer + explanation (Stage 2) | done + tested | `src/sentineldesk/classifier/scorer.py`, `explain.py` |
| Layman map (enrichment seed) | done + tested | `data_gen/layman_map.py`, `data/layman_map.csv` |
| Streamlit prototype | done | `app/streamlit_app.py` |
| Agents 1-5 + LangGraph | pending | |
| BM25 retrieval | next | testable in CI |
| Qdrant / BGE-M3 / Graph RAG | pending | your machine |
| FastAPI + Streamlit + Postgres | pending | |

## Run the tests

    pip install -r requirements.txt
    pytest -q

## Run the prototype UI

    pip install streamlit
    streamlit run app/streamlit_app.py

## Inspect the vocabulary tables

    # N-gram + TF-IDF top-N per category, with exclusivity metric (no Ollama):
    python scripts/vocab_table.py --out-dir data/vocab_tables
    # re-run on your real corpus to verify it stays exclusive + abundant:
    python scripts/vocab_table.py --in data/synthetic_tickets.csv --out-dir data/vocab_tables

## Environment split

## Build the vocabulary

    # CPU-only (layers 1-3), no Ollama needed:
    python scripts/build_vocabulary.py --in data/seed_tickets.csv --out data/vocabulary.json

    # Full build with local Mistral enrichment (layer 4):
    ollama pull mistral:7b
    python scripts/build_vocabulary.py --in data/synthetic_tickets.csv \
        --out data/vocabulary.json --enrich --model mistral:7b

## Run the prototype UI

    pip install streamlit
    streamlit run app/streamlit_app.py

## Inspect the vocabulary tables

    # N-gram + TF-IDF top-N per category, with exclusivity metric (no Ollama):
    python scripts/vocab_table.py --out-dir data/vocab_tables
    # re-run on your real corpus to verify it stays exclusive + abundant:
    python scripts/vocab_table.py --in data/synthetic_tickets.csv --out-dir data/vocab_tables

## Environment split

The deterministic spine (models, Stage 0, N-gram/TF-IDF vocab, SVM, BM25,
ensemble logic, LangGraph wiring) runs and is tested anywhere. The
service-dependent pieces (Mistral via Ollama, BGE-M3 embeddings, Qdrant,
Presidio, PostgreSQL) require local services and a GPU and are validated on
the target machine (RTX 5070, per Round 2 §12.2).
