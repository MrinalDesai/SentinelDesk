# web — SentinelDesk demo surfaces

Two pages:

## 1. Showcase (static, no backend) — `index.html`
A polished single-file overview: architecture, real metrics, test grid, security
table, and an interactive walkthrough of recorded runs. Just open it in a browser
(online, for the charts/fonts). Good for sharing / GitHub Pages.

## 2. Live Console (real backend) — `console.html` + `server.py`
A judge-facing console that runs your REAL pipeline live:
- pick a real ticket (from `data/real_3000.csv`) or type one,
- watch every layer fire with its actual output (safety → synonym normalization →
  SVM with/without normalization → scorer → kNN → resolver → Graph RAG → reasoning
  chain → verdict),
- press an experiment button to run the real ablation/accuracy/PII/etc. scripts and
  show their actual output — for when a judge asks "show me the synonym ablation"
  or "how does PII redaction work".

### Run it
```
pip install fastapi uvicorn       # (already in requirements.txt)
python web/server.py              # serves console at http://127.0.0.1:8000
```
Then open http://127.0.0.1:8000 . Start Ollama first if you want the live
LLM-tiebreak layer; otherwise that layer falls back gracefully.

Nothing is simulated — every result is your real code, reproducible on the spot.
