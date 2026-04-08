# 🛡️ SentinelDesk
### AI Powered Intelligent Ticket Routing & Resolution Agent
**NASSCOM Agentic AI Hackathon 2026 | Use Case 1**

[![Live Demo](https://img.shields.io/badge/Live-Demo-green)](https://beth-relocation-hawk-grateful.trycloudflare.com)

---

## 🎯 What It Does
SentinelDesk is a privacy-preserving, multi-agent AI system that:
- **Classifies** IT tickets into 6 domains automatically
- **Routes** to correct department via RBAC rules
- **Resolves** using RAG from historical tickets
- **Escalates** when confidence is low
- **Redacts** all PII before any AI processing

## 🏗️ Architecture
5-Agent LangGraph Pipeline:
1. **Agent 1 — Intake**: Presidio PII redaction
2. **Agent 2 — Classifier**: BGE-M3 + Mistral 7B
3. **Agent 3 — Router**: RBAC + PostgreSQL
4. **Agent 4 — RAG Resolver**: Qdrant MMR + Mistral
5. **Agent 5 — Escalation**: LLM-as-judge quality scoring

## 🔒 Privacy First
- Zero PII leaves the system
- Fully local inference (Ollama + Mistral 7B)
- DPDP Act 2023 compliant
- Immutable audit trail

## 🛠️ Tech Stack
| Layer | Tool |
|-------|------|
| LLM | Mistral 7B Q8 (Ollama) |
| Embeddings | BGE-M3 |
| Vector DB | Qdrant |
| Agents | LangGraph |
| Database | PostgreSQL |
| Privacy | Presidio |
| Frontend | Streamlit |
| Tunnel | Cloudflare |

## 🚀 Run Locally

### Prerequisites
- Python 3.11
- Docker Desktop
- Ollama

### Setup
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/SentinelDesk.git
cd SentinelDesk

# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download models
ollama pull mistral:7b-instruct-q8_0
ollama pull bge-m3

# Start databases
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
docker run -d --name postgres -e POSTGRES_PASSWORD=sentineldesk123 -e POSTGRES_DB=sentineldesk -p 5432:5432 postgres:16

# Setup database
python utils/database.py

# Generate synthetic dataset
python data/generate_tickets.py

# Ingest tickets
python data/ingest_tickets.py

# Run app
streamlit run app.py
```

## 📊 Performance
| Metric | Score |
|--------|-------|
| Classification | Network ✅ |
| Confidence | 86-92% |
| Quality Score | 4/5 |
| PII Redaction | 100% |
| Response Time | <15 seconds |

## 🏆 NASSCOM Hackathon
- **Team**: SentinelDesk
- **Use Case**: 1 — AI Powered Ticket Routing
- **Category**: Agentic AI
- **Level**: Medium