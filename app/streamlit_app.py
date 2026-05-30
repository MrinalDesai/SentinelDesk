"""
SentinelDesk — Streamlit prototype.

Runs the deterministic path live: Stage 0 safety gate + the explainable
classifier, with the generated explanation as the headline output. LLM-backed
stages (Mistral tiebreak, RAG resolution) are marked as stubs until wired.

Run:
    pip install streamlit
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sentineldesk.classifier import DeterministicScorer, VocabModel, explain  # noqa: E402
from sentineldesk.corpus import load_tickets_csv                              # noqa: E402
from sentineldesk.data_gen.controlled import generate_controlled_corpus       # noqa: E402
from sentineldesk.safety import safety_check                                  # noqa: E402

# category -> owning team (display only; real routing is Postgres RBAC later)
ROUTING = {
    "Network": "Network Team",
    "Application": "App / Dev Team",
    "Database": "DBA Team",
    "Storage": "Storage Team",
    "Infrastructure": "Infrastructure Team",
    "Security": "SOC Team",
    "Access Management": "IAM Team",
}


@st.cache_resource
def load_scorer():
    csv = Path(__file__).resolve().parents[1] / "data" / "synthetic_tickets.csv"
    if csv.exists():
        tickets = load_tickets_csv(csv)
        source = f"synthetic_tickets.csv ({len(tickets)} tickets)"
    else:
        tickets = generate_controlled_corpus(per_category=120)
        source = "controlled reference corpus (run generate_data.py for the real one)"
    return DeterministicScorer(VocabModel.build(tickets)), source


st.set_page_config(page_title="SentinelDesk", page_icon=":mag:", layout="centered")
st.title("SentinelDesk")
st.caption("Agentic ITSM ticket routing — explainable classifier prototype")

scorer, source = load_scorer()
st.caption(f"Vocabulary source: {source}")

title = st.text_input("Ticket title", "VPN keeps dropping")
desc = st.text_area(
    "Description",
    "Cannot connect to the vpn tunnel from home and dns resolution keeps failing.",
    height=110,
)

if st.button("Classify", type="primary"):
    # Stage 0 — safety gate
    safety = safety_check(title, desc)
    if safety.bypass_llm:
        st.error(
            f"Stage 0 — high-stakes escalation: **{safety.matched_category}** "
            f"→ {safety.department} (matched '{safety.trigger}', {safety.latency_ms:.2f} ms). "
            "All AI stages bypassed."
        )
        st.stop()

    # Stage 2 — classify
    r = scorer.classify(title, desc)

    if r.is_edge_case:
        st.warning(f"Edge case → resolver ladder. Best guess: {r.category}")
    else:
        st.success(f"Routed to {r.category} → {ROUTING.get(r.category, '—')}")

    c1, c2 = st.columns(2)
    c1.metric("Category", r.category)
    c2.metric("Confidence", f"{r.confidence:.0%}")

    st.subheader("Explanation")
    st.write(explain(r))

    st.subheader("Score breakdown")
    st.bar_chart({c: r.scores[c] for c in sorted(r.scores)})

    with st.expander("Matched terms (why each signal fired)"):
        st.write({k: v for k, v in r.matched.items() if v})
        if r.common_hits:
            st.caption(f"Shared/common words present: {', '.join(r.common_hits)}")

    st.divider()
    st.caption(
        "Stubbed for now: Stage 1 PII redaction, the LLM tiebreak on edge cases, "
        "Stage 4 RAG resolution, Stage 5 quality judge."
    )
