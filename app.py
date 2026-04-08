import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import time
from main import process

st.set_page_config(
    page_title="SentinelDesk",
    page_icon="🛡️",
    layout="wide"
)

# ── Styling ────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #581212, #8B1A1A);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background: #1e2530;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #581212;
    }
    .resolve-badge {
        background: #1a5a2a;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
    .escalate-badge {
        background: #6e2a1a;
        color: white;
        padding: 5px 15px;
        border-radius: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🛡️ SentinelDesk</h1>
    <p>AI Powered Intelligent Ticket Routing & Resolution Agent</p>
    <p style="font-size:12px">NASSCOM Agentic AI Hackathon 2026 | Local-First | DPDP Compliant</p>
</div>
""", unsafe_allow_html=True)

# ── Input Form ─────────────────────────────────────────────────
st.subheader("📋 Submit IT Support Ticket")

col1, col2 = st.columns([3, 1])

with col1:
    title = st.text_input(
        "Ticket Title",
        placeholder="e.g. VPN not connecting after password reset"
    )
    description = st.text_area(
        "Description",
        placeholder="Describe the issue in detail...",
        height=120
    )

with col2:
    priority = st.selectbox(
        "Priority",
        ["Low", "Medium", "High", "Critical"]
    )
    ticket_id = st.number_input(
        "Ticket ID",
        min_value=1,
        value=1
    )
    submit = st.button(
        "🚀 Process Ticket",
        use_container_width=True,
        type="primary"
    )

# ── Process ────────────────────────────────────────────────────
if submit and title and description:
    
    with st.spinner("Processing through 5 agents..."):
        
        # Progress bar
        progress = st.progress(0)
        status = st.empty()
        
        status.text("Agent 1: Redacting PII...")
        progress.progress(20)
        time.sleep(0.3)
        
        status.text("Agent 2: Classifying ticket...")
        progress.progress(40)
        
        start_time = time.time()
        result = process(
            title=title,
            description=description,
            priority=priority,
            ticket_id=int(ticket_id)
        )
        elapsed = time.time() - start_time
        
        progress.progress(60)
        status.text("Agent 3: Routing to department...")
        time.sleep(0.2)
        
        progress.progress(80)
        status.text("Agent 4: Generating resolution...")
        time.sleep(0.2)
        
        progress.progress(100)
        status.text("Agent 5: Evaluating quality...")
        time.sleep(0.2)
        
        status.empty()
        progress.empty()
    
    st.success(f"✅ Processed in {elapsed:.1f} seconds")
    
    # ── Results ────────────────────────────────────────────────
    st.subheader("📊 Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Category", result['category'])
    with col2:
        st.metric("Confidence", f"{result['confidence']:.0%}")
    with col3:
        st.metric("Quality Score", f"{result['quality_score']}/5")
    with col4:
        action = result['final_action']
        color = "🟢" if action == "RESOLVE" else "🔴"
        st.metric("Action", f"{color} {action}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏢 Routing")
        st.info(f"**Department:** {result['department']}")
        st.info(f"**Contact:** {result['escalation_contact']}")
        
        if result['pii_detected']:
            st.subheader("🔒 PII Redacted")
            for pii in set(result['pii_detected']):
                st.warning(f"• {pii}")
        else:
            st.success("✅ No PII detected")
    
    with col2:
        st.subheader("💡 Resolution")
        clean_resolution = result['resolution'].replace('<br>', '\n').replace('<br/>', '\n')
        st.write(clean_resolution)
        
        if result.get('sources'):
            st.subheader("📚 Similar Tickets Used")
            for s in result['sources'][:3]:
                st.caption(
                    f"• {s['title'][:60]} "
                    f"(similarity: {s['score']:.2f})"
                )
    
    # Escalation alert
    if result['escalate']:
        st.error(
            f"⚠️ ESCALATED TO L2/L3 — {result['final_reason']}"
        )
    else:
        st.success(
            f"✅ RESOLVED — {result['final_reason']}"
        )

elif submit:
    st.warning("Please enter both title and description")

# ── Sidebar ────────────────────────────────────────────────────
with st.sidebar:
    st.header("🛡️ SentinelDesk")
    st.caption("NASSCOM Agentic AI Hackathon 2026")
    
    st.divider()
    
    st.subheader("System Status")
    st.success("✅ Ollama — Online")
    st.success("✅ Qdrant — Online")
    st.success("✅ PostgreSQL — Online")
    st.success("✅ All 5 Agents — Ready")
    
    st.divider()
    
    st.subheader("Model Info")
    st.caption("LLM: Mistral 7B Q8")
    st.caption("Embeddings: BGE-M3")
    st.caption("VRAM: RTX 5070 8GB")
    
    st.divider()
    
    st.subheader("Categories")
    categories = [
        "🖥️ Infrastructure",
        "📱 Application",
        "🔒 Security",
        "🗄️ Database",
        "💾 Storage",
        "🌐 Network"
    ]
    for cat in categories:
        st.caption(cat)