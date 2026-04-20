"""
RAG Knowledge Agent — Streamlit UI
Ask questions about farming and government schemes.
Answers are grounded in real data, not hallucinated.
"""
import streamlit as st
import os
from pathlib import Path
from agent.rag_agent import RAGAgent

st.set_page_config(page_title="📚 RAG Knowledge Agent", page_icon="📚", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #6a1b9a, #ab47bc);
        color: white; padding: 1.5rem 2rem; border-radius: 12px;
        margin-bottom: 1rem; text-align: center;
    }
    .source-card {
        background: #f3e5f5; border-left: 4px solid #7b1fa2;
        padding: 0.7rem 1rem; border-radius: 8px; margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .confidence-high { color: #2e7d32; }
    .confidence-med { color: #f57f17; }
    .confidence-low { color: #c62828; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def build_knowledge_base():
    """Build knowledge base from data files (cached — runs once)."""
    data_dir = Path(__file__).parent / "data"
    json_files = [
        str(data_dir / "crops_knowledge.json"),
        str(data_dir / "schemes_knowledge.json"),
    ]
    existing = [f for f in json_files if Path(f).exists()]
    if not existing:
        return None, "No data files found"

    agent = RAGAgent()
    try:
        kb = agent.build_knowledge_from_json(existing, source_label="farming_knowledge")
        return agent, None
    except Exception as e:
        return None, str(e)


def main():
    st.markdown("""
    <div class="main-header">
        <h1>📚 RAG Knowledge Agent</h1>
        <p style="font-size:1.1rem; margin:0;">Ask questions about farming, crops, and government schemes</p>
        <p style="font-size:0.9rem; margin:0.5rem 0 0 0; opacity:0.9;">
            Retrieval-Augmented Generation — answers grounded in real data, not hallucinated
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.title("⚙️ Settings")
        api_key = st.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...",
            help="Free at console.groq.com")
        
        st.divider()
        st.subheader("📊 Knowledge Base")
        
        agent, error = build_knowledge_base()
        if error:
            st.error(f"Error: {error}")
            st.info("This needs: pip install sentence-transformers faiss-cpu numpy")
            return
        
        if agent:
            st.success(f"✅ {len(agent.kb.chunks)} chunks loaded")
            st.caption(f"Model: {agent.kb.embedding_model_name}")
        
        st.divider()
        st.subheader("💡 How RAG Works")
        st.markdown("""
        1. **Your question** → converted to vector
        2. **Search** → find similar chunks in knowledge base
        3. **Retrieve** → top-k relevant passages
        4. **Generate** → LLM answers using retrieved context
        
        This means answers come from **your data**, not the LLM's training data.
        """)

        st.divider()
        st.caption("By [Nawang Dorjay](https://github.com/nawangdorjay) — GSSoC 2026")

    if not agent:
        st.error("Knowledge base failed to load.")
        return

    # Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg:
                with st.expander("📚 Sources used"):
                    for s in msg["sources"]:
                        st.markdown(f'<div class="source-card">{s}</div>', unsafe_allow_html=True)

    prompt = st.chat_input("Ask about crops, farming, or government schemes...")

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching knowledge base..."):
                if api_key:
                    agent.api_key = api_key
                result = agent.query(prompt)

            if "error" in result:
                st.error(result["error"])
                return

            st.markdown(result["answer"])

            # Confidence
            conf = result.get("confidence", 0)
            conf_class = "confidence-high" if conf > 0.7 else ("confidence-med" if conf > 0.4 else "confidence-low")
            st.markdown(f'<span class="{conf_class}">Confidence: {conf:.0%}</span>', unsafe_allow_html=True)

            # Sources
            sources = result.get("sources", [])
            source_texts = []
            if sources:
                with st.expander("📚 Sources used (what the AI referenced)"):
                    for s in sources:
                        topic = s.get("metadata", {}).get("topic", "unknown")
                        text = s["text"]
                        st.markdown(f'<div class="source-card"><b>{topic}</b>: {text}</div>', unsafe_allow_html=True)
                        source_texts.append(f"[{topic}] {text[:100]}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": source_texts,
        })

    # Examples
    if not st.session_state.messages:
        st.subheader("💡 Try asking:")
        for q in [
            "What is the best time to sow wheat?",
            "PM-KISAN ke liye kaun se documents chahiye?",
            "How much water does rice need?",
            "Kisan Credit Card ka interest rate kya hai?",
            "What are common pests in cotton?",
            "MGNREGA mein 15 din mein kaam nahi mila toh kya karein?",
        ]:
            st.markdown(f"- *{q}*")


if __name__ == "__main__":
    main()
