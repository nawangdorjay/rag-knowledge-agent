# 📚 RAG Knowledge Agent

A Retrieval-Augmented Generation agent that answers questions using your own data — not the LLM's training data. Answers are grounded, accurate, and verifiable.

Built by [Nawang Dorjay](https://github.com/nawangdorjay) — for **GSSoC 2026**.

---

## 🎯 What is RAG?

Most AI chatbots rely on the LLM's training data → they hallucinate, give outdated info, or don't know your domain.

**RAG fixes this:**
1. **Chunk** your documents into passages
2. **Embed** each chunk into a vector (numerical representation of meaning)
3. **Store** vectors in a searchable index (FAISS)
4. **Retrieve** the most relevant chunks when a user asks a question
5. **Generate** an answer grounded in retrieved data

```
User: "When to sow wheat?"
         ↓
Vector search → finds: "Wheat: Sown Oct-Nov. Each week delay after Nov 15 loses 50kg/ha."
         ↓
LLM + context → "Wheat should be sown in October-November. Delaying after November 15 reduces yield by ~50 kg per hectare per week."
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Fast, good quality, 384-dim vectors |
| **Vector DB** | FAISS (Facebook AI Similarity Search) | Fast nearest-neighbor search, no server |
| **LLM** | Groq Llama 3.3 / OpenAI GPT-4o-mini | Grounded answer generation |
| **UI** | Streamlit | Interactive chat interface |

---

## 🚀 Quick Start

```bash
git clone https://github.com/nawangdorjay/rag-knowledge-agent.git
cd rag-knowledge-agent
pip install -r requirements.txt
cp .env.example .env
# Add GROQ_API_KEY=gsk_xxxxx
streamlit run app.py
```

---

## 📁 Structure

```
rag-knowledge-agent/
├── app.py                        # Streamlit chat UI
├── agent/
│   ├── __init__.py
│   ├── knowledge_base.py         # Vector store, chunking, search
│   └── rag_agent.py              # RAG pipeline (retrieve → generate)
├── data/
│   ├── crops_knowledge.json      # Crop knowledge (rice, wheat, cotton, etc.)
│   └── schemes_knowledge.json    # Government schemes (PM-KISAN, Ayushman, etc.)
├── tests/
│   └── test_rag.py               # 6 tests
├── requirements.txt
└── .github/workflows/ci.yml
```

---

## 🧠 Learn & Extend

This project is designed as a **learning base**. Here's what to explore:

### Level 1: Understand the basics
- Read `knowledge_base.py` — understand chunking, embedding, indexing
- Run `test_rag.py` — see how data flows through the pipeline
- Try different questions in the UI

### Level 2: Improve retrieval
- **TODO:** Add overlapping chunks (sliding window) for better context
- **TODO:** Try different embedding models (multilingual for Hindi)
- **TODO:** Add MMR (Maximal Marginal Relevance) for diverse results
- **TODO:** Implement a re-ranker (cross-encoder) for better ranking

### Level 3: Add data sources
- **TODO:** Parse PDF files (use PyPDF2 or pdfplumber)
- **TODO:** Scrape web pages (requests + BeautifulSoup)
- **TODO:** Add markdown/text file support
- **TODO:** Build from your existing farmer/health agent data

### Level 4: Production features
- **TODO:** Add source citation in answers
- **TODO:** Implement conversation memory
- **TODO:** Add evaluation metrics (retrieval precision, answer faithfulness)
- **TODO:** Cache embeddings for faster startup

---

## 🧪 Tests

```bash
python tests/test_rag.py
```

6 tests covering: JSON chunking, knowledge base operations, data loading, and edge cases.

---

## 📄 License

MIT

## 👨‍💻 Author

**Nawang Dorjay** — B.Tech CSE (Data Science), MAIT Delhi | [GitHub](https://github.com/nawangdorjay)

---

## 🤖 AI-Assisted Development

This project was built with AI assistance. See [BUILDING.md](BUILDING.md) for full transparency.
