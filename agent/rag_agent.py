"""
RAG Agent — Combines retrieval with generation.

Flow:
1. User asks a question
2. Search knowledge base for relevant chunks
3. Feed chunks + question to LLM
4. LLM generates answer grounded in retrieved data

This is the standard RAG pipeline used in production by companies like:
- Perplexity AI
- You.com
- GitHub Copilot Chat (retrieves from your codebase)
- Most enterprise chatbots

For you to learn and extend:
- Add re-ranking (cross-encoder) after initial retrieval
- Try different prompt templates
- Add citation/source tracking
- Implement conversation memory
"""
import os
import json
from typing import Optional, List
from agent.knowledge_base import KnowledgeBase

RAG_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on the provided context.

RULES:
1. Answer ONLY based on the provided context. If the context doesn't contain the answer, say "I don't have information about this in my knowledge base."
2. Be specific and cite details from the context (numbers, names, dates).
3. If the user asks in Hindi, respond in Hindi.
4. Keep answers concise but complete.
5. Never make up information not in the context.

You will be given CONTEXT from a knowledge base, then the USER's question."""


class RAGAgent:
    """
    RAG-powered agent that answers questions using retrieved knowledge.
    
    Usage:
        agent = RAGAgent()
        agent.load_knowledge("path/to/knowledge")
        answer = agent.query("What is the best time to sow wheat?")
    """

    def __init__(self, api_key: Optional[str] = None, provider: str = "groq"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.provider = provider
        self.kb: Optional[KnowledgeBase] = None
        self.conversation_history = []

    def build_knowledge_from_json(self, json_files: List[str], source_label: str = "data"):
        """
        Build knowledge base from JSON files.
        Each JSON file should be a dict or list that gets chunked into text.
        
        TODO for you:
        - Add support for PDF files (use PyPDF2)
        - Add support for markdown files
        - Add support for web pages (use requests + BeautifulSoup)
        - Implement smarter chunking (by paragraph, with overlap)
        """
        kb = KnowledgeBase()
        chunks = []
        metadatas = []

        for json_file in json_files:
            path = Path(json_file)
            if not path.exists():
                print(f"Warning: {json_file} not found, skipping")
                continue

            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Chunk the JSON data into text passages
            new_chunks, new_meta = self._chunk_json(data, str(path.name))
            chunks.extend(new_chunks)
            metadatas.extend(new_meta)

        kb.add_documents(chunks, metadatas)
        kb.build_index()
        self.kb = kb

        print(f"Knowledge base built: {len(chunks)} chunks from {len(json_files)} files")
        return kb

    def _chunk_json(self, data, source: str):
        """
        Convert JSON data into text chunks.
        
        This is a simple chunker. For production, you'd want:
        - Recursive chunking for nested structures
        - Overlapping chunks (sliding window)
        - Semantic chunking (split by meaning, not just structure)
        
        TODO for you: Improve this chunker for your data.
        """
        chunks = []
        metadatas = []

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, dict):
                    # Flatten nested dict into a readable chunk
                    text = f"{key}:\n"
                    for k, v in value.items():
                        if isinstance(v, list):
                            text += f"  {k}: {', '.join(str(i) for i in v)}\n"
                        else:
                            text += f"  {k}: {v}\n"
                    chunks.append(text.strip())
                    metadatas.append({"source": source, "topic": key})
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            text = f"{key}:\n" + "\n".join(f"  {k}: {v}" for k, v in item.items())
                        else:
                            text = f"{key}: {item}"
                        chunks.append(text.strip())
                        metadatas.append({"source": source, "topic": key})
                else:
                    chunks.append(f"{key}: {value}")
                    metadatas.append({"source": source, "topic": key})

        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    text = "\n".join(f"{k}: {v}" for k, v in item.items())
                else:
                    text = str(item)
                chunks.append(text.strip())
                metadatas.append({"source": source})

        return chunks, metadatas

    def load_knowledge(self, path: str):
        """Load a previously built knowledge base."""
        self.kb = KnowledgeBase()
        self.kb.load(path)

    def save_knowledge(self, path: str):
        """Save the knowledge base to disk."""
        if self.kb:
            self.kb.save(path)

    def query(self, question: str, top_k: int = 3) -> dict:
        """
        Answer a question using RAG.
        
        Returns:
            dict with:
            - 'answer': The generated answer
            - 'sources': List of retrieved chunks used
            - 'confidence': Based on retrieval scores
        """
        if not self.kb:
            return {"error": "No knowledge base loaded. Call build_knowledge_from_json() or load_knowledge() first."}

        # Step 1: Retrieve relevant chunks
        results = self.kb.search(question, top_k=top_k)

        if not results:
            return {"answer": "I don't have any relevant information in my knowledge base.", "sources": []}

        # Step 2: Build context from retrieved chunks
        context_parts = []
        for i, r in enumerate(results):
            context_parts.append(f"[Source {i+1} — {r['metadata'].get('topic', 'unknown')}]\n{r['text']}")
        context = "\n\n".join(context_parts)

        # Step 3: Generate answer with LLM
        answer = self._generate_with_context(question, context)

        # Step 4: Calculate confidence (inverse of distance — lower distance = higher confidence)
        avg_score = sum(r["score"] for r in results) / len(results)
        confidence = max(0, 1 - (avg_score / 10))  # Rough normalization

        return {
            "answer": answer,
            "sources": [{"text": r["text"][:200], "metadata": r["metadata"]} for r in results],
            "confidence": round(confidence, 2),
            "avg_retrieval_score": round(avg_score, 3),
        }

    def _generate_with_context(self, question: str, context: str) -> str:
        """Generate answer using LLM with retrieved context."""
        try:
            import openai
        except ImportError:
            return "Error: pip install openai"

        is_groq = self.provider == "groq" or "groq" in os.getenv("GROQ_API_KEY", "").lower()
        client = openai.OpenAI(
            api_key=self.api_key,
            base_url="https://api.groq.com/openai/v1" if is_groq else None,
        )

        messages = [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": f"CONTEXT:\n{context}\n\nQUESTION: {question}"},
        ]

        try:
            model = "llama-3.3-70b-versatile" if is_groq else "gpt-4o-mini"
            response = client.chat.completions.create(
                model=model, messages=messages,
                temperature=0.3,  # Low temp for factual answers
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating answer: {e}"
