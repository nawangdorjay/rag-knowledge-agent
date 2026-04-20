"""
RAG Knowledge Agent — Core Module

RAG = Retrieval-Augmented Generation
Instead of relying only on an LLM's training data, RAG:
1. Splits documents into chunks
2. Converts chunks to vector embeddings
3. Stores vectors in a searchable index (FAISS)
4. At query time: retrieves relevant chunks → feeds them to the LLM

This makes answers ACCURATE and GROUNDED in your data, not hallucinated.

For you to learn and extend:
- Understand how chunking affects retrieval quality
- Experiment with different embedding models
- Try different retrieval strategies (similarity, MMR)
- Add re-ranking for better results
"""
import os
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


class KnowledgeBase:
    """
    Manages the vector knowledge base.
    
    Architecture:
    - Documents are split into chunks (paragraphs or fixed-size)
    - Each chunk is converted to a vector using an embedding model
    - Vectors are stored in FAISS for fast similarity search
    - At query time, top-k similar chunks are retrieved
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize the knowledge base.
        
        Args:
            embedding_model: Sentence-transformer model name.
                Options:
                - "all-MiniLM-L6-v2" — fast, decent quality (384 dim)
                - "all-mpnet-base-v2" — slower, better quality (768 dim)
                - "paraphrase-multilingual-MiniLM-L12-v2" — supports Hindi/multilingual
        
        TODO for you: Try the multilingual model for Hindi queries.
        """
        self.embedding_model_name = embedding_model
        self.chunks: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings = None
        self.index = None
        self._model = None

    def _get_model(self):
        """Lazy-load the embedding model."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embedding_model_name)
            except ImportError:
                raise ImportError(
                    "Install sentence-transformers: pip install sentence-transformers\n"
                    "This is the core of RAG — don't skip it!"
                )
        return self._model

    def add_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Add documents to the knowledge base.
        
        Args:
            texts: List of text chunks
            metadatas: Optional metadata for each chunk (source, category, etc.)
        
        Example:
            kb.add_documents(
                texts=["Rice needs 1200mm rainfall", "Wheat is a Rabi crop"],
                metadatas=[{"source": "crops.json", "topic": "rice"}, 
                          {"source": "crops.json", "topic": "wheat"}]
            )
        """
        self.chunks.extend(texts)
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in texts])

        # Invalidate index — needs rebuild after adding docs
        self.index = None

    def build_index(self):
        """
        Build the FAISS vector index from all chunks.
        Call this after adding all documents, before querying.
        """
        if not self.chunks:
            raise ValueError("No documents added. Call add_documents() first.")

        model = self._get_model()

        try:
            import faiss
            import numpy as np
        except ImportError:
            raise ImportError("Install faiss-cpu and numpy: pip install faiss-cpu numpy")

        print(f"Encoding {len(self.chunks)} chunks...")
        self.embeddings = model.encode(self.chunks, show_progress_bar=True)
        self.embeddings = np.array(self.embeddings).astype("float32")

        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance
        self.index.add(self.embeddings)

        print(f"Index built: {self.index.ntotal} vectors, {dimension} dimensions")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks given a query.
        
        Args:
            query: User's question
            top_k: Number of results to return
        
        Returns:
            List of dicts with 'text', 'score', and 'metadata'
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        model = self._get_model()
        import numpy as np

        query_vector = model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vector, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append({
                    "text": self.chunks[idx],
                    "score": float(dist),  # Lower = more similar
                    "metadata": self.metadata[idx],
                })

        return results

    def save(self, path: str):
        """Save the index and chunks to disk."""
        import numpy as np
        import faiss

        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(save_dir / "index.faiss"))
        np.save(str(save_dir / "embeddings.npy"), self.embeddings)

        with open(save_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump({"chunks": self.chunks, "metadata": self.metadata}, f, ensure_ascii=False)

        print(f"Saved to {path}")

    def load(self, path: str):
        """Load a previously saved index."""
        import numpy as np
        import faiss

        load_dir = Path(path)

        self.index = faiss.read_index(str(load_dir / "index.faiss"))
        self.embeddings = np.load(str(load_dir / "embeddings.npy"))

        with open(load_dir / "chunks.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            self.chunks = data["chunks"]
            self.metadata = data["metadata"]

        print(f"Loaded {len(self.chunks)} chunks from {path}")
