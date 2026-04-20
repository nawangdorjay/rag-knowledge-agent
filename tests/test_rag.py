"""Tests for RAG Knowledge Agent."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.knowledge_base import KnowledgeBase
from agent.rag_agent import RAGAgent


def test_chunk_json_dict():
    """Test JSON chunking for dict data."""
    data = {
        "rice": {"name": "Rice", "season": "Kharif", "water": "High"},
        "wheat": {"name": "Wheat", "season": "Rabi", "water": "Moderate"},
    }
    agent = RAGAgent.__new__(RAGAgent)
    chunks, meta = agent._chunk_json(data, "test.json")
    assert len(chunks) == 2
    assert "rice" in chunks[0].lower() or "Rice" in chunks[0]
    assert "wheat" in chunks[1].lower() or "Wheat" in chunks[1]
    print(f"✅ test_chunk_json_dict ({len(chunks)} chunks)")


def test_chunk_json_list():
    """Test JSON chunking for list data."""
    data = [
        {"name": "Scheme A", "benefit": "₹5000"},
        {"name": "Scheme B", "benefit": "₹10000"},
    ]
    agent = RAGAgent.__new__(RAGAgent)
    chunks, meta = agent._chunk_json(data, "test.json")
    assert len(chunks) == 2
    assert "Scheme A" in chunks[0]
    print(f"✅ test_chunk_json_list ({len(chunks)} chunks)")


def test_chunk_json_nested():
    """Test chunking with nested data."""
    data = {
        "crops": [
            {"name": "Rice", "yield": "3t/ha"},
            {"name": "Wheat", "yield": "4t/ha"},
        ]
    }
    agent = RAGAgent.__new__(RAGAgent)
    chunks, meta = agent._chunk_json(data, "test.json")
    assert len(chunks) >= 2
    print(f"✅ test_chunk_json_nested ({len(chunks)} chunks)")


def test_knowledge_base_add():
    """Test adding documents to knowledge base."""
    kb = KnowledgeBase()
    kb.add_documents(
        texts=["Rice is a Kharif crop", "Wheat is a Rabi crop", "Cotton needs black soil"],
        metadatas=[{"topic": "rice"}, {"topic": "wheat"}, {"topic": "cotton"}],
    )
    assert len(kb.chunks) == 3
    assert len(kb.metadata) == 3
    print("✅ test_knowledge_base_add passed")


def test_real_data_loading():
    """Test loading actual data files."""
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    
    crops_file = os.path.join(data_dir, "crops_knowledge.json")
    schemes_file = os.path.join(data_dir, "schemes_knowledge.json")
    
    assert os.path.exists(crops_file), "crops_knowledge.json not found"
    assert os.path.exists(schemes_file), "schemes_knowledge.json not found"
    
    with open(crops_file) as f:
        crops = json.load(f)
    with open(schemes_file) as f:
        schemes = json.load(f)
    
    assert len(crops) > 0
    assert len(schemes) > 0
    
    agent = RAGAgent.__new__(RAGAgent)
    chunks_c, _ = agent._chunk_json(crops, "crops_knowledge.json")
    chunks_s, _ = agent._chunk_json(schemes, "schemes_knowledge.json")
    
    total = len(chunks_c) + len(chunks_s)
    assert total > 10
    print(f"✅ test_real_data_loading ({len(chunks_c)} crop chunks + {len(chunks_s)} scheme chunks = {total} total)")


def test_search_without_index():
    """Test that search fails gracefully without index."""
    kb = KnowledgeBase()
    try:
        kb.search("test")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not built" in str(e)
        print("✅ test_search_without_index passed")


if __name__ == "__main__":
    tests = [
        test_chunk_json_dict, test_chunk_json_list, test_chunk_json_nested,
        test_knowledge_base_add, test_real_data_loading, test_search_without_index,
    ]
    passed = failed = 0
    for t in tests:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"❌ {t.__name__}: {e}"); failed += 1
    print(f"\n{'='*40}\n{passed} passed, {failed} failed")
    if not failed:
        print("All tests passed! 🎉")
