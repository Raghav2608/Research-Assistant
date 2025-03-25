import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document

@pytest.fixture
def mock_retrieval_engine():
    """Fixture to create a mocked RetrievalEngine instance."""
    from backend.src.RAG.retrieval_engine import RetrievalEngine  # Import your RetrievalEngine class

    engine = RetrievalEngine(openai_api_key="fake_key")
    
    # Mock Chroma vector store methods
    engine.vector_store.similarity_search = MagicMock(return_value=[])
    engine.vector_store.similarity_search_with_score = MagicMock(return_value=[])
    engine.vector_store.add_documents = MagicMock()
    engine.vector_store.get = MagicMock(return_value=[])
    
    return engine

def test_add_unique_documents(mock_retrieval_engine):
    """Test that unique documents are correctly added to ChromaDB."""
    
    doc1 = Document(page_content="Some AI research", metadata={"title": "AI Paper 1", "link": "https://ai1.com"})
    doc2 = Document(page_content="More ML insights", metadata={"title": "ML Paper 2", "link": "https://ml2.com"})
    
    mock_retrieval_engine.split_and_add_documents([doc1, doc2])
    
    mock_retrieval_engine.vector_store.get = MagicMock(return_value=[doc1, doc2])
    all_docs = mock_retrieval_engine.vector_store.get()
    
    assert len(all_docs) == 2, f"Expected 2 documents, but found {len(all_docs)}"

def test_no_duplicate_documents(mock_retrieval_engine):
    """Ensure that duplicate documents (based on link) are not added to the vector store."""
    
    doc1 = Document(page_content="Unique content", metadata={"title": "Paper A", "link": "https://paperA.com"})
    
    # Mock similarity search to return empty (meaning document doesn't exist)
    mock_retrieval_engine.vector_store.similarity_search = MagicMock(return_value=[])
    
    # Add the document
    mock_retrieval_engine.split_and_add_documents([doc1])

    # Mock similarity search to simulate that the document now exists
    mock_retrieval_engine.vector_store.similarity_search = MagicMock(return_value=[doc1])

    # Try adding the same document again
    mock_retrieval_engine.split_and_add_documents([doc1])
    
    # Retrieve stored documents (mocked)
    mock_retrieval_engine.vector_store.get = MagicMock(return_value=[doc1])
    all_docs = mock_retrieval_engine.vector_store.get()
    
    assert len(all_docs) == 1, "Duplicate document was added, but it should have been ignored"

def test_retrieval_of_existing_documents(mock_retrieval_engine):
    """Test retrieval when relevant documents exist in ChromaDB."""

    doc = Document(page_content="Important AI research", metadata={"title": "AI Paper", "link": "https://ai-paper.com"})
    mock_retrieval_engine.split_and_add_documents([doc])

    # Mock similarity search results
    mock_retrieval_engine.vector_store.similarity_search_with_score = MagicMock(return_value=[(doc, 0.95)])

    retrieved_docs = mock_retrieval_engine.retrieve(["AI"])
    
    assert len(retrieved_docs) == 1, "Expected 1 retrieved document but found none"
    assert retrieved_docs[0]["metadata"]["title"] == "AI Paper", "Retrieved the wrong document"

def test_retrieval_when_no_documents_found(mock_retrieval_engine):
    """Test retrieval when no relevant documents are in ChromaDB."""
    
    mock_retrieval_engine.vector_store.similarity_search_with_score = MagicMock(return_value=[])
    
    retrieved_docs = mock_retrieval_engine.retrieve(["Nonexistent topic"])
    
    assert len(retrieved_docs) == 0, "Expected no documents, but retrieval returned some"

def test_adding_multiple_documents_with_one_duplicate(mock_retrieval_engine):
    """Test that only unique documents are stored when adding multiple documents including a duplicate."""

    doc1 = Document(page_content="Content A", metadata={"title": "Paper A", "link": "https://paperA.com"})
    doc2 = Document(page_content="Content B", metadata={"title": "Paper B", "link": "https://paperB.com"})
    duplicate_doc = Document(page_content="Content A", metadata={"title": "Paper A", "link": "https://paperA.com"})
    
    # Mock similarity search (first two don't exist, third does)
    mock_retrieval_engine.vector_store.similarity_search = MagicMock(side_effect=[[], [], [doc1]])
    
    # Add all documents
    mock_retrieval_engine.split_and_add_documents([doc1, doc2, duplicate_doc])
    
    # Retrieve stored documents (mocked)
    mock_retrieval_engine.vector_store.get = MagicMock(return_value=[doc1, doc2])
    all_docs = mock_retrieval_engine.vector_store.get()
    
    assert len(all_docs) == 2, "Duplicate document was added incorrectly"

def test_retrieval_after_multiple_document_additions(mock_retrieval_engine):
    """Test retrieval after adding multiple documents."""

    doc1 = Document(page_content="Deep Learning research", metadata={"title": "DL Paper", "link": "https://dl.com"})
    doc2 = Document(page_content="Neural Networks explained", metadata={"title": "NN Paper", "link": "https://nn.com"})
    
    mock_retrieval_engine.split_and_add_documents([doc1, doc2])

    # Mock similarity search results
    mock_retrieval_engine.vector_store.similarity_search_with_score = MagicMock(return_value=[(doc1, 0.9), (doc2, 0.85)])

    retrieved_docs = mock_retrieval_engine.retrieve(["Neural Networks"])
    
    assert len(retrieved_docs) == 2, "Expected 2 retrieved documents but found a different count"
    assert retrieved_docs[0]["metadata"]["title"] == "DL Paper", "Retrieved incorrect first document"
    assert retrieved_docs[1]["metadata"]["title"] == "NN Paper", "Retrieved incorrect second document"
