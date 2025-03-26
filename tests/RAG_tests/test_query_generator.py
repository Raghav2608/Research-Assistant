import pytest
from unittest.mock import MagicMock, patch
from backend.src.RAG.query_generator import ResearchQueryGenerator  

@pytest.fixture
def mock_query_generator():
    """Fixture to create a ResearchQueryGenerator instance with mocked dependencies."""
    with patch("backend.src.RAG.query_generator.ChatOpenAI") as mock_llm, \
         patch("backend.src.RAG.query_generator.RunnableWithMessageHistory") as mock_runnable, \
         patch("backend.src.RAG.query_generator.Memory") as mock_memory:
        
        mock_llm.return_value = MagicMock()
        mock_runnable.return_value = MagicMock()
        mock_memory.return_value = MagicMock()
        
        generator = ResearchQueryGenerator(openai_api_key="fake_key", session_id="1234")
        return generator

def test_initialization(mock_query_generator):
    """Test if ResearchQueryGenerator initializes correctly."""
    assert mock_query_generator.session_id == "1234"
    assert mock_query_generator.query_chain is not None
    assert mock_query_generator.memory is not None

def test_generate_valid_query(mock_query_generator):
    """Test if generate correctly returns valid query variations."""
    mock_query_generator.query_chain.invoke.return_value.content = '["AI research", "Machine Learning trends"]'
    response = mock_query_generator.generate("Tell me about AI research")
    assert response == ["AI research", "Machine Learning trends"]

def test_generate_invalid_query(mock_query_generator):
    """Test if generate handles an invalid query response."""
    mock_query_generator.query_chain.invoke.return_value.content = '["ERROR: Invalid query."]'
    response = mock_query_generator.generate("??")
    assert response == "ERROR"

def test_generate_json_decode_error(mock_query_generator):
    """Test if generate handles JSON decode errors correctly."""
    mock_query_generator.query_chain.invoke.return_value.content = "Not a JSON string"
    response = mock_query_generator.generate("Tell me about AI research")
    assert response == ["ERROR: Failed to generate valid queries. Please try again."]

def test_generate_query_with_history(mock_query_generator):
    """Test if generate correctly utilizes history when relevant."""
    mock_query_generator.query_chain.invoke.return_value.content = '["AI ethics", "Fairness in AI"]'
    response = mock_query_generator.generate("What about ethics in AI?")
    assert response == ["AI ethics", "Fairness in AI"]

