import pytest
from unittest.mock import MagicMock, patch
from backend.src.RAG.query_responder import QueryResponder  

@pytest.fixture
def mock_query_responder():
    """Fixture to create a QueryResponder instance with mocked dependencies."""
    with patch("backend.src.RAG.query_responder.ChatOpenAI") as mock_llm, \
         patch("backend.src.RAG.query_responder.RunnableWithMessageHistory") as mock_runnable, \
         patch("backend.src.RAG.query_responder.Memory") as mock_memory:
        
        mock_llm.return_value = MagicMock()
        mock_runnable.return_value = MagicMock()
        mock_memory.return_value = MagicMock()
        
        responder = QueryResponder(openai_api_key="fake_key", session_id="1234")
        return responder

def test_initialization(mock_query_responder):
    """Test if QueryResponder initializes correctly."""
    assert mock_query_responder.session_id == "1234"
    assert mock_query_responder.qa_chain is not None
    assert mock_query_responder.memory is not None

def test_format_documents(mock_query_responder):
    """Test if format_documents correctly formats the retrieved documents."""
    docs = [
        {"metadata": {"link": "http://example.com"}, "page_content": "This is content 1."},
        {"metadata": {"link": "http://example2.com"}, "page_content": "This is content 2."}
    ]
    expected_output = (
        "Source: http://example.com\nContent: This is content 1.\n\n"
        "Source: http://example2.com\nContent: This is content 2."
    )
    assert mock_query_responder.format_documents(docs) == expected_output

def test_format_documents_empty(mock_query_responder):
    """Test if format_documents handles an empty list correctly."""
    assert mock_query_responder.format_documents([]) == ""

def test_combine_context_and_question(mock_query_responder):
    """Test if combine_context_and_question correctly combines context and query."""
    context = "This is some academic content."
    query = "What is the main argument of the paper?"
    expected_output = {"context": context, "question": query}
    assert mock_query_responder.combine_context_and_question(context, query) == expected_output

def test_generate_answer_no_docs(mock_query_responder):
    """Test if generate_answer handles the case where no documents are retrieved."""
    mock_query_responder.qa_chain.invoke.return_value.content = "General answer with no context."
    response = mock_query_responder.generate_answer([], "What is AI?")
    assert response == "General answer with no context."

def test_generate_answer_with_docs(mock_query_responder):
    """Test if generate_answer correctly calls the LLM with formatted documents."""
    docs = [
        {"metadata": {"link": "http://example.com"}, "page_content": "AI is the study of intelligence."}
    ]
    mock_query_responder.qa_chain.invoke.return_value.content = "AI is the study of intelligence. [Source: http://example.com]"
    response = mock_query_responder.generate_answer(docs, "What is AI?")
    assert "AI is the study of intelligence." in response
    assert "[Source: http://example.com]" in response
