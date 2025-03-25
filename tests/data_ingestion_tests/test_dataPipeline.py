import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from backend.src.data_ingestion.data_pipeline import DataPipeline

"""
This file contains comprehensive unit tests for the DataPipeline class in the data ingestion module.
The tests cover the following aspects:

1. process_query:
   - Verifies that the text preprocessor methods (keep_only_alphanumeric, remove_newlines, remove_stopwords)
     are called in the correct sequence.
   - Ensures that the clean_search_query function is applied to the processed query, and the final output
     is as expected.

2. remove_duplicate_entries:
   - Checks that duplicate entries (based on the "title" field) are removed properly,
     returning only unique entries.

3. select_entries:
   - Tests the random selection logic for combining ArXiv and Semantic Scholar entries.
   - Uses a custom fake_choice function to simulate predictable random behavior, ensuring that out-of-range
     indices are handled gracefully.

4. retrieve_documents:
   - Verifies that the ingestion pipelines (ArXiv and Semantic Scholar) are called with the correct arguments
     and that the expected documents are returned.

5. run:
   - Tests the overall run method by mocking all internal steps (query processing, document retrieval,
     entry selection, duplicate removal, and data processing) to ensure that the complete pipeline flow works
     as intended.

6. run_no_data:
   - Simulates an edge case where the ingestion pipelines return no data,
     ensuring that the run method ultimately returns an empty list.

Overall, these tests validate the internal logic of the DataPipeline and its interactions with external components,
ensuring robust and predictable behavior.
"""

@pytest.fixture
def mock_data_pipeline(monkeypatch):
    """
    Fixture that returns a DataPipeline instance with its dependencies mocked.
    This includes:
      - data_processing_pipeline (and its text_preprocessor)
      - arxiv_data_ingestion_pipeline
      - ss_data_ingestion_pipeline
    """
    pipeline = DataPipeline()

    # Mock the data processing pipeline and text preprocessor.
    mock_processing_pipeline = MagicMock()
    mock_processing_pipeline.entry_processor.text_preprocessor = MagicMock()
    pipeline.data_processing_pipeline = mock_processing_pipeline

    pipeline.text_preprocessor = pipeline.data_processing_pipeline.entry_processor.text_preprocessor
    pipeline.text_preprocessor.keep_only_alphanumeric.return_value = "mocked_alphanumeric"
    # Simulate two calls to remove_newlines: first and second.
    pipeline.text_preprocessor.remove_newlines.side_effect = ["mocked_newlines_removed", "mocked_newlines_removed"]
    pipeline.text_preprocessor.remove_stopwords.return_value = "mocked_stopwords_removed"

    # Mock the ingestion pipelines.
    mock_arxiv_pipeline = MagicMock()
    mock_ss_pipeline = MagicMock()
    pipeline.arxiv_data_ingestion_pipeline = mock_arxiv_pipeline
    pipeline.ss_data_ingestion_pipeline = mock_ss_pipeline

    # Set max_total_entries for predictable iteration.
    pipeline.max_total_entries = 3

    return pipeline

def test_process_query(mock_data_pipeline):
    """
    Test that process_query calls the text preprocessor methods and clean_search_query
    in the correct sequence and returns the final processed query.
    """
    from backend.src.RAG.utils import clean_search_query
    with patch("backend.src.data_ingestion.data_pipeline.clean_search_query", return_value="cleaned_query") as mock_clean:
        result = mock_data_pipeline.process_query("original query")
        # Expect keep_only_alphanumeric called with "original query"
        mock_data_pipeline.text_preprocessor.keep_only_alphanumeric.assert_called_once_with("original query")
        # Expect remove_newlines to be called at least twice.
        assert mock_data_pipeline.text_preprocessor.remove_newlines.call_count >= 2
        mock_data_pipeline.text_preprocessor.remove_stopwords.assert_called_once()
        # Because the final call in process_query is:
        #   processed_query = clean_search_query(processed_query)
        # and the last remove_newlines returns "mocked_newlines_removed",
        # we expect clean_search_query to be called with "mocked_newlines_removed".
        mock_clean.assert_called_once_with("mocked_newlines_removed")
        assert result == "cleaned_query"

def test_remove_duplicate_entries(mock_data_pipeline):
    """
    Test that remove_duplicate_entries returns only unique entries by title.
    """
    entries = [
        {"title": "Paper One", "content": "A"},
        {"title": "Paper Two", "content": "B"},
        {"title": "Paper One", "content": "C"},  # Duplicate title
    ]
    unique = mock_data_pipeline.remove_duplicate_entries(entries)
    assert len(unique) == 2
    titles = [entry["title"] for entry in unique]
    assert "Paper One" in titles
    assert "Paper Two" in titles

def fake_choice(arg):
    """
    Custom fake_choice function to replace np.random.choice.
    - If arg is a list of booleans, alternate: first call returns True, second returns False, etc.
    - If arg is an integer (n), return 0 (i.e. the first valid index).
    - Otherwise, if arg is a list of numbers, return the first element.
    """
    if isinstance(arg, list):
        if all(isinstance(x, bool) for x in arg):
            fake_choice.bool_count += 1
            return True if fake_choice.bool_count % 2 == 1 else False
        else:
            return arg[0]
    elif isinstance(arg, int):
        # np.random.choice(n) is equivalent to choosing from range(n)
        return 0
    return arg
fake_choice.bool_count = 0

def test_select_entries(mock_data_pipeline, monkeypatch):
    """
    Test that select_entries correctly selects entries from the ArXiv and Semantic Scholar lists.
    This test uses a custom fake_choice to simulate predictable random selections.
    """
    # Setup sample data for two user queries.
    all_arxiv_entries = [
        [{"title": "Arxiv A1"}, {"title": "Arxiv A2"}],  # For query index 0
        [{"title": "Arxiv B1"}],                           # For query index 1
    ]
    all_ss_entries = [
        [{"title": "SS A1"}, {"title": "SS A2"}],
        [{"title": "SS B1"}],
    ]
    num_user_queries = 2

    # Patch numpy.random.choice with our custom fake_choice.
    monkeyatch_patch = patch("numpy.random.choice", side_effect=fake_choice)
    with monkeyatch_patch:
        selected = mock_data_pipeline.select_entries(all_arxiv_entries, all_ss_entries, num_user_queries)
    # We expect 3 entries.
    assert len(selected) == 3
    for entry in selected:
        assert "title" in entry

def test_retrieve_documents(mock_data_pipeline):
    """
    Test that retrieve_documents calls the ingestion pipelines with the processed query.
    """
    mock_data_pipeline.arxiv_data_ingestion_pipeline.fetch_entries.return_value = [{"title": "arxiv doc"}]
    mock_data_pipeline.ss_data_ingestion_pipeline.get_entries.return_value = [{"title": "ss doc"}]

    arxiv_docs, ss_docs = mock_data_pipeline.retrieve_documents("test query")
    mock_data_pipeline.arxiv_data_ingestion_pipeline.fetch_entries.assert_called_once_with(
        topic="test query", max_results=mock_data_pipeline.max_total_entries
    )
    mock_data_pipeline.ss_data_ingestion_pipeline.get_entries.assert_called_once_with(
        topic="test query", max_results=mock_data_pipeline.max_total_entries, desired_total=mock_data_pipeline.max_total_entries
    )
    assert arxiv_docs == [{"title": "arxiv doc"}]
    assert ss_docs == [{"title": "ss doc"}]

def test_run(mock_data_pipeline):
    """
    Test the run method of the DataPipeline by mocking all internal steps.
    """
    from backend.src.data_ingestion.data_pipeline import DataPipeline
    # Mock process_query to simply prepend "processed_" to the query.
    mock_data_pipeline.process_query = MagicMock(side_effect=lambda q: f"processed_{q}")
    # Mock retrieve_documents to return dummy documents for each query.
    mock_data_pipeline.retrieve_documents = MagicMock(side_effect=lambda q: (
        [{"title": f"arxiv_{q}_doc"}],
        [{"title": f"ss_{q}_doc"}]
    ))
    # Mock select_entries to return a fixed final list.
    mock_data_pipeline.select_entries = MagicMock(return_value=[
        {"title": "final doc1"},
        {"title": "final doc2"}
    ])
    # Mock remove_duplicate_entries to return the same list.
    mock_data_pipeline.remove_duplicate_entries = MagicMock(return_value=[
        {"title": "final doc1"}, {"title": "final doc2"}
    ])
    # Mock the final data processing step.
    mock_data_pipeline.data_processing_pipeline.process.return_value = [{"title": "processed doc"}]

    user_queries = ["query1", "query2"]
    result = mock_data_pipeline.run(user_queries)

    assert mock_data_pipeline.process_query.call_count == len(user_queries)
    assert mock_data_pipeline.retrieve_documents.call_count == len(user_queries)
    mock_data_pipeline.select_entries.assert_called_once()
    mock_data_pipeline.remove_duplicate_entries.assert_called_once()
    mock_data_pipeline.data_processing_pipeline.process.assert_called_once()
    assert result == [{"title": "processed doc"}]

def test_run_no_data(monkeypatch, mock_data_pipeline):
    """
    Simulate an edge case where the ingestion pipelines return no data.
    The final run should return an empty list after processing.
    """
    mock_data_pipeline.process_query = MagicMock(side_effect=lambda q: f"processed_{q}")
    mock_data_pipeline.retrieve_documents = MagicMock(return_value=([], []))
    mock_data_pipeline.select_entries = MagicMock(return_value=[])
    mock_data_pipeline.remove_duplicate_entries = MagicMock(return_value=[])
    mock_data_pipeline.data_processing_pipeline.process.return_value = []

    user_queries = ["query1", "query2"]
    result = mock_data_pipeline.run(user_queries)
    assert result == []
