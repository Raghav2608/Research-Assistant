import pytest
from backend.src.data_processing.entry_processor import EntryProcessor

"""
This file contains integration tests for the EntryProcessor class.
These tests use the real TextPreprocessor (and its dependencies) to process paper entries end-to-end.
They verify that:
  - The entry is validated correctly.
  - The text fields (content, summary, title) are processed using real NLP operations.
  - The summarise_entry method produces a summary string containing all required information.
"""

@pytest.mark.integration
def test_entry_processor_integration_valid():
    """
    Integration test for the __call__ method of EntryProcessor.
    Processes a real entry using the actual TextPreprocessor and verifies that the text fields are modified.
    """
    ep = EntryProcessor()
    valid_entry = {
        "id": "1234.56789",
        "title": "A Sample Paper Title\nWith Newline",
        "summary": "This paper presents a novel approach to testing something, what I know is nothing.",
        "authors": ["Mick", "Jagger"],
        "published": "1945-12-25",
        "paper_link": "https://example.com",
        "content": "The content of the paper is very informative and detailed."
    }
    processed = ep(valid_entry)
    
    # Check that the title has been cleaned (i.e., newlines replaced with spaces and stripped).
    expected_title = ep.text_preprocessor.remove_newlines(valid_entry["title"]).strip()
    assert processed["title"] == expected_title
    
    # Verify that it has been processed (i.e. lowercased, normalized, and non-empty).
    assert isinstance(processed["summary"], str)
    assert processed["summary"].strip() != "", "The processed summary should be non-empty."
    # Check that the summary is lowercased.
    assert processed["summary"] == processed["summary"].lower()
    # Optionally, ensure that URLs have been removed.
    assert "http" not in processed["summary"]
    
    # For content similar processing.
    if "content" in valid_entry:
        assert isinstance(processed["content"], str)
        assert processed["content"].strip() != "", "The processed content should be non-empty."
        assert processed["content"] == processed["content"].lower()
        assert "http" not in processed["content"]

@pytest.mark.integration
def test_summarise_entry_integration():
    """
    Integration test for summarise_entry.
    Verifies that the summary string contains all key labels and parts of the entry.
    """
    ep = EntryProcessor()
    entry = {
        "id": "1234.56789",
        "title": "Systemic underserving in crisp packets",
        "summary": "Paper challenges the insane amount of air in crisp packets.",
        "authors": ["Eddie", "Murphy"],
        "published": "2023-09-16",
        "paper_link": "https://example.com",
        "content": "Detailed content for integration test."
    }
    summary = ep.summarise_entry(entry)
    required_labels = ["Time:", "ID:", "Title:", "Summary:", "Authors:", "Published:", "Semantic Scholar Link:", "Content:"]
    for label in required_labels:
        assert label in summary
    # Optionally check that key values appear.
    for field in ["1234.56789", "Systemic underserving in crisp packets", "Paper challenges the insane amount of air in crisp packets", "Eddie, Murphy", "2023-09-16", "https://example.com", "Detailed content"]:
        assert field in summary
