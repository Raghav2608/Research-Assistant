import pytest
from backend.src.data_processing.entry_processor import EntryProcessor

"""
This file contains unit tests for the EntryProcessor class.
It verifies the behavior of:
  - validate_entry: ensuring that valid entries pass and invalid ones raise errors.
  - summarise_entry: checking that the generated summary string contains the required fields.
  - __call__: confirming that the entries text fields are processed correctly.
  
The tests isolate the EntryProcessor by replacing its TextPreprocessor with dummy functions
to ensure that tests run quickly and deterministically.
"""

@pytest.fixture
def dummy_entry_processor(monkeypatch):
    """
    Fixture that returns an EntryProcessor instance with its text_preprocessor replaced
    by dummy functions for predictable behavior.
    """
    ep = EntryProcessor()
    
    # Replace the text_preprocessor with dummy lambdas:
    # When called, simply prefix the input with "processed:".
    dummy_preprocessor = lambda text: f"processed:{text}"
    ep.text_preprocessor = dummy_preprocessor
    # Also override remove_newlines to simulate cleaning.
    ep.text_preprocessor.remove_newlines = lambda text: text.replace("\n", " ").strip()
    return ep

def test_validate_entry_valid(dummy_entry_processor):
    """
    Test that validate_entry passes for a valid entry.
    """
    valid_entry = {
        "id": "1234.56789",
        "title": "How to make money in London",
        "summary": "This paper presents a new method of phone stealing.",
        "authors": ["Maximus", "Maximillian"],
        "published": "2025-03-25",
        "paper_link": "https://www.semanticscholar.org/paper/1234.56789",
        "content": ""
    }
    # Should not raise an exception.
    dummy_entry_processor.validate_entry(valid_entry)

def test_validate_entry_invalid_type(dummy_entry_processor):
    """
    Test that validate_entry raises ValueError for an entry with incorrect types.
    """
    invalid_entry = {
        "id": 12345,  # Should be a string.
        "title": "A Sample Paper Title",
        "summary": "This paper presents a new method.",
        "authors": ["Alice", "Bob"],
        "published": "2020-03-17",
        "paper_link": "https://www.semanticscholar.org/paper/1234.56789",
        "content": ""
    }
    with pytest.raises(ValueError, match="Entry key: id has the wrong type"):
        dummy_entry_processor.validate_entry(invalid_entry)

def test_validate_entry_missing_optional_key(dummy_entry_processor):
    """
    Test that validate_entry does not raise an error if an optional key is missing.
    For example, if the "content" key is missing, which is acceptable.
    """
    entry_missing_optional = {
        "id": "1234.56789",
        "title": "How to lose a guy in 10 days",
        "summary": "This paper presents a new method to be single.",
        "authors": ["Matthew", "Reese"],
        "published": "2025-03-25",
        "paper_link": "https://www.semanticscholar.org/paper/1234.56789"
        # "content" is missing.
    }
    # This should not raise an error.
    dummy_entry_processor.validate_entry(entry_missing_optional)

def test_validate_entry_invalid_authors(dummy_entry_processor):
    """
    Test that validate_entry raises ValueError if the authors list contains non-string values.
    """
    entry_invalid_authors = {
        "id": "1234.56789",
        "title": "Kunf-fu panda lore",
        "summary": "This paper presents a new method to be the dragon warrior.",
        "authors": ["Po", 123],  # 123 is not a string.
        "published": "2021-01-01",
        "paper_link": "https://www.semanticscholar.org/paper/1234.56789",
        "content": ""
    }
    with pytest.raises(ValueError, match="Entry authors list contains non-string values"):
        dummy_entry_processor.validate_entry(entry_invalid_authors)

def test_summarise_entry(dummy_entry_processor):
    """
    Test that summarise_entry produces a summary string that contains key information.
    """
    entry = {
        "id": "1234.56789",
        "title": "Live, Laugh, Love",
        "summary": "This paper presents a new method on how to be insufferable",
        "authors": ["Rob", "Bob"],
        "published": "2025-03-25",
        "paper_link": "https://example.com",
        "content": "Some content"
    }
    summary = dummy_entry_processor.summarise_entry(entry)
    # Check that all required fields are in the summary string.
    for label in ["Time:", "ID:", "Title:", "Summary:", "Authors:", "Published:", "Semantic Scholar Link:", "Content:"]:
        assert label in summary

def test_call_processing(dummy_entry_processor):
    """
    Test the __call__ method.
    It should validate the entry and process its text fields using the dummy text_preprocessor.
    """
    entry = {
        "id": "1234.56789",
        "title": "Test Title\nwith newline",
        "summary": "Test Summary",
        "authors": ["Drew", "Barry"],
        "published": "2004-10-08",
        "paper_link": "https://example.com",
        "content": "Original Content"
    }
    processed = dummy_entry_processor(entry)
    # For content and summary, the dummy preprocessor prefixes "processed:".
    assert processed["content"] == "processed:Original Content"
    assert processed["summary"] == "processed:Test Summary"
    # For title, __call__ applies remove_newlines then strips it.
    expected_title = "Test Title with newline".strip()
    # Our dummy remove_newlines just replaces newlines with space.
    assert processed["title"] == expected_title
