import re
import pytest
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from backend.src.data_processing.text_preprocessor import TextPreprocessor

"""
This file contains comprehensive unit tests for the TextPreprocessor class.
It isolates individual operations by directly testing methods such as:
  - remove_newlines
  - remove_links
  - keep_only_alphanumeric
  - remove_non_english_words
  - extract_words
  - remove_repeated_words_and_adjacent_numbers
  - lemmatize
  - remove_stopwords
Additionally, it tests the __call__ method by monkeypatching the contextual_filter
to simply return the input, so that only the text processing operations are validated.
"""

# For unit tests bypass contextual filtering by monkeypatching it.
@pytest.fixture
def preprocessor(monkeypatch):
    tp = TextPreprocessor()
    # Override the contextual_filter to be identity
    tp.contextual_filter = lambda text: text
    return tp

def test_remove_newlines(preprocessor):
    text = "This   is a \n test.\tAnother line."
    result = preprocessor.remove_newlines(text)
    # All consecutive whitespace should be replaced by a single space.
    assert re.search(r"\s+", result) is None or "  " not in result
    # Check that there are no newline characters.
    assert "\n" not in result

def test_remove_links(preprocessor):
    text = "Check out https://example.com and http://example2.org for more info."
    result = preprocessor.remove_links(text)
    # URLs should be removed.
    assert "https://example.com" not in result
    assert "http://example2.org" not in result

def test_keep_only_alphanumeric(preprocessor):
    text = "Hello, Jeff! This is a test: 123."
    result = preprocessor.keep_only_alphanumeric(text)
    # Non-alphanumeric characters should be removed (except underscore).
    assert re.search(r"[^a-zA-Z0-9_ ]", result) is None

def test_remove_non_english_words(preprocessor, monkeypatch):
    # For testing, override wordnet.synsets to simulate english words.
    monkeypatch.setattr("backend.src.data_processing.text_preprocessor.wordnet.synsets", lambda word: [] if word == "dfghje" else ["dummy"])
    text = "hello dfghje i a"
    # "hello", "i", and "a" should be kept (assuming i and a are allowed as single letters)
    result = preprocessor.remove_non_english_words(text)
    # "dfghje" should be removed.
    assert "dfghje" not in result
    # Check that words "hello" and "i" (or "a") appear.
    assert "hello" in result
    # Depending on stopwords filtering inside remove_non_english_words, the test may include "i" or "a".
    
def test_extract_words(preprocessor):
    
    # Using wordninja.split. For example, "thequickbrownfoxjumpsoverthelazydog" should split to [the quick brown fox jumps over the lazy dog]
    text = "thequickbrownfoxjumpsoverthelazydog"
    result = preprocessor.extract_words(text)
    assert result == "the quick brown fox jumps over the lazy dog"

def test_remove_repeated_words_and_adjacent_numbers(preprocessor):
    text = "word word 123 456 123 word"
    result = preprocessor.remove_repeated_words_and_adjacent_numbers(text)
    # "word" should not be repeated and adjacent numbers (123 and 456) should be reduced.
    # Expected behavior: "word 123 word" if repeated words and consecutive numbers are removed.
    expected = "word 123 word"
    assert result.strip() == expected

def test_lemmatize(preprocessor):
    # Test that lemmatize converts words to their base form.
    # Note: Without POS tagging, WordNetLemmatizer defaults to noun form.
    text = "cars running"
    result = preprocessor.lemmatize(text)
    # "cars" should become "car". "running" may remain "running" as a noun.
    assert "car" in result.lower()

def test_remove_stopwords(preprocessor):
    text = "this is a simple test of stopword removal"
    result = preprocessor.remove_stopwords(text)

    # Split the result into tokens for precise checking
    tokens = result.lower().split()

    # Common stopwords like "is", "a", "of", "this" should be removed.
    for stop in ["this", "is", "a", "of"]:
        assert stop not in tokens

def test_call(preprocessor, monkeypatch):
    """
    Test the __call__ method by bypassing contextual filtering.
    """
    # For unit test, ensure that __call__ uses the existing operations.
    # Since __call__ applies operations until stable, we expect the text to be processed repeatedly.
    # We'll supply a sample text and check for a non-empty output.
    sample_text = "Check out https://example.com! ThisIsATestTest."
    result = preprocessor(sample_text)
    # Because our __call__ applies our operations and then contextual filtering (identity in this test),
    # we expect the result to be non-empty.
    assert isinstance(result, str)
    assert result.strip() != ""
