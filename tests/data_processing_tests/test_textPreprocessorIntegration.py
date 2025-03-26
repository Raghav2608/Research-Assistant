import re
import pytest
from backend.src.data_processing.text_preprocessor import TextPreprocessor

"""
This script contains integration tests for the TextPreprocessor class.
It verifies the complete end-to-end functionality of the text preprocessing pipeline.
The tests ensure that the pipeline:
  - Normalizes whitespace and removes newlines.
  - Removes URLs from the input.
  - Keeps only alphanumeric characters.
  - Splits concatenated words using wordninja.
  - Filters out non-English words.
  - Removes common English stopwords.
  - Lemmatizes words to their base forms.
  - Eliminates repeated words and adjacent numbers.
  - Applies (or bypasses) contextual filtering.
The final output is validated to be a clean, normalized, and non-empty string.
"""

@pytest.mark.integration
def test_text_preprocessor():
    # Instantiate the TextPreprocessor.
    tp = TextPreprocessor()
    # For testing purposes, override the contextual_filter with an identity function.
    tp.contextual_filter = lambda text: text

    # Sample input text that contains various artifacts:
    # - A URL that should be removed.
    # - Newlines and extra whitespace.
    # - A concatenated word ("beachclub") that should be split.
    # - Repeated words and adjacent numbers.
    # - Stopwords and non-English words.
    sample_text = (
        "This is a test URL: https://example.com and some repeated repeated words. \n"
        "Also, consider a concatenated word like beachclub. \n"
        "There are some numbers   123 456 123 and stopwords such as this, is, a, of. \n"
        "And non-English words: asdf qwerty."
    )

    # Process the sample text using the complete preprocessing pipeline.
    processed_text = tp(sample_text)

    # Ensure that URLs have been removed.
    assert "http" not in processed_text, "URLs should be removed."

    # Check that newlines are removed and excessive whitespace is normalized.
    assert "\n" not in processed_text, "Newlines should be normalized."
    assert not re.search(r'\s{2,}', processed_text), "Multiple consecutive spaces should be reduced to a single space."

    # Verify that repeated words and adjacent numbers have been reduced.
    assert "repeated repeated" not in processed_text, "Repeated words should be removed."
    assert "123 456 123" not in processed_text, "Adjacent numbers should be reduced."

    # Check that common stopwords are removed.
    tokens = processed_text.split()
    for stop in ["this", "is", "a", "of"]:
        assert stop not in tokens, f"Stopword '{stop}' should be removed."

    # Verify that concatenated words have been split.
    # For example, "beachclub" should be split into "beach" and "club".
    assert "beach" in processed_text, "The word 'beach' should appear after splitting concatenated words."
    assert "club" in processed_text, "The word 'club' should appear after splitting concatenated words."

    # Check that non-English words are removed.
    for non_eng in ["asdf", "qwerty"]:
        assert non_eng not in processed_text, f"Non-English word '{non_eng}' should be removed."

    # Finally, ensure that the processed text is non-empty.
    assert processed_text.strip() != "", "Processed text should be a non-empty string."
