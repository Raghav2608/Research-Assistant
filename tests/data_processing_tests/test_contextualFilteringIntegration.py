import pytest
from backend.src.data_processing.contextual_filtering import ContextualFilter
import time

"""
This file contains an integration test for the ContextualFilter class.
It uses the real BERT model and tokenizer (from Hugging Face) and the real chunker to perform
end-to-end topic filtering on a sample text. This test verifies that the __call__ method completes
successfully and returns a non-empty filtered text. Since this test downloads model weights and may be slow,
it is marked as an integration test.
"""

@pytest.mark.integration
def test_contextual_filter_integration_performance():
    """
    Integration test for ContextualFilter using real dependencies.
    Processes a sample text and checks that the operation completes within a performance threshold.
    """
    cf = ContextualFilter()
    text = "This is a test sentence about computing and the recent advances of transformer models."
    start_time = time.time()
    result = cf(text)
    elapsed_time = time.time() - start_time
    # Assert that filtering completes within 60 seconds.
    assert elapsed_time < 60, f"ContextualFilter took too long: {elapsed_time} seconds"
    assert isinstance(result, str)
    assert result.strip() != ""

@pytest.mark.integration
def test_contextual_filter_integration_multiple_samples():
    """
    Integration test for ContextualFilter using multiple sample texts.
    Processes a list of texts and verifies that each returns a non-empty filtered output.
    """
    cf = ContextualFilter()
    sample_texts = [
        "What are the latest breakthroughs in computer vision?",
        "Tell me about advancements in natural language processing.",
        "How is artificial intelligence changing healthcare?",
        "What new research is being conducted in quantum computing?",
    ]
    results = [cf(text) for text in sample_texts]
    for result in results:
        assert isinstance(result, str)
        # We expect each filtered output to be non-empty.
        assert result.strip() != ""
