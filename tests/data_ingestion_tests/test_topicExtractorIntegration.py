import pytest
from backend.src.data_ingestion.arxiv.topic_extractor import TopicExtractor
import time

"""
This file contains an integration test for the TopicExtractor class.
It uses the real pre-trained T5 model and tokenizer from Hugging Face to verify that:

1. get_result:
   - Returns a non-empty string when processing a sentence.
2. __call__:
   - Processes a list of sentences and returns a list of non-empty topic strings.

Because this test uses the actual model, it may be slower and require network access.
Mark this test as an integration test so it can be run selectively.
"""

@pytest.mark.integration
def test_topic_extractor_integration():
    extractor = TopicExtractor()
    
    # Test get_result with a sample sentence.
    sentence = "What are the latest advancements in Computer Vision?"
    start_time = time.time()
    result = extractor.get_result(sentence)
    elapsed_time = time.time() - start_time
    # Since the model output can vary check that the result is a non-empty string.
    assert isinstance(result, str)
    assert result.strip() != ""
    assert elapsed_time < 30, f"get_result took too long: {elapsed_time} seconds"
    
    # Test the __call__ method with a list of sentences.
    sentences = [
        "What are the latest advancements in Computer Vision?",
        "Tell me about natural language processing."
    ]
    topics = extractor(sentences)
    assert isinstance(topics, list)
    assert len(topics) == len(sentences)
    for topic in topics:
        assert isinstance(topic, str)
        assert topic.strip() != ""

@pytest.mark.integration
def test_topic_extractor_edge_conditions():
    extractor = TopicExtractor()
    
    # Test get_result with an empty string.
    empty_result = extractor.get_result("")
    # Even if input is empty, get_result should return a string (which may be empty).
    assert isinstance(empty_result, str)
    
    # Test __call__ with a list of empty or whitespace-only strings.
    edge_sentences = ["", "   "]
    results = extractor(edge_sentences)
    assert isinstance(results, list)
    assert len(results) == len(edge_sentences)
    for res in results:
        # The extractor should handle edge cases.
        assert isinstance(res, str)