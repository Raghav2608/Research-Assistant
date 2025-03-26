import pytest
import torch
import numpy as np
from backend.src.data_processing.contextual_filtering import ContextualFilter
from typing import List, Dict, Any

"""
This file contains unit tests for the ContextualFilter class in the data_processing module.
It isolates the ContextualFilter by replacing its heavy dependencies (the BERT model, tokenizer, and chunker)
with dummy implementations, allowing you to verify the internal logic without incurring the cost of loading
large models. The tests cover:
  - get_similarity: ensuring cosine similarity returns 1 for identical embeddings.
  - filter_based_on_context: checking that tokens (except first and last) are kept based on high similarity.
  - __call__: verifying that the overall filtering pipeline returns the expected output based on dummy chunks.
"""

# Dummy implementations to replace heavy dependencies.
class DummyModel:
    def __init__(self):
        pass
    def __call__(self, **inputs):
        # inputs should have an "input_ids" tensor with shape [1, L].
        L = inputs["input_ids"].shape[1]
        # Return dummy output with all ones, shape [1, L, 768].
        dummy_output = type("DummyOutput", (), {"last_hidden_state": torch.ones(1, L, 768)})
        return dummy_output

class DummyChunker:
    def __init__(self, chunk_size):
        self.chunk_size = chunk_size
    def get_chunks(self, text: str, return_as_text: bool, stride: int):
        # For simplicity, always return one chunk with a fixed tensor.
        dummy_input_ids = torch.tensor([[101, 200, 300, 400, 102]])  # Simulate 5 tokens.
        return [{"input_ids": dummy_input_ids}]

class DummyTokenizer:
    def __init__(self):
        pass
    def decode(self, tokens):
        # Return a constant word for each call.
        return "word"

@pytest.fixture
def dummy_contextual_filter(monkeypatch):
    """
    Fixture that returns a ContextualFilter instance with dummy model, chunker, and tokenizer.
    Also sets similarity_threshold=0 so that all tokens (except first and last) are kept.
    """
    cf = ContextualFilter()
    cf.tokenizer = DummyTokenizer()
    cf.model = DummyModel()
    cf.chunker = DummyChunker(chunk_size=512)
    cf.similarity_threshold = 0.0
    return cf

def test_get_similarity_identical(dummy_contextual_filter):
    """
    Test that get_similarity returns 1.0 for identical embeddings.
    """
    emb = torch.rand(1, 768)
    sim = dummy_contextual_filter.get_similarity(emb, emb)
    np.testing.assert_almost_equal(sim[0][0], 1.0, decimal=5)

def test_get_similarity_different(dummy_contextual_filter):
    """
    Test that get_similarity returns a value less than 1.0 for different embeddings.
    """
    emb1 = torch.ones(1, 768)
    emb2 = torch.zeros(1, 768)
    sim = dummy_contextual_filter.get_similarity(emb1, emb2)
    # Cosine similarity should be 0 in this case.
    np.testing.assert_almost_equal(sim[0][0], 0.0, decimal=5)

def test_filter_based_on_context_normal(dummy_contextual_filter):
    """
    Test that filter_based_on_context returns expected text when given a normal tensor.
    Using dummy tokens and embeddings, with threshold set to 0 so all inner tokens are kept.
    """
    tokens = torch.tensor([[101, 200, 300, 400, 102]])
    embeddings = torch.ones(1, 5, 768)
    result = dummy_contextual_filter.filter_based_on_context(tokens, embeddings)
    # Tokens 0 and 4 are skipped. Tokens 1,2,3 yield "word" each.
    expected = "word word word"
    assert result.strip() == expected

def test_filter_based_on_context_edge_case(dummy_contextual_filter):
    """
    Test filter_based_on_context for edge case when token tensor has fewer than 3 tokens.
    In such case, since first and last tokens are always skipped, expect an empty result.
    """
    # Only 2 tokens: shape [1,2]
    tokens = torch.tensor([[101, 102]])
    embeddings = torch.ones(1, 2, 768)
    result = dummy_contextual_filter.filter_based_on_context(tokens, embeddings)
    # With only 2 tokens, both will be skipped (as first and last).
    assert result.strip() == ""

def test_call(dummy_contextual_filter):
    """
    Test the __call__ method to ensure it processes text and returns filtered topics.
    """
    result = dummy_contextual_filter("This is a test sentence for contextual filtering.")
    # Our dummy implementations cause chunker to always return a fixed chunk leading to filtered text "word word word"
    expected = "word word word"
    assert result.strip() == expected