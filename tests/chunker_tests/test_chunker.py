import pytest
import torch
from backend.src.chunking.chunker import Chunker
from typing import List, Dict

"""
This file contains unit tests for the Chunker class.
It isolates the Chunker by replacing its BertTokenizer with a dummy tokenizer that
returns predictable outputs for tokenization and decoding. The tests cover:
  - convert_text_to_tokens: ensuring that a text is converted into a dummy token dictionary.
  - get_chunks with return_as_text=False: verifying that the chunks are returned as dictionaries of tensors.
  - get_chunks with return_as_text=True: verifying that the chunks are returned as a list of decoded token strings.
"""

class DummyTokenizer:
    def __init__(self):
        pass

    def __call__(self, text: str, return_tensors: str):
        # Assume the dummy tokenization returns 10 tokens.
        # Create dummy tensors with values 1 through 10.
        tokens = torch.tensor([[1,2,3,4,5,6,7,8,9,10]])
        # For simplicity, token_type_ids and attention_mask are the same shape.
        return {
            "input_ids": tokens,
            "token_type_ids": tokens,
            "attention_mask": torch.ones_like(tokens)
        }
    
    def decode(self, token: torch.Tensor) -> str:
        # Decode a token (an integer) into a string "token_{value}"
        value = token.item()
        return f"token_{value}"

@pytest.fixture
def dummy_chunker(monkeypatch):
    """
    Creates a Chunker instance with a dummy tokenizer.
    """
    chunker = Chunker(chunk_size=5)
    # Override the tokenizer with dummy tokenizer.
    dummy_tokenizer = DummyTokenizer()
    chunker.tokenizer = dummy_tokenizer
    return chunker

def test_convert_text_to_tokens(dummy_chunker):
    text = "Hello world"
    tokens = dummy_chunker.convert_text_to_tokens(text)
    # tokens should be a dict with keys: input_ids, token_type_ids, attention_mask.
    for key in ["input_ids", "token_type_ids", "attention_mask"]:
        assert key in tokens
        # Check that the tensor shape is [1, 10] as defined in DummyTokenizer.
        assert tokens[key].shape == (1, 10)

def test_get_chunks_as_tensors(dummy_chunker):
    text = "Test text for chunking."
    # With chunk_size=5 and default stride=512, 10 tokens < 512 so only one chunk is returned.
    chunks = dummy_chunker.get_chunks(text, return_as_text=False)
    # Expect one chunk
    assert isinstance(chunks, list)
    assert len(chunks) == 1
    chunk = chunks[0]
    # Each chunk should be a dict with tensors of shape [1, L], where L <= 5.
    for key in ["input_ids", "token_type_ids", "attention_mask"]:
        assert key in chunk
        shape = chunk[key].shape
        assert shape[0] == 1
        assert shape[1] <= 5

def test_get_chunks_as_text(dummy_chunker):
    text = "Test text for chunking."
    # Force stride to 5 so that with 10 tokens and chunk_size=5 = two chunks.
    chunks = dummy_chunker.get_chunks(text, return_as_text=True, stride=5)
    assert isinstance(chunks, list)
    assert len(chunks) == 2
    # Each chunk should be a list of strings.
    for chunk in chunks:
        assert isinstance(chunk, list)
        for token_str in chunk:
            assert token_str.startswith("token_")
