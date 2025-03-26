import pytest
import torch
from backend.src.chunking.chunker import Chunker

"""
This file contains integration tests for the Chunker class.
These tests use the real BertTokenizer from Hugging Face to verify end-to-end behavior.
They check that:
  - The convert_text_to_tokens method returns a valid token dictionary.
  - The get_chunks method correctly splits the tokenized text into chunks.
  - When return_as_text is True, the chunks are returned as lists of decoded token strings.
"""

@pytest.mark.integration
def test_convert_text_to_tokens_integration():
    chunker = Chunker(chunk_size=512)
    text = "This is an integration test for tokenization."
    tokens = chunker.convert_text_to_tokens(text)
    # Check that tokens is a dict and contains the expected keys.
    for key in ["input_ids", "token_type_ids", "attention_mask"]:
        assert key in tokens
        # Check that the tensor has at least one dimension.
        assert isinstance(tokens[key], torch.Tensor)

@pytest.mark.integration
def test_get_chunks_integration():
    chunker = Chunker(chunk_size=5)
    text = "This is an integration test for chunking. It should split the text into chunks based on the chunk size."
    # Call get_chunks with return_as_text False.
    chunks = chunker.get_chunks(text, return_as_text=False, stride=5)
    # Since real tokenization length may vary, check that we get at least one chunk and each chunk is a dict.
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    for chunk in chunks:
        assert isinstance(chunk, dict)
        for key in ["input_ids", "token_type_ids", "attention_mask"]:
            assert key in chunk
            # Check that each tensor has shape [1, L] where L <= 5
            assert chunk[key].dim() == 2
            assert chunk[key].shape[0] == 1
            assert chunk[key].shape[1] <= 5

    # Now test get_chunks with return_as_text True.
    text2 = "Another test for text output."
    chunks_text = chunker.get_chunks(text2, return_as_text=True, stride=5)
    # Expect chunks_text to be a list of lists of strings.
    assert isinstance(chunks_text, list)
    for chunk in chunks_text:
        assert isinstance(chunk, list)
        for token_str in chunk:
            assert isinstance(token_str, str)
