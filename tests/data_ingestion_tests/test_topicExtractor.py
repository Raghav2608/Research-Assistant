import pytest
from backend.src.data_ingestion.arxiv.topic_extractor import TopicExtractor
from typing import List

"""
This file contains comprehensive tests for the TopicExtractor class 
in the semantic scholar module. It verifies the following functionality:

1. create_prompt:
   - Ensures the prompt is correctly constructed from a given sentence.

2. postprocess_text:
   - Checks that the text is split, concatenated without spaces, and converted to lower case.

3. get_result:
   - Simulates the model and tokenizer behavior (using monkeypatching) to ensure that get_result 
     returns the expected topic.

4. __call__:
   - Processes a list of sentences and returns a list of topics, applying get_result and postprocessing.

Note:
Since the real model and tokenizer are large and slow to load, dummy replacements 
are used in tests to simulate their behavior.
"""

# Fixture to create a TopicExtractor instance with dummy tokenizer and model.
@pytest.fixture
def dummy_topic_extractor(monkeypatch):
    extractor = TopicExtractor()
    
    # Override the tokenizer with a dummy object.
    class DummyTokenizer:
        def __call__(self, prompt, return_tensors):
            # Return a dummy object with an input_ids attribute.
            class DummyInput:
                input_ids = "dummy_input"
            return DummyInput()
        def decode(self, output, skip_special_tokens):
            # Return a fixed dummy topic.
            return "Dummy Topic"
    
    # Override the model with a dummy object.
    class DummyModel:
        def generate(self, input_ids, max_length, num_beams, early_stopping):
            # Return a list with a dummy output.
            return [b"dummy_output"]
    
    # Replace the real tokenizer and model with dummy ones.
    extractor.tokenizer = DummyTokenizer()
    extractor.model = DummyModel()
    
    return extractor

def test_create_prompt():
    """
    Test that create_prompt correctly embeds the input sentence into the prompt.
    """
    extractor = TopicExtractor()
    sentence = "What is the future of AI?"
    expected_prompt = f"Extract the main topic that the user is asking about in the sentence: {sentence}"
    prompt = extractor.create_prompt(sentence)
    assert prompt == expected_prompt

def test_postprocess_text():
    """
    Test that postprocess_text concatenates words and converts text to lower case.
    """
    extractor = TopicExtractor()
    text = "Hello World"
    # "Hello World" -> ["Hello", "World"] -> "HelloWorld" -> "helloworld"
    processed = extractor.postprocess_text(text)
    assert processed == "helloworld"

def test_get_result(dummy_topic_extractor):
    """
    Test that get_result returns the expected dummy topic.
    """
    # Using the dummy_topic_extractor fixture, get_result should return "Dummy Topic"
    sentence = "What are the latest trends in robotics?"
    result = dummy_topic_extractor.get_result(sentence)
    # Since get_result does not call postprocess_text, it should return the decoded string "Dummy Topic".
    assert result == "Dummy Topic"

def test_call(dummy_topic_extractor):
    """
    Test the __call__ method to ensure it processes a list of sentences and returns topics.
    """
    sentences: List[str] = [
        "What is the future of AI?",
        "How is robotics evolving?",
        "Tell me about advances in biotechnology."
    ]
    # Each call to get_result will return "Dummy Topic", and then postprocess_text will convert it to "dummytopic"
    expected_output = ["dummytopic"] * len(sentences)
    result = dummy_topic_extractor(sentences)
    assert result == expected_output