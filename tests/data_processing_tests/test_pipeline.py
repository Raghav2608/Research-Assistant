import pytest
import time
from backend.src.data_processing.pipeline import DataProcessingPipeline

"""
This file contains unit tests for the Data Processing Pipeline.
It isolates the DataProcessingPipeline by replacing its entry_processor with dummy functions,
so that we can verify the behavior of the process method without incurring the cost or variability
of the full text processing.
The tests cover:
  - Successful processing of all entries.
  - Handling of errors during processing (one entry fails but processing continues).
  - That processing is performed in-place and returns the modified entries.
"""
@pytest.fixture
def dummy_pipeline(monkeypatch):
    """
    Fixture that creates a DataProcessingPipeline instance with a dummy entry_processor.
    The dummy entry_processor simply returns a modified version of the entry,
    unless a special flag "fail" is set to simulate an error.
    """
    dp = DataProcessingPipeline()
    
    # Create a dummy entry processor function.
    def dummy_entry_processor(entry):
        if entry.get("fail"):
            raise Exception("Simulated processing error")
        # For demonstration, simply add a "processed" field.
        processed_entry = entry.copy()
        processed_entry["title"] = f"processed:{entry['title']}"
        processed_entry["summary"] = f"processed:{entry['summary']}"
        if "content" in entry:
            processed_entry["content"] = f"processed:{entry['content']}"
        return processed_entry

    dp.entry_processor = dummy_entry_processor
    return dp

def test_process_success(dummy_pipeline):
    """
    Test that process correctly processes each entry and returns the modified list.
    """
    entries = [
        {
            "id": "1",
            "title": "Title One",
            "summary": "Summary One",
            "authors": ["Jimmy"],
            "published": "2025-03-26",
            "paper_link": "https://example.com/1",
            "content": "Content One"
        },
        {
            "id": "2",
            "title": "Title Two",
            "summary": "Summary Two",
            "authors": ["Kendra"],
            "published": "2025-03-26",
            "paper_link": "https://example.com/2",
            "content": "Content Two"
        }
    ]
    processed = dummy_pipeline.process(entries)
    # Check that each entry was processed.
    assert processed[0]["title"] == "processed:Title One"
    assert processed[0]["summary"] == "processed:Summary One"
    assert processed[0]["content"] == "processed:Content One"
    assert processed[1]["title"] == "processed:Title Two"

def test_process_with_error(dummy_pipeline):
    """
    Test that process continues processing even if one entry fails.
    The entry with "fail": True should trigger an exception and be replaced by its error-handling output.
    """
    entries = [
        {
            "id": "1",
            "title": "Title One",
            "summary": "Summary One",
            "authors": ["Jimmy"],
            "published": "2025-03-26",
            "paper_link": "https://example.com/1",
            "content": "Content One"
        },
        {
            "id": "2",
            "title": "Title Two",
            "summary": "Summary Two",
            "authors": ["Kendra"],
            "published": "2025-03-26",
            "paper_link": "https://example.com/2",
            "content": "Content Two",
            "fail": True  # This entry will raise an exception in the dummy processor.
        }
    ]
    processed = dummy_pipeline.process(entries)
    # The first entry should be processed normally.
    assert processed[0]["title"] == "processed:Title One"
    # The second entry, which failed, should remain unchanged (or follow our error handling logic)
    # Here, our process() catches the exception and leaves the entry as-is.
    assert processed[1] == entries[1]
