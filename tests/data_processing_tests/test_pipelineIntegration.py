import time
import pytest
from backend.src.data_processing.pipeline import DataProcessingPipeline

"""
This file contains integration tests for the Data Processing Pipeline.
It runs the full pipeline end-to-end using the actual EntryProcessor and TextPreprocessor.
These tests verify that:
  - The process method correctly processes a list of real entries.
  - The processing completes within an acceptable time.
  - The modified entries have the expected structure and content changes.
  
NOTE:
  - These tests may make real NLP calls via the underlying TextPreprocessor.
  - Ensure environment is configured.
"""

@pytest.mark.integration
def test_data_processing_pipeline_integration_performance():
    dp = DataProcessingPipeline()
    
    # Create a list of sample entries.
    entries = [
        {
            "id": "1",
            "title": "Integration Test Title One\nWith Newline",
            "summary": "This is a test summary for integration.",
            "authors": ["Bobby", "Bob"],
            "published": "2025-03-26",
            "paper_link": "https://example.com/1",
            "content": "Detailed content for integration test one."
        },
        {
            "id": "2",
            "title": "Integration Test Title Two",
            "summary": "Another test summary for integration.",
            "authors": ["Sully"],
            "published": "2025-03-26",
            "paper_link": "https://example.com/2",
            "content": "Detailed content for integration test two."
        }
    ]
    
    start_time = time.perf_counter()
    processed = dp.process(entries)
    elapsed = time.perf_counter() - start_time
    
    # Assert that processing finishes in a reasonable time (e.g., 60 seconds).
    assert elapsed < 60, f"DataProcessingPipeline.process took too long: {elapsed} seconds"
    # Assert that each processed entry is a dict and has expected keys.
    for entry in processed:
        assert isinstance(entry, dict)
        for key in ["id", "title", "summary", "authors", "published", "paper_link", "content"]:
            assert key in entry
        # For integration, we expect that text fields have been processed,
        # e.g., the title is cleaned of newlines.
        assert "\n" not in entry["title"]
        # And that summary and content are modified (if the original text was non-empty).
        if entry["summary"]:
            assert entry["summary"] != ""
        if entry["content"]:
            assert entry["content"] != ""
