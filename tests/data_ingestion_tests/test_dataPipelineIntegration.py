import time
import pytest
from backend.src.data_ingestion.data_pipeline import DataPipeline

"""
This file contains an end-to-end integration test for the DataPipeline class 
in the data ingestion module. The test performs the following functions:
s
1. Runs the complete data pipeline using real ingestion and processing pipelines.
2. Measures the execution time of the pipeline to ensure it completes within an acceptable threshold (60 seconds).
3. Verifies that the output is structured correctly (i.e., a non-empty list of entries, each containing at least a "title" field).

NOTE:
  - This test makes real API calls and depends on external services.
  - Ensure that the environment is configured with necessary API keys.
  - The test is marked as an integration test and can be run selectively.
"""

@pytest.mark.integration
def test_data_pipeline_integration_performance():
    """
    End-to-end integration test for DataPipeline.
    Runs the full pipeline with real ingestion and processing pipelines,
    measures execution time, and verifies the overall output structure.
    
    NOTE:
      - This test may make real API calls and may be subject to network delays.
      - Ensure your environment is configured with any necessary API keys.
    """
    pipeline = DataPipeline(max_total_entries=5, min_entries_per_query=2)
    user_queries = ["machine learning", "quantum computing"]

    start_time = time.time()
    try:
        results = pipeline.run(user_queries)
    except Exception as e:
        pytest.skip(f"Integration test skipped due to external service error: {e}")
    elapsed_time = time.time() - start_time

    # Check that the pipeline completes within 60 seconds.
    assert elapsed_time < 60, f"DataPipeline.run took too long: {elapsed_time} seconds"
    assert isinstance(results, list)
    assert len(results) > 0
    for entry in results:
        # Check that each processed entry contains a title.
        assert "title" in entry
