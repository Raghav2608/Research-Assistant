import os
import pytest
from backend.src.data_ingestion.semantic_scholar. utils_ss import (
    fetch_semantic_scholar_papers,
    fetch_all_semantic_scholar_papers,
    parse_semantic_scholar_papers,
)

"""
This file contains integration tests for the Semantic Scholar utilities 
in the data ingestion module (utils_ss). These tests require a valid Semantic Scholar 
API key (retrieved via a fixture) and are skipped if the key is not provided.

The tests verify the following functions:
1. fetch_semantic_scholar_papers_integration:
   - Calls the real API using a common search query ("machine learning")
   - Verifies that the returned JSON has the expected structure and required fields.

2. fetch_all_semantic_scholar_papers_integration:
   - Tests the pagination logic by fetching a limited number of results (using a limit of 3 per page)
   - Ensures that the total number of papers fetched does not exceed the specified maximum.

3. parse_semantic_scholar_papers_integration:
   - Fetches a real API response with a query ("deep learning")
   - Parses the response and verifies that each parsed paper contains all required keys.
"""

# Fixture to retrieve the API key from an environment variable.
@pytest.fixture(scope="module")
def semantic_api_key():
    key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if key is None:
        pytest.skip("No Semantic Scholar API key provided. Skipping integration tests.")
    return key

@pytest.mark.integration
def test_fetch_semantic_scholar_papers_integration(semantic_api_key):
    """
    Integration test for fetch_semantic_scholar_papers.
    This test calls the real API using a common search query and verifies the structure
    of the returned JSON.
    """
    search_query = "machine learning"
    offset = 0
    limit = 5
    result = fetch_semantic_scholar_papers(search_query, offset, limit, api_key=semantic_api_key)
    
    # Basic structure assertions
    assert isinstance(result, dict)
    assert "data" in result
    if result["data"]:
        sample = result["data"][0]
        for field in [
            "paperId",
            "title",
            "abstract",
            "authors",
            "year",
            "citationCount",
            "influentialCitationCount",
            "openAccessPdf"
        ]:
            assert field in sample

@pytest.mark.integration
def test_fetch_all_semantic_scholar_papers_integration(semantic_api_key):
    """
    Integration test for fetch_all_semantic_scholar_papers.
    Tests pagination by fetching at most 6 results with a limit of 3 per page.
    """
    search_query = "artificial intelligence"
    limit = 3
    max_results = 6
    result = fetch_all_semantic_scholar_papers(search_query, limit, max_results, api_key=semantic_api_key)
    
    assert isinstance(result, dict)
    assert "data" in result
    # The total number of fetched papers should be less than or equal to max_results.
    assert len(result["data"]) <= max_results

@pytest.mark.integration
def test_parse_semantic_scholar_papers_integration(semantic_api_key):
    """
    Integration test for parse_semantic_scholar_papers.
    This test fetches a real API response and then parses it, verifying that each parsed
    paper contains the required keys.
    """
    search_query = "deep learning"
    offset = 0
    limit = 10
    raw_response = fetch_semantic_scholar_papers(search_query, offset, limit, api_key=semantic_api_key)
    
    # Use a desired_total that is reasonable for the number of returned papers.
    desired_total = 5
    parsed = parse_semantic_scholar_papers(raw_response, desired_total)
    
    assert isinstance(parsed, list)
    # Since the API might return fewer than desired_total papers, we check <= desired_total.
    assert len(parsed) <= desired_total
    
    required_keys = [
        "id",
        "title",
        "summary",
        "authors",
        "published",
        "paper_link",
        "citationCount",
        "influentialCitationCount",
    ]
    for paper in parsed:
        for key in required_keys:
            assert key in paper
