import pytest
from backend.src.data_ingestion.semantic_scholar.ss_pipeline import SSDataIngestionPipeline

"""
This file contains comprehensive tests for the Semantic Scholar ingestion pipeline 
(SSDataIngestionPipeline) in the data ingestion module. It verifies the following:

1. Missing API Key:
   - Ensures that get_entries raises an exception when no Semantic Scholar API key is present.
   
2. Successful Data Fetching:
   - Confirms that get_entries correctly calls the underlying fetch and parse functions 
     with the correct parameters and returns the expected parsed entries.
     
3. Empty Response Handling:
   - Tests that if the fetch function returns an empty data list, the pipeline returns an empty list.
   
4. Pagination Behavior (Parameterized Test):
   - Uses different combinations of max_results and desired_total to ensure that the underlying 
     fetch function is called the expected number of times and that the output matches the dummy entries.
"""

dummy_semantic_json = {
    "data": [
        {
            "paperId": "1",
            "title": "The Bee Movie Analysis",
            "abstract": "Do you like jazz?",
            "authors": [{"name": "Quentin Tarantino"}],
            "year": 2020,
            "citationCount": 10,
            "influentialCitationCount": 5,
            "openAccessPdf": {"url": "http://dummy.com/dummy.pdf"},
        }
    ]
}
dummy_entries = [
    {
        "id": "1",
        "title": "The Bee Movie Analysis",
        "summary": "Do you like jazz?",
        "authors": ["Quentin Tarantino"],
        "published": "2020",
        "paper_link": "https://www.semanticscholar.org/paper/1",
        "citationCount": 10,
        "influentialCitationCount": 5,
    }
]

# Automatically set a dummy API key for tests that require one.
@pytest.fixture(autouse=True)
def set_dummy_api_key(monkeypatch):
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "dummy_api_key")

def test_get_entries_raises_exception_if_no_api_key(monkeypatch):
    """
    Test that get_entries raises an exception when the API key is missing.
    """
    # Remove the API key from the environment.
    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    # Override load_dotenv to do nothing.
    monkeypatch.setattr("backend.src.data_ingestion.semantic_scholar.ss_pipeline.load_dotenv", lambda: None)
    
    pipeline = SSDataIngestionPipeline()
    with pytest.raises(Exception, match="SEMANTIC_SCHOLAR_API_KEY not found in environment variables."):
        pipeline.get_entries("machine learning")

def test_get_entries_success(monkeypatch):
    """
    Test that get_entries successfully calls the fetch and parse functions with the
    correct parameters and returns the expected entries.
    """
    from backend.src.data_ingestion.semantic_scholar import ss_pipeline as pipeline_module

    # Fake fetch_all function that simulates a successful API call.
    def fake_fetch_all(search_query, limit, max_results, api_key):
        # Verify that the pipeline passes the expected parameters.
        assert search_query == "machine learning"
        assert limit == 50
        assert max_results == 100
        assert api_key == "dummy_api_key"
        return dummy_semantic_json

    # Fake parse function that returns our dummy entries.
    def fake_parse(semantic_json, desired_total):
        assert semantic_json == dummy_semantic_json
        # desired_total is expected to be 20 in this test.
        assert desired_total == 20
        return dummy_entries

    # Patch the functions in the pipeline module.
    monkeypatch.setattr(pipeline_module, "fetch_all_semantic_scholar_papers", fake_fetch_all)
    monkeypatch.setattr(pipeline_module, "parse_semantic_scholar_papers", fake_parse)

    pipeline_instance = SSDataIngestionPipeline()
    entries = pipeline_instance.get_entries("machine learning", max_results=100, desired_total=20)
    assert entries == dummy_entries

def test_get_entries_empty(monkeypatch):
    """
    Test that if the fetch function returns an empty 'data' list, the pipeline returns an empty list.
    """
    from backend.src.data_ingestion.semantic_scholar import ss_pipeline as pipeline_module

    def fake_fetch_all(search_query, limit, max_results, api_key):
        # Simulate an empty response.
        return {"data": []}

    def fake_parse(semantic_json, desired_total):
        # If there is no data, parse should return an empty list.
        return []

    monkeypatch.setattr(pipeline_module, "fetch_all_semantic_scholar_papers", fake_fetch_all)
    monkeypatch.setattr(pipeline_module, "parse_semantic_scholar_papers", fake_parse)

    pipeline_instance = SSDataIngestionPipeline()
    entries = pipeline_instance.get_entries("quantum physics")
    assert entries == []

@pytest.mark.parametrize("max_results, desired_total, expected_fetch_calls", [
    (50, 10, 1),
    (100, 20, 2),
])
def test_get_entries_parameterized(monkeypatch, max_results, desired_total, expected_fetch_calls):
    """
    Use parameterization to test get_entries with different max_results and desired_total values.
    Count how many times the underlying fetch function is called.
    """
    from backend.src.data_ingestion.semantic_scholar import utils_ss as utils_module
    call_counter = {"count": 0}

    # Fake fetch function that simulates the API call.
    def fake_fetch_all(search_query, limit, max_results_arg, api_key):
        call_counter["count"] += 1
        # On first call, return dummy data; on subsequent calls, return empty data.
        if call_counter["count"] == 1:
            return dummy_semantic_json
        else:
            return {"data": []}

    # Patch the fetch_semantic_scholar_papers function in the utils module.
    monkeypatch.setattr(utils_module, "fetch_semantic_scholar_papers", fake_fetch_all)
    # Patch parse_semantic_scholar_papers to simply return dummy_entries.
    monkeypatch.setattr(utils_module, "parse_semantic_scholar_papers", lambda semantic_json, desired_total_arg: dummy_entries)

    pipeline_instance = SSDataIngestionPipeline()
    entries = pipeline_instance.get_entries("parameterized test", max_results=max_results, desired_total=desired_total)
    # Verify the number of fetch calls.
    assert call_counter["count"] == expected_fetch_calls
    # Verify that the returned entries match dummy_entries.
    assert entries == dummy_entries
