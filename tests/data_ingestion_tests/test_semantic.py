import pytest

from backend.src.data_ingestion.semantic_scholar.utils_ss import (
    fetch_semantic_scholar_papers,
    fetch_all_semantic_scholar_papers,
    parse_semantic_scholar_papers,
)

"""
This file contains comprehensive tests for the Semantic Scholar utilities 
in the data ingestion module (utils_ss). The tests use monkeypatching to simulate 
HTTP responses and cover the following functions:

1. fetch_semantic_scholar_papers:
   - Verifies that a 200 HTTP response returns the expected JSON.
   - Tests retry logic when a 504 response is initially received and then a success.
   - Confirms that an unexpected HTTP status (e.g. 400) immediately raises an Exception.
   - Checks that after max retries with repeated 504 responses, an Exception is raised.

2. fetch_all_semantic_scholar_papers:
   - An integration test that simulates pagination: one page with data and one empty page to break the loop.
   - Verifies that the aggregated results match the expected output.

3. parse_semantic_scholar_papers:
   - Parameterized tests ensuring that the parsing function returns the expected number of papers.
   - Checks that each parsed paper includes the required keys and correct data.
"""

# Dummy response class to simulate requests responses.
class DummyResponse:
    def __init__(self, status_code, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data


# Fixtures for sample API responses and JSON
@pytest.fixture
def sample_success_response():
    """Fixture for a successful API JSON response."""
    return {
        "data": [
            {
                "paperId": "1",
                "title": "Test Paper",
                "abstract": "Optimus Prime is the leader of the autobots",
                "authors": [{"name": "Author"}],
                "year": 2025,
                "citationCount": 15,
                "influentialCitationCount": 5,
                "openAccessPdf": {"url": "http://example.com/pdf"},
            }
        ]
    }

@pytest.fixture
def sample_semantic_json():
    """Fixture for a sample semantic scholar JSON structure with multiple papers."""
    return {
        "data": [
            {
                "paperId": "1",
                "title": "Paper One",
                "abstract": "Abstract One",
                "authors": [{"name": "Author A"}],
                "year": 2022,
                "citationCount": 10,
                "influentialCitationCount": 5,
                "openAccessPdf": {"url": "http://example.com/1.pdf"},
            },
            {
                "paperId": "2",
                "title": "Paper Two",
                "abstract": None,
                "authors": [{"name": "Author B"}],
                "year": 2019,
                "citationCount": 20,
                "influentialCitationCount": 15,
                "openAccessPdf": None,
            },
            {
                "paperId": "3",
                "title": "Paper Three",
                "abstract": "Abstract Three",
                "authors": [{"name": "Author C"}],
                "year": 2018,
                "citationCount": 5,
                "influentialCitationCount": 2,
                "openAccessPdf": {"url": "http://example.com/3.pdf"},
            },
            {
                "paperId": "4",
                "title": "Paper Four",
                "abstract": "Abstract Four",
                "authors": [{"name": "Author D"}],
                "year": 2017,
                "citationCount": 50,
                "influentialCitationCount": 25,
                "openAccessPdf": None,
            },
        ]
    }


# Tests for fetch_semantic_scholar_papers
def test_fetch_semantic_scholar_papers_success(monkeypatch, sample_success_response):
    """Test that a 200 response returns the expected JSON."""
    def fake_get(url, headers):
        return DummyResponse(200, sample_success_response)

    monkeypatch.setattr("backend.src.data_ingestion.semantic_scholar.utils_ss.requests.get", fake_get)
    result = fetch_semantic_scholar_papers("test query", offset=0, limit=50)
    assert result == sample_success_response

def test_fetch_semantic_scholar_papers_retry_success(monkeypatch, sample_success_response):
    """
    Test that the function handles a 504 error on the first call and then succeeds.
    We override time.sleep to avoid actual delays.
    """
    call_count = {"count": 0}

    def fake_get(url, headers):
        if call_count["count"] < 1:
            call_count["count"] += 1
            return DummyResponse(504)
        else:
            return DummyResponse(200, sample_success_response)

    monkeypatch.setattr("backend.src.data_ingestion.semantic_scholar.utils_ss.requests.get", fake_get)
    monkeypatch.setattr("backend.src.data_ingestion.semantic_scholar.utils_ss.time.sleep", lambda x: None)
    result = fetch_semantic_scholar_papers("test query", offset=0, limit=50)
    assert result == sample_success_response

def test_fetch_semantic_scholar_papers_error(monkeypatch):
    """
    Test that an unexpected HTTP status (e.g., 400) immediately raises an Exception.
    """
    def fake_get(url, headers):
        return DummyResponse(400)

    monkeypatch.setattr("backend.src.data_ingestion.semantic_scholar.utils_ss.requests.get", fake_get)
    with pytest.raises(Exception, match="Error fetching Semantic Scholar papers: 400"):
        fetch_semantic_scholar_papers("test query", offset=0, limit=50)

def test_fetch_semantic_scholar_papers_max_retries(monkeypatch):
    """
    Test that when the API keeps returning a 504, the function eventually raises an Exception
    after max retries.
    """
    def fake_get(url, headers):
        return DummyResponse(504)

    monkeypatch.setattr("backend.src.data_ingestion.semantic_scholar.utils_ss.requests.get", fake_get)
    monkeypatch.setattr("backend.src.data_ingestion.semantic_scholar.utils_ss.time.sleep", lambda x: None)
    with pytest.raises(Exception, match="Max retries exceeded. Please try again later."):
        fetch_semantic_scholar_papers("test query", offset=0, limit=50)


# Test for fetch_all_semantic_scholar_papers (marked as integration)
@pytest.mark.integration
def test_fetch_all_semantic_scholar_papers(monkeypatch, sample_success_response):
    """
    Test the pagination logic in fetch_all_semantic_scholar_papers by simulating
    two pages: one with data and one empty to break the loop.
    """
    call_count = {"count": 0}

    def fake_fetch(search_query, offset, limit, api_key=None):
        if call_count["count"] == 0:
            call_count["count"] += 1
            return sample_success_response
        else:
            return {"data": []}

    monkeypatch.setattr("backend.src.data_ingestion.semantic_scholar.utils_ss.fetch_semantic_scholar_papers", fake_fetch)
    result = fetch_all_semantic_scholar_papers("test query", limit=50, max_results=100)
    assert result == {"data": sample_success_response["data"]}


# Parameterized tests for parse_semantic_scholar_papers
@pytest.mark.parametrize("desired_total, expected_count", [
    (2, 2),
    (3, 3),
    (5, 4)  
])
def test_parse_semantic_scholar_papers(sample_semantic_json, desired_total, expected_count):
    """
    Test that parse_semantic_scholar_papers returns the expected number of parsed papers,
    and that each parsed paper contains the required fields.
    """
    parsed = parse_semantic_scholar_papers(sample_semantic_json, desired_total)
    assert isinstance(parsed, list)
    assert len(parsed) == expected_count
    for paper in parsed:
        # Ensure every parsed paper has the required keys.
        for key in [
            "id",
            "title",
            "summary",
            "authors",
            "published",
            "paper_link",
            "citationCount",
            "influentialCitationCount",
        ]:
            assert key in paper
