import pytest
from backend.src.data_ingestion.arxiv.arxiv_pipeline import ArXivDataIngestionPipeline

"""
This test file contains:
  - A success test (fetch_entries_success) verifying that the pipeline constructs the query correctly 
    and returns the expected entries.
  - An empty response test (fetch_entries_empty) ensuring that when no entries are parsed, the pipeline returns [].
  - A parameterized test (fetch_entries_parameterized) running with different topics and max_results values,
    verifying that the underlying fetch function is called exactly once and that the output matches the expected entries.
"""

# Dummy values for testing
dummy_xml = "<feed><entry><id>dummy1</id></entry></feed>"
dummy_entries = [
    {
        "id": "dummy1",
        "title": "Godzilla",
        "summary": "Big lizard",
        "authors": ["Author One"],
        "published": "2020",
        "pdf_link": "dummy1",
    }
]

def test_fetch_entries_success(monkeypatch):
    """
    Test that fetch_entries correctly constructs the query and returns dummy entries.
    """
    # Fake function to simulate fetching XML from arXiv.
    def fake_fetch_arxiv_papers(search_query, start, max_results):
        # Verify that the search query is correctly constructed.
        assert search_query == "all:machine learning"
        assert start == 0
        assert max_results == 4
        return dummy_xml
    
    # Fake function to simulate parsing the XML.
    def fake_parse_papers(xml_string):
        # Verify that the XML passed in is the dummy XML.
        assert xml_string == dummy_xml
        return dummy_entries
    
    # Patch the functions in the pipeline module where they are used.
    from backend.src.data_ingestion.arxiv import arxiv_pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "fetch_arxiv_papers", fake_fetch_arxiv_papers)
    monkeypatch.setattr(pipeline_module, "parse_papers", fake_parse_papers)
    
    pipeline = ArXivDataIngestionPipeline()
    entries = pipeline.fetch_entries("machine learning", max_results=4)
    assert entries == dummy_entries

def test_fetch_entries_empty(monkeypatch):
    """
    Test that if parse_papers returns an empty list, fetch_entries returns an empty list.
    """
    def fake_fetch_arxiv_papers(search_query, start, max_results):
        return dummy_xml
    
    def fake_parse_papers(xml_string):
        return []  # Simulate no valid entries being parsed.
    
    from backend.src.data_ingestion.arxiv import arxiv_pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "fetch_arxiv_papers", fake_fetch_arxiv_papers)
    monkeypatch.setattr(pipeline_module, "parse_papers", fake_parse_papers)
    
    pipeline = ArXivDataIngestionPipeline()
    entries = pipeline.fetch_entries("quantum computing", max_results=4)
    assert entries == []

@pytest.mark.parametrize("topic, max_results, expected_xml", [
    ("machine learning", 4, dummy_xml),
    ("quantum physics", 8, dummy_xml),
])
def test_fetch_entries_parameterized(monkeypatch, topic, max_results, expected_xml):
    """
    Parameterized test to check that fetch_entries passes the correct parameters to the underlying functions
    and returns the expected entries.
    """
    call_counter = {"count": 0}
    
    def fake_fetch_arxiv_papers(search_query, start, max_results_arg):
        call_counter["count"] += 1
        # Ensure that start is always 0.
        assert start == 0
        # We ignore max_results_arg here (in a real scenario, pagination might be applied).
        return expected_xml
    
    def fake_parse_papers(xml_string):
        assert xml_string == expected_xml
        return dummy_entries
    
    from backend.src.data_ingestion.arxiv import arxiv_pipeline as pipeline_module
    monkeypatch.setattr(pipeline_module, "fetch_arxiv_papers", fake_fetch_arxiv_papers)
    monkeypatch.setattr(pipeline_module, "parse_papers", fake_parse_papers)
    
    pipeline = ArXivDataIngestionPipeline()
    entries = pipeline.fetch_entries(topic, max_results=max_results)
    # Verify that our fake fetch function was called exactly once.
    assert call_counter["count"] == 1
    assert entries == dummy_entries
