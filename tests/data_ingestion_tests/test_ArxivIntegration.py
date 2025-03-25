import pytest

from backend.src.data_ingestion.arxiv.utils import (
    fetch_arxiv_papers,
    fetch_and_extract_pdf_content,
    parse_papers,
)

"""
This file contains integration tests for the arXiv utilities 
in the data ingestion module (utils). These tests make live calls 
to the arXiv API and, in one case, download a sample PDF, so they 
should be run selectively (e.g., using the "integration" marker).

The tests verify the following functions:
1. fetch_arxiv_papers_integration:
   - Calls the real arXiv API using a common search query (e.g., "all:deep learning")
   - Verifies that the returned XML string contains the <feed> element.

2. fetch_and_extract_pdf_content_integration:
   - Downloads a real PDF from arXiv and extracts text from it.
   - Checks that the extracted text is a non-empty string.

3. parse_papers_integration:
   - Fetches a real XML response from arXiv, parses it, and verifies that each 
     parsed paper contains all required keys.
"""

@pytest.mark.integration
def test_fetch_arxiv_papers_integration():
    """
    Integration test for fetch_arxiv_papers.
    This test calls the real arXiv API using a common search query ("all:deep+learning")
    and verifies that the returned XML contains a <feed> element.
    """
    search_query = "all:deep+learning"  # '+' instead of space
    offset = 0
    max_results = 5
    result = fetch_arxiv_papers(search_query, offset, max_results)
    
    assert isinstance(result, str)
    assert "<feed" in result  # More flexible than checking for exact start

@pytest.mark.integration
def test_fetch_and_extract_pdf_content_integration():
    """
    Integration test for fetch_and_extract_pdf_content.
    This test downloads a sample PDF from arXiv and extracts its text.
    It verifies that the extracted text is a non-empty string.

    NOTE: This test depends on network access and the availability of the PDF.
    """
    pdf_url = "https://arxiv.org/pdf/2401.00317.pdf"  # Uses a stable arXiv paper on nuclear physics from 2024
    text = fetch_and_extract_pdf_content(pdf_url)
    assert isinstance(text, str)
    assert text.strip() != ""

@pytest.mark.integration
def test_parse_papers_integration():
    """
    Integration test for parse_papers.
    This test fetches a real XML response from arXiv using a common search query 
    ("all:machine+learning"), parses the XML, and verifies that each parsed paper 
    contains all required keys.
    """
    search_query = "all:machine+learning"
    offset = 0
    max_results = 5
    xml_response = fetch_arxiv_papers(search_query, offset, max_results)

    entries = parse_papers(xml_response)
    assert isinstance(entries, list)
    assert len(entries) > 0

    required_keys = ["id", "title", "summary", "authors", "published", "pdf_link"]
    for entry in entries:
        for key in required_keys:
            assert key in entry
