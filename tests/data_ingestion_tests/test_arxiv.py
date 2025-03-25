import urllib.request
import requests
import pdfplumber
from io import BytesIO
from typing import List, Dict, Any
import pytest

"""
This file contains comprehensive tests for the arXiv utilities in the data ingestion module.
It verifies the following functions:

1. fetch_arxiv_papers:
   - Ensures the URL is built correctly with the proper query parameters.
   - Returns the expected XML string from the arXiv API (simulated via monkeypatching).

2. fetch_and_extract_pdf_content:
   - Tests successful PDF content extraction by faking a 200 response and simulating PDF parsing.
   - Checks that a non-200 HTTP response results in an empty string after error handling.
   - Verifies that network errors raise an exception.
   - Simulates PDF parsing errors and confirms that the function returns an empty string.

3. parse_papers:
   - Confirms that valid XML with one entry is parsed into a list containing a single dictionary.
   - Converts an author element given as a dict into a list.
   - Returns an empty list when provided with malformed XML.
"""

from backend.src.data_ingestion.arxiv.utils import (
    fetch_arxiv_papers,
    fetch_and_extract_pdf_content,
    parse_papers,
)

# Tests for fetch_arxiv_papers
def test_fetch_arxiv_papers(monkeypatch):
    dummy_xml = "<feed><entry><id>1</id></entry></feed>"

    class DummyResponse:
        def __init__(self, data):
            self.data = data
        def read(self):
            return self.data.encode("utf-8")

    def fake_urlopen(url):
        # Check that the URL contains expected query parameters.
        assert "search_query=all:attention" in url
        assert "start=0" in url
        assert "max_results=5" in url
        return DummyResponse(dummy_xml)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    result = fetch_arxiv_papers("all:attention", 0, 5)
    assert result == dummy_xml

# Tests for fetch_and_extract_pdf_content
def test_fetch_and_extract_pdf_content_success(monkeypatch):
    # Dummy PDF binary content
    dummy_pdf_content = b"%PDF-1.4 dummy content"
    
    class DummyResponse:
        def __init__(self, status_code, content):
            self.status_code = status_code
            self.content = content

    def fake_requests_get(url):
        return DummyResponse(200, dummy_pdf_content)
    
    monkeypatch.setattr(requests, "get", fake_requests_get)
    
    # Create a fake pdfplumber.open that returns an object with one page.
    class DummyPage:
        def extract_words(self):
            return [{"text": "Hello"}, {"text": "World"}]
    
    class DummyPDF:
        pages = [DummyPage()]
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def fake_pdfplumber_open(file_obj):
        return DummyPDF()
    
    monkeypatch.setattr(pdfplumber, "open", fake_pdfplumber_open)
    
    result = fetch_and_extract_pdf_content("http://dummy.com/dummy.pdf")
    # Expect the extracted text to be "Hello World\n"
    expected = "Hello World\n"
    assert result == expected

def test_fetch_and_extract_pdf_content_non200(monkeypatch):
    """
    Test that when the PDF fetch returns a non-200 status,
    the function raises the exception internally and then catches it, 
    resulting in an empty string.
    """
    class DummyResponse:
        def __init__(self, status_code):
            self.status_code = status_code
            self.content = b""
    def fake_requests_get(url):
        return DummyResponse(404)
    monkeypatch.setattr(requests, "get", fake_requests_get)
    result = fetch_and_extract_pdf_content("http://dummy.com/dummy.pdf")
    # The raised exception is caught so the function returns "".
    assert result == ""

def test_fetch_and_extract_pdf_content_network_exception(monkeypatch):
    def fake_requests_get(url):
        raise requests.exceptions.RequestException("network error")
    monkeypatch.setattr(requests, "get", fake_requests_get)
    with pytest.raises(Exception, match="Failed to download PDF"):
        fetch_and_extract_pdf_content("http://dummy.com/dummy.pdf")

def test_fetch_and_extract_pdf_content_pdf_error(monkeypatch):
    # Simulate a successful download but a failure during PDF parsing.
    class DummyResponse:
        def __init__(self, status_code, content):
            self.status_code = status_code
            self.content = content
    def fake_requests_get(url):
        return DummyResponse(200, b"dummy content")
    monkeypatch.setattr(requests, "get", fake_requests_get)
    
    def fake_pdfplumber_open(file_obj):
        raise Exception("PDF parsing error")
    monkeypatch.setattr(pdfplumber, "open", fake_pdfplumber_open)
    
    # In case of an error in processing the PDF, the function returns an empty string.
    result = fetch_and_extract_pdf_content("http://dummy.com/dummy.pdf")
    assert result == ""

# Tests for parse_papers
def test_parse_papers_single_entry():
    # A dummy XML string representing one paper.
    dummy_xml = """
    <feed>
        <entry>
            <id>http://arxiv.org/abs/1234.5678v1</id>
            <title>Sample Paper Title</title>
            <summary> This is a sample summary. </summary>
            <author>
                <name>Author One</name>
            </author>
            <link rel="alternate" type="text/html" href="http://arxiv.org/abs/1234.5678v1"/>
            <link title="pdf" href="http://arxiv.org/pdf/1234.5678v1"/>
            <published>2021-01-01T00:00:00Z</published>
        </entry>
    </feed>
    """
    results = parse_papers(dummy_xml)
    assert isinstance(results, list)
    # Now we expect one valid entry.
    assert len(results) == 1
    entry = results[0]
    expected_keys = ["id", "title", "summary", "authors", "published", "pdf_link"]
    for key in expected_keys:
        assert key in entry
    assert isinstance(entry["authors"], list)
    # Ensure the summary is stripped.
    assert entry["summary"] == "This is a sample summary."

def test_parse_papers_author_dict():
    # Test where the author element is a dict (not a list) and should be converted.
    dummy_xml = """
    <feed>
        <entry>
            <id>http://arxiv.org/abs/9876.5432v1</id>
            <title>Another Sample Paper</title>
            <summary> Another sample summary. </summary>
            <author>
                <name>Author Two</name>
            </author>
            <link rel="alternate" type="text/html" href="http://arxiv.org/abs/9876.5432v1"/>
            <link title="pdf" href="http://arxiv.org/pdf/9876.5432v1"/>
            <published>2022-02-02T00:00:00Z</published>
        </entry>
    </feed>
    """
    results = parse_papers(dummy_xml)
    assert isinstance(results, list)
    assert len(results) == 1
    entry = results[0]
    assert isinstance(entry["authors"], list)
    assert entry["authors"] == ["Author Two"]

def test_parse_papers_malformed():
    # Test with an invalid XML string.
    dummy_xml = "Not a valid XML"
    results = parse_papers(dummy_xml)
    # Expect an empty list because the parsing fails.
    assert results == []
