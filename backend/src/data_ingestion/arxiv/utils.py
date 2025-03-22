import urllib
import urllib.request
import urllib.parse
import xmltodict
import requests
import pdfplumber
from io import BytesIO

from typing import List, Dict, Any

def fetch_arxiv_papers(search_query:str, start:int, max_results:int) -> str:
    """
    Fetches papers from the arXiv API, returning the XML result as a string.

    Args:
        search_query (str): The search query to use, e.g., "all:attention".
        start (int): The index of the first result to return.
        max_results (int): The maximum number of results to return.
    """
    encoded_query=urllib.parse.quote(search_query)
    url = f'http://export.arxiv.org/api/query?search_query={encoded_query}&start={start}&max_results={max_results}&sortBy=relevance&sortOrder=descending'
    data = urllib.request.urlopen(url)
    result = data.read().decode("utf-8")
    return result

def fetch_and_extract_pdf_content(pdf_url:str):
    """
    Fetches and extracts the text content from a PDF file.
    
    TODO: Could improve on how text is extracted, the format of the papers
    are not always the same and has a big impact on the quality of the extracted text.
    
    Args:
        pdf_url: The URL of the PDF file to fetch and extract text from.
    """
    try:
        response = requests.get(pdf_url)
        print(pdf_url)
        if response.status_code != 200:
            raise Exception(f"Failed to fetch PDF: {response.status_code}")
    
        pdf_file = BytesIO(response.content)
        pdf = pdfplumber.open(pdf_file)

        paper_text = ""
        for page in pdf.pages:
            words = page.extract_words()
            if words:
                text = " ".join([word["text"] for word in words])
                # print(page.extract_words())
                # print(text)
                paper_text += text + "\n"
        return paper_text
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to download PDF {pdf_url}: {e}")

    except Exception as e:
        print(f"Error processing PDF {pdf_url}: {e}")  
        return "" # So eda.py will not crash


def parse_papers(papers_string: str) -> List[Dict[str, Any]]:
    """
    Converts the XML result from the arXiv API into a list of dictionaries
    containing the data from each paper.

    Args:
        papers_string (str): The XML string from the arXiv API.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a paper.
    """
    try:
        result = xmltodict.parse(papers_string)
        entries = []

        # Handle single entry (dict) vs. multiple entries (list)
        feed_entries = result.get("feed", {}).get("entry", [])
        if isinstance(feed_entries, dict):  # Single entry
            feed_entries = [feed_entries]

        for entry in feed_entries:
            if not isinstance(entry, dict):
                print("Skipping invalid entry")
                continue

            # Extract PDF link
            pdf_link = None
            if "link" in entry:
                links = entry["link"] if isinstance(entry["link"], list) else [entry["link"]]
                pdf_link = next((link["@href"] for link in links if link.get("@title") == "pdf"), None)

            # Extract authors
            authors = []
            if "author" in entry:
                if isinstance(entry["author"], dict):  # Single author
                    authors = [entry["author"]["name"]]
                elif isinstance(entry["author"], list):  # Multiple authors
                    authors = [author["name"] for author in entry["author"]]

            # Extract summary
            summary = entry.get("summary", "").strip()

            # Fetch content safely
            content = ""
            if pdf_link:
                try:
                    content = fetch_and_extract_pdf_content(pdf_link)
                    if not content.strip():  # If PDF has no text, fallback to summary
                        print(f"Warning: No extractable text found in PDF for '{entry.get('title')}'. Using summary instead.")
                        content = summary
                except Exception as e:
                    print(f"Error fetching PDF for '{entry.get('title')}': {e}. Using summary instead.")
                    content = summary
            else:
                print(f"Warning: No PDF found for '{entry.get('title')}'. Using summary instead.")
                content = summary

            # Build paper data dictionary
            paper_data = {
                "id": entry.get("id", "No ID"),
                "title": entry.get("title", "No Title"),
                "summary": summary,
                "authors": authors,
                "published": entry.get("published", "No Date"),
                "pdf_link": pdf_link if pdf_link else "No PDF available",
                "content": content
            }
            entries.append(paper_data)

        return entries

    except Exception as e:
        print(f"Error parsing arXiv API response: {e}")
        return []
    
    # Example usage
if __name__ == "__main__":
    # Fetch papers from arXiv
    search_query = "1510.00726"
    papers_xml = fetch_arxiv_papers(search_query, start=0, max_results=1)

    # Parse papers
    papers = parse_papers(papers_xml)
    for paper in papers:
        print(f"Title: {paper['title']}")
        print(f"Authors: {', '.join(paper['authors'])}")
        print(f"Summary: {paper['summary']}")
        print(f"PDF Link: {paper['pdf_link']}")
        print(f"Content (first 200 chars): {paper['content'][:200]}...")
        print("-" * 80)