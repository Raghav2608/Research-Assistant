import urllib
import urllib.request
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
    url = f'http://export.arxiv.org/api/query?search_query={search_query}&start={start}&max_results={max_results}&sortBy=relevance&sortOrder=descending'
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
    """
    entries = []
    try:
        result = xmltodict.parse(papers_string)
        raw_entries = result["feed"]["entry"] 
        #If there is only one entry, wrap it in a list
        if isinstance(raw_entries, dict):
            raw_entries = [raw_entries]
        for entry in raw_entries:
            print(entry)
            if not isinstance(entry, dict):
                print("Skipping invalid entry")
                continue

            # Preventing StopIteration error
            pdf_link = next((link["@href"] for link in entry["link"] if link.get("@title") == "pdf"), None)

            print(entry["author"])

            # Ensure author is always a list 
            if isinstance(entry["author"], dict):
                entry["author"] = [entry["author"]]

            paper_data = {
                "id": entry["id"],
                "title": entry["title"],
                "summary": entry["summary"].strip(),
                "authors": [author["name"] for author in entry["author"]],
                "published": entry["published"],
                "pdf_link": entry["id"]
            }
            entries.append(paper_data)
    except Exception as e:
        #On an error (i.e. malformed xml), now returns an empty list
        return []
    return entries