import urllib
import urllib.request
import xmltodict
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

def parse_papers(papers_string:str) -> List[Dict[str, Any]]:
    """
    Converts the XML result from the arXiv API into a list of dictionaries
    containing the data from each paper.

    Args:
        papers_string (str): The XML string from the arXiv API.
    """
    result = xmltodict.parse(papers_string)
    entries = []
    for entry in result["feed"]["entry"]:
        paper_data = {
            "id": entry["id"],
            "title": entry["title"],
            "summary": entry["summary"].strip(),
            "authors": [author["name"] for author in entry["author"]],
            "published": entry["published"],
            "pdf_link": next(link["@href"] for link in entry["link"] if link.get("@title") == "pdf")
        }
        entries.append(paper_data)
    return entries

if __name__ == "__main__":

    search_query = "all:attention"
    start = 0
    max_results = 3

    xml_papers = fetch_arxiv_papers(search_query, start, max_results)
    entries = parse_papers(xml_papers)
    
    for paper in entries:
        print("ID:", paper["id"])
        print("Title:", paper["title"])
        print("Summary:", paper["summary"])
        print("Authors:", paper["authors"])
        print("Published:", paper["published"])
        print("PDF Link:", paper["pdf_link"])
        print("\n")