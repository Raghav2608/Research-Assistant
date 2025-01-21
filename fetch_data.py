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

def summarise_papers(entries:List[Dict[str, Any]]) -> List[str]:
    """
    Summarises a list of paper entries into a list of strings 
    containing the key information about each paper.
    """
    summarising_strings = []
    for paper in entries:
        paper_string = ""
        paper_string += f"ID: {paper['id']}\n"
        paper_string += f"Title: {paper['title']}\n"
        paper_string += f"Summary: {paper['summary']}\n"
        paper_string += f"Authors: {', '.join(paper['authors'])}\n"
        paper_string += f"Published: {paper['published']}\n"
        paper_string += f"PDF Link: {paper['pdf_link']}\n"
        summarising_strings.append(paper_string)
    return summarising_strings

if __name__ == "__main__":

    search_query = "all:attention"
    start = 0
    max_results = 3

    xml_papers = fetch_arxiv_papers(search_query, start, max_results)
    entries = parse_papers(xml_papers)
    summarising_strings = summarise_papers(entries)
    for i, res in enumerate(summarising_strings):
        print(f"Paper: {i+1}")
        print(res)
        print("\n")