import urllib
import requests
import time
from typing import List, Dict, Any

def fetch_semantic_scholar_papers(search_query: str, offset: int, limit: int, api_key: str = None) -> dict:
    """
    Fetches papers from the Semantic Scholar API with search query, offset, and limit.
    Implements exponential backoff in case of gateway timeout (HTTP 504).
    Omits the citation fields (citations.paperId and citations.title).

    Args:
        search_query (str): The search query to use.
        offset (int): The offset (i.e. index of the first result to return).
        limit (int): The number of results to return (e.g., 50).
        api_key (str, optional): Your Semantic Scholar API key.

    Returns:
        dict: The JSON response from the API.
    """
    encoded_query = urllib.parse.quote(search_query)
    url = (
        f"https://api.semanticscholar.org/graph/v1/paper/search?"
        f"query={encoded_query}&offset={offset}&limit={limit}"
        f"&fields=paperId,title,abstract,authors,year,"
        f"citationCount,influentialCitationCount,openAccessPdf"
    )
    
    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    max_retries = 5
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        elif response.status_code in (429, 504):
            wait_time = 2 ** attempt
            print(f"Received status code {response.status_code}. Waiting for {wait_time} seconds (attempt {attempt+1}/{max_retries})...")
            time.sleep(wait_time)
        else:
            raise Exception(f"Error fetching Semantic Scholar papers: {response.status_code}")
    raise Exception("Max retries exceeded. Please try again later.")

def fetch_all_semantic_scholar_papers(search_query: str, limit: int, max_results: int, api_key: str = None) -> dict:
    """
    Fetches papers from the Semantic Scholar API using pagination. It repeatedly calls
    fetch_semantic_scholar_papers until max_results is reached or no more results are returned.

    Args:
        search_query (str): The search query to use.
        limit (int): The number of results per API call (page size).
        max_results (int): The maximum total number of results to fetch.
        api_key (str, optional): Your Semantic Scholar API key.

    Returns:
        dict: A dictionary with a "data" key containing a list of all fetched papers.
    """
    offset = 0
    results = []
    while offset < max_results:
        response = fetch_semantic_scholar_papers(search_query, offset, limit, api_key=api_key)
        data = response.get("data", [])
        if not data:
            break
        results.extend(data)
        offset += limit
    return {"data": results}

def parse_semantic_scholar_papers(semantic_json: dict, desired_total: int) -> List[Dict[str, Any]]:
    """
    Converts the JSON result from the Semantic Scholar API into a list of dictionaries.
    Filters final papers so half are open access (with a valid PDF URL)
    and half are selected based on popularity (citationCount + influentialCitationCount).

    The final dictionary for each paper contains:
      - "id"
      - "title"
      - "summary"
      - "authors"
      - "published"
      - "paper_link"
      - "citationCount"
      - "influentialCitationCount"
    
    Args:
        semantic_json (dict): The JSON result from the Semantic Scholar API.
        desired_total (int): The total number of papers to return.

    Returns:
        List[Dict[str, Any]]: A list of parsed paper dictionaries.
    """
    papers = semantic_json.get("data", [])
    
    def popularity_score(paper):
        return paper.get("citationCount", 0) + paper.get("influentialCitationCount", 0)
    
    # Filter for papers with an open access PDF.
    open_access_papers = [paper for paper in papers if paper.get("openAccessPdf") and paper["openAccessPdf"].get("url")]
    
    desired_open_access = desired_total // 2
    desired_popular = desired_total - desired_open_access
    
    selected_open_access = open_access_papers[:desired_open_access]
    
    # For popular papers, sort by popularity score (descending) and skip duplicates.
    all_papers_sorted = sorted(papers, key=popularity_score, reverse=True)
    selected_popular = []
    selected_ids = {paper.get("paperId") for paper in selected_open_access}
    
    for paper in all_papers_sorted:
        if paper.get("paperId") in selected_ids:
            continue
        selected_popular.append(paper)
        if len(selected_popular) >= desired_popular:
            break
    
    final_selection = selected_open_access + selected_popular
    
    # Fills with additional papers if not enough selected.
    if len(final_selection) < desired_total:
        all_selected_ids = {paper.get("paperId") for paper in final_selection}
        for paper in all_papers_sorted:
            if paper.get("paperId") not in all_selected_ids:
                final_selection.append(paper)
            if len(final_selection) >= desired_total:
                break
    
    parsed_papers = []
    for paper in final_selection:

        scholar_link = f"https://www.semanticscholar.org/paper/{paper.get('paperId')}"
        
        paper_data = {
            "id": paper.get("paperId"),
            "title": paper.get("title"),
            # Handles for cases where abstract is None
            "summary": (paper.get("abstract") or "").strip(),
            "authors": [author.get("name", "") for author in paper.get("authors", [])],
            "published": str(paper.get("year", "")),
            "paper_link": scholar_link,
            "citationCount": paper.get("citationCount", 0),
            "influentialCitationCount": paper.get("influentialCitationCount", 0)
        }
        parsed_papers.append(paper_data)
    
    return parsed_papers
