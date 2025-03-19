from src.data_ingestion.arxiv.utils import fetch_arxiv_papers, parse_papers
from typing import Dict, Any, List

class ArXivDataIngestionPipeline:
    """
    The data ingestion pipeline for fetching papers using the arXiv API.
    """
    def fetch_entries(self, topic:str, max_results:int=4) -> List[Dict[str, Any]]:
        """
        Fetches papers from the arXiv API for a given topic.

        Args:
            topic (str): The topic to fetch papers for.
            max_results (int): The maximum number of results to return.
        """
        search_query = f"all:{topic}"
        start = 0
        xml_papers = fetch_arxiv_papers(search_query, start, max_results)
        entries = parse_papers(xml_papers)
        return entries