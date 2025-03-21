from src.data_ingestion.semantic_scholar.utils_ss import fetch_all_semantic_scholar_papers, parse_semantic_scholar_papers
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

class SSDataIngestionPipeline:
    """
    The data ingestion pipeline for fetching papers using the Semantic Scholar API.
    """
    def get_entries(self, topic: str, max_results: int = 100, desired_total: int = 20) -> List[Dict[str, Any]]:
        """
        Fetches papers from the Semantic Scholar API for a given topic.
        
        Args:
            topic (str): The topic to fetch papers for.
            max_results (int): The maximum total number of papers to fetch via pagination.
            desired_total (int): The final number of papers to return.
            
        Returns:
            List[Dict[str, Any]]: A list of paper dictionaries.
        """
        # Load environment variables from the .env file.
        load_dotenv()
        api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        if not api_key:
            raise Exception("SEMANTIC_SCHOLAR_API_KEY not found in environment variables.")

        # Use the topic as the search query; adjust as needed.
        search_query = topic
        # Set the page size for each API call.
        limit = 50
        # Fetch papers using pagination.
        semantic_json = fetch_all_semantic_scholar_papers(search_query, limit, max_results, api_key=api_key)
        # Parse and filter the fetched data.
        entries = parse_semantic_scholar_papers(semantic_json, desired_total)
        return entries
