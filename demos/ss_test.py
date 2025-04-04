"""
Script to test the Semantic Scholar data ingestion pipeline.
"""

import set_path
import os

from dotenv import load_dotenv
from backend.src.data_ingestion.semantic_scholar.ss_pipeline import SSDataIngestionPipeline


def main():
    # Load environment variables from the .env file
    load_dotenv()
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        raise Exception("SEMANTIC_SCHOLAR_API_KEY not found in environment variables.")

    # search_query = "Are+there+are+any+recent+advancements+in+transformer+models" # Doesn't work
    # search_query = "Are there are any recent advancements in transformer models" # Doesn't work
    search_query = "recent+advancements+in+transformer+models" # Works
    max_results = 100  # Total maximum number of papers to fetch via pagination
    desired_total = 20 # Final number of papers to process (e.g., 10 open access and 10 popular)

    ss_pipeline = SSDataIngestionPipeline()
    try:
        # # Fetch papers using pagination6
        # semantic_json = fetch_all_semantic_scholar_papers(search_query, limit, max_results, api_key=api_key)
        
        # # Parse and filter the fetched data
        # papers = parse_semantic_scholar_papers(semantic_json, desired_total)
        papers = ss_pipeline.get_entries(search_query, max_results, desired_total)
        
        print("Fetched and parsed Semantic Scholar papers:")
        for i, paper in enumerate(papers, start=1):
            print(f"\nPaper {i}:")
            print(f"ID: {paper['id']}")
            print(f"Title: {paper['title']}")
            print(f"Published: {paper['published']}")
            print(f"SS Link: {paper['paper_link']}")
            print(f"Citation Count: {paper['citationCount']}")
            print(f"Influential Citation Count: {paper['influentialCitationCount']}")
            print("Authors:", ", ".join(paper['authors']))
            print("Summary:", paper['summary'][:200], "...")
    except Exception as e:
        print(f"An error occurred during the test: {e}")

if __name__ == "__main__":
    main()
