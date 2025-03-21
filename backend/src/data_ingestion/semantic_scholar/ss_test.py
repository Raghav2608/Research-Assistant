from utils_ss import fetch_all_semantic_scholar_papers, parse_semantic_scholar_papers
from dotenv import load_dotenv
import os

def main():
    # Load environment variables from the .env file
    load_dotenv()
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if not api_key:
        raise Exception("SEMANTIC_SCHOLAR_API_KEY not found in environment variables.")

    search_query = "Penguins"
    limit = 50         # Page size: number of papers to fetch per API call
    max_results = 100  # Total maximum number of papers to fetch via pagination
    desired_total = 20 # Final number of papers to process (e.g., 10 open access and 10 popular)

    try:
        # Fetch papers using pagination
        semantic_json = fetch_all_semantic_scholar_papers(search_query, limit, max_results, api_key=api_key)
        
        # Parse and filter the fetched data
        papers = parse_semantic_scholar_papers(semantic_json, desired_total)
        
        print("Fetched and parsed Semantic Scholar papers:")
        for i, paper in enumerate(papers, start=1):
            print(f"\nPaper {i}:")
            print(f"ID: {paper['id']}")
            print(f"Title: {paper['title']}")
            print(f"Published: {paper['published']}")
            print(f"PDF Link: {paper['pdf_link']}")
            print(f"Citation Count: {paper['citationCount']}")
            print(f"Influential Citation Count: {paper['influentialCitationCount']}")
            print("Authors:", ", ".join(paper['authors']))
            print("Summary:", paper['summary'][:200], "...")
    except Exception as e:
        print(f"An error occurred during the test: {e}")

if __name__ == "__main__":
    main()
