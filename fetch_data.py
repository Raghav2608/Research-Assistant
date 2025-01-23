"""
Small script for testing the data ingestion pipeline for arXiv research papers.
"""

from src.data_ingestion.arxiv.utils import fetch_arxiv_papers, parse_papers, summarise_papers
from src.data_processing.entry_processor import EntryProcessor

if __name__ == "__main__":

    search_query = "all:transformer"
    start = 0
    max_results = 3

    xml_papers = fetch_arxiv_papers(search_query, start, max_results)
    entries = parse_papers(xml_papers)
    # summarising_strings = summarise_papers(entries_content)

    entry_processor = EntryProcessor()

    for i, entry in enumerate(entries):
        print(f"Paper: {i+1}")
        paper_content = entry["content"]
        print(paper_content)
        print("Number of characters (before processing):", len(paper_content))
        processed_entry = entry_processor(entry)

        processed_content = entry["content"]
        print(processed_content)
        print("Number of characters (after processing):", len(processed_content))
        print("\n")