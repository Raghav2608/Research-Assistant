"""
Small script for testing the data ingestion pipeline for arXiv research papers.
"""

from src.data_ingestion.arxiv.utils import fetch_arxiv_papers, parse_papers, summarise_papers
from src.data_processing.text_preprocessor import TextPreprocessor

if __name__ == "__main__":

    search_query = "all:attention"
    start = 0
    max_results = 3

    xml_papers = fetch_arxiv_papers(search_query, start, max_results)
    entries = parse_papers(xml_papers)
    summarising_strings = summarise_papers(entries)

    text_preprocessor = TextPreprocessor()

    for i, res in enumerate(summarising_strings):
        print(f"Paper: {i+1}")
        print(res)
        print("Number of characters (before processing):", len(res))
        print("Number of characters (after processing):", len(text_preprocessor(res)))
        print("\n")