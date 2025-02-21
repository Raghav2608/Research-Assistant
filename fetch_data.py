"""
Small script for demonstrating the data ingestion pipeline for arXiv research papers.
"""

from src.data_ingestion.arxiv.utils import fetch_arxiv_papers, parse_papers
from src.data_ingestion.arxiv.topic_extractor import TopicExtractor
from src.data_processing.entry_processor import EntryProcessor
import time

if __name__ == "__main__":
    
    test_sentence = "Are there any recent advancements in transformer models?"
    topic_extractor = TopicExtractor()
    topic = topic_extractor([test_sentence])[0]
    print(topic)

    search_query = f"all:{topic}"
    start = 0
    max_results = 4

    start_time = time.perf_counter()
    xml_papers = fetch_arxiv_papers(search_query, start, max_results)
    entries = parse_papers(xml_papers)
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

    end_time = time.perf_counter()
    print(f"Time taken: {end_time-start_time:.2f} seconds")