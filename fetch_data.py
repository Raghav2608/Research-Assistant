"""
Small script for demonstrating the data ingestion pipeline for arXiv research papers.
"""

from src.data_ingestion.arxiv.utils import fetch_arxiv_papers, parse_papers
from src.data_ingestion.arxiv.topic_extractor import TopicExtractor
from src.data_processing.pipeline import DataProcessingPipeline

import time

if __name__ == "__main__":
    
    test_sentence = "Are there any recent advancements in transformer models?"
    topic_extractor = TopicExtractor()
    topic = topic_extractor([test_sentence])[0]
    print(topic)

    search_query = f"all:{topic}"
    start = 0
    max_results = 4

    xml_papers = fetch_arxiv_papers(search_query, start, max_results)
    entries = parse_papers(xml_papers)
    data_processing_pipeline = DataProcessingPipeline()
    entries = data_processing_pipeline.process(entries)