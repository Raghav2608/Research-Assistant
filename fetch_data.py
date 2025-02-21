"""
Small script for demonstrating the data ingestion pipeline for arXiv research papers.
"""

from src.data_ingestion.arxiv.utils import fetch_arxiv_papers, parse_papers
from src.data_ingestion.arxiv.topic_extractor import TopicExtractor
from src.data_processing.pipeline import DataProcessingPipeline
from src.data_ingestion.arxiv.pipeline import ArXivDataIngestionPipeline

if __name__ == "__main__":
    
    test_sentence = "Are there any recent advancements in transformer models?"
    topic_extractor = TopicExtractor()
    topic = topic_extractor([test_sentence])[0]
    print(topic)
    
    arxiv_data_ingestion_pipeline = ArXivDataIngestionPipeline()
    entries = arxiv_data_ingestion_pipeline.fetch_entries(topic=topic, max_results=4)
    data_processing_pipeline = DataProcessingPipeline()
    entries = data_processing_pipeline.process(entries)