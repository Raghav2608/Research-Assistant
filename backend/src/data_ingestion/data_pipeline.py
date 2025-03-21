from typing import List, Dict, Any

from backend.src.data_ingestion.arxiv.topic_extractor import TopicExtractor
from backend.src.data_processing.pipeline import DataProcessingPipeline
from backend.src.data_ingestion.arxiv.arxiv_pipeline import ArXivDataIngestionPipeline
from backend.src.data_ingestion.semantic_scholar.ss_pipeline import SSDataIngestionPipeline

class DataPipeline:
    """
    The main data pipeline class that orchestrates the data ingestion and processing pipelines.
    This class is responsible for fetching and processing data from various sources.
    """

    def __init__(self):
        self.topic_extractor = TopicExtractor()
        self.data_processing_pipeline = DataProcessingPipeline()

        # ADD DATA INGESTION PIPELINES HERE:
        self.arxiv_data_ingestion_pipeline = ArXivDataIngestionPipeline()
        self.ss_data_ingestion_pipeline = SSDataIngestionPipeline()
        #########################################
        #########################################

    def run(self, user_query:str) -> List[Dict[str, Any]]:
        all_entries = []

        # Fetch entries from all data ingestion pipelines
        topic = self.topic_extractor([user_query])[0]
        arxiv_entries = self.arxiv_data_ingestion_pipeline.fetch_entries(topic=topic, max_results=4)
        ss_entries = self.ss_data_ingestion_pipeline.get_entries(topic=topic, max_results=100, desired_total=20)
        # Add entries from all data ingestion pipelines into a single list
        all_entries.extend(arxiv_entries)
        all_entries.extend(ss_entries)
        # ADD MORE DATA INGESTION PIPELINES HERE:
        #########################################
        #########################################
        #########################################

        # Process all entries
        all_entries = self.data_processing_pipeline.process(all_entries)
        return all_entries