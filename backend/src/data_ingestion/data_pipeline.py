from typing import List, Dict, Any
from backend.src.data_processing.pipeline import DataProcessingPipeline
from backend.src.data_ingestion.semantic_scholar.ss_pipeline import SSDataIngestionPipeline

class DataPipeline:
    """
    The main data pipeline class that orchestrates the data ingestion and processing pipelines.
    This class is responsible for fetching and processing data from various sources.
    """

    def __init__(self):
        self.data_processing_pipeline = DataProcessingPipeline()
        # ADD DATA INGESTION PIPELINES HERE:
        self.ss_data_ingestion_pipeline = SSDataIngestionPipeline()
        #########################################
        #########################################

    def run(self, user_query:str) -> List[Dict[str, Any]]:
        all_entries = []

        ss_entries = self.ss_data_ingestion_pipeline.get_entries(topic=user_query, max_results=20, desired_total=10)
        all_entries.extend(ss_entries)
        # ADD MORE DATA INGESTION PIPELINES HERE:
        #########################################
        #########################################
        #########################################

        # Process all entries
        all_entries = self.data_processing_pipeline.process(all_entries)
        return all_entries