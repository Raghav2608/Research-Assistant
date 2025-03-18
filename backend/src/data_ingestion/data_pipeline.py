from typing import List, Dict, Any

from backend.src.data_processing.pipeline import DataProcessingPipeline
from backend.src.data_ingestion.arxiv.pipeline import ArXivDataIngestionPipeline

class DataPipeline:
    """
    The main data pipeline class that orchestrates the data ingestion and processing pipelines.
    This class is responsible for fetching and processing data from various sources.
    """

    def __init__(self):
        self.data_processing_pipeline = DataProcessingPipeline()

        # ADD DATA INGESTION PIPELINES HERE:
        self.arxiv_data_ingestion_pipeline = ArXivDataIngestionPipeline()
        #########################################
        #########################################
        #########################################

    def run(self, user_queries:List[str]) -> List[Dict[str, Any]]:
        all_entries = []

        # Fetch entries from all data ingestion pipelines
        for query in user_queries:
            arxiv_entries = self.arxiv_data_ingestion_pipeline.fetch_entries(topic=query, max_results=4)

            # ADD MORE DATA INGESTION PIPELINES HERE:
            #########################################
            #########################################
            #########################################

            # Add entries from all data ingestion pipelines into a single list
            all_entries.extend(arxiv_entries)
        
        # Process all entries
        all_entries = self.data_processing_pipeline.process(all_entries)
        return all_entries