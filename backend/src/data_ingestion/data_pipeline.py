from typing import List, Dict, Any

from backend.src.data_processing.pipeline import DataProcessingPipeline
from backend.src.data_ingestion.arxiv.pipeline import ArXivDataIngestionPipeline

class DataPipeline:
    """
    The main data pipeline class that orchestrates the data ingestion and processing pipelines.
    This class is responsible for fetching and processing data from various sources.
    """

    def __init__(self, max_total_entries:int=10, min_entries_per_query:int=3):
        """
        Initialises the data pipeline with the specified parameters.

        Args:
            max_total_entries (int): The maximum total number of entries to fetch given a list of user queries.
            min_entries_per_query (int): The minimum number of entries to fetch from each user query.
        """
        self.data_processing_pipeline = DataProcessingPipeline()

        # ADD DATA INGESTION PIPELINES HERE:
        self.arxiv_data_ingestion_pipeline = ArXivDataIngestionPipeline()
        #########################################
        #########################################
        #########################################
        self.max_total_entries = max_total_entries
        self.min_entries_per_query = min_entries_per_query

    def run(self, user_queries:List[str]) -> List[Dict[str, Any]]:
        all_entries = []

        # Fetch entries from all data ingestion pipelines
        for query in user_queries:
            remaining_entries_left = self.max_total_entries - len(all_entries)
            arxiv_entries = self.arxiv_data_ingestion_pipeline.fetch_entries(
                                                                            topic=query, 
                                                                            max_results=min(
                                                                                            self.min_entries_per_query, 
                                                                                            remaining_entries_left
                                                                                            ) + 1 # +1 as this is non-inclusive
                                                                            )

            # ADD MORE DATA INGESTION PIPELINES HERE:
            #########################################
            #########################################
            #########################################

            # Add entries from all data ingestion pipelines into a single list
            for entry in arxiv_entries:
                if len(all_entries) >= self.max_total_entries:
                    break
                all_entries.append(entry)

        # Process all entries
        all_entries = self.data_processing_pipeline.process(all_entries)
        return all_entries