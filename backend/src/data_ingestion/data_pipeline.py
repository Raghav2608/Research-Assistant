from typing import List, Dict, Any
from backend.src.data_processing.pipeline import DataProcessingPipeline
from backend.src.data_ingestion.semantic_scholar.ss_pipeline import SSDataIngestionPipeline
from backend.src.data_ingestion.arxiv.arxiv_pipeline import ArXivDataIngestionPipeline
from backend.src.RAG.utils import clean_search_query

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
        self.ss_data_ingestion_pipeline = SSDataIngestionPipeline()

        #########################################
        #########################################
        #########################################
        self.max_total_entries = max_total_entries
        self.min_entries_per_query = min_entries_per_query

    def run(self, user_queries:List[str]) -> List[Dict[str, Any]]:
        """
        Fetches data from various sources using the user queries and processes
        the data to standardise the structure of the entries.

        Args:
            user_queries (List[str]): The list of user queries to fetch data for.
        """
        all_entries = []

        # Fetch entries from all data ingestion pipelines
        for query in user_queries:
            remaining_entries_left = self.max_total_entries - len(all_entries)
            arxiv_entries = self.arxiv_data_ingestion_pipeline.fetch_entries(
                                                                            topic=clean_search_query(query), 
                                                                            max_results=min(
                                                                                            self.min_entries_per_query, 
                                                                                            remaining_entries_left
                                                                                            ) + 1 # +1 as this is non-inclusive
                                                                            )
            # TODO: Implement cleaning search query for semantic scholar as queries do not work as expected.
            # ss_entries = self.ss_data_ingestion_pipeline.get_entries(
            #                                                         topic=query, 
            #                                                         max_results=self.min_entries_per_query, 
            #                                                         desired_total=remaining_entries_left
            #                                                         )
            # ss_entries = self.ss_data_ingestion_pipeline.get_entries(
            #                                                         topic=query, 
            #                                                         max_results=20, 
            #                                                         desired_total=10
            #                                                         )                                                         

            # ADD MORE DATA INGESTION PIPELINES HERE:
            #########################################
            #########################################


            # Add entries from all data ingestion pipelines into a single list
            for entry in arxiv_entries:
                if len(all_entries) >= self.max_total_entries:
                    break
                all_entries.append(entry)

            # for entry in ss_entries:
            #     if len(all_entries) >= self.max_total_entries:
            #         break
            #     all_entries.append(entry)

        # Process all entries
        all_entries = self.data_processing_pipeline.process(all_entries)
        return all_entries