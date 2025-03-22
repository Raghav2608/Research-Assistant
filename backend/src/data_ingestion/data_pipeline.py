from typing import List, Dict, Any
from backend.src.data_processing.pipeline import DataProcessingPipeline
from backend.src.data_ingestion.semantic_scholar.ss_pipeline import SSDataIngestionPipeline
from backend.src.data_ingestion.arxiv.arxiv_pipeline import ArXivDataIngestionPipeline
from backend.src.RAG.utils import clean_search_query

import numpy as np

class DataPipeline:
    """
    The main data pipeline class that orchestrates the data ingestion and processing pipelines.
    This class is responsible for fetching and processing data from various sources.
    """

    def __init__(self, max_total_entries:int=25, min_entries_per_query:int=3):
        """
        Initialises the data pipeline with the specified parameters.

        Args:
            max_total_entries (int): The maximum total number of entries to fetch given a list of user queries.
            min_entries_per_query (int): The minimum number of entries to fetch from each user query.
        """
        self.data_processing_pipeline = DataProcessingPipeline()

        self.arxiv_data_ingestion_pipeline = ArXivDataIngestionPipeline()
        self.ss_data_ingestion_pipeline = SSDataIngestionPipeline()

        self.text_preprocessor = self.data_processing_pipeline.entry_processor.text_preprocessor

        self.max_total_entries = max_total_entries
        self.min_entries_per_query = min_entries_per_query

    def process_query(self, user_query:str) -> str:
        """
        Processes the user query by removing non-alphanumeric characters, stopwords
        and formats the query to be used for fetching data.

        Args:
            user_query (str): The user query to process.
        """
        processed_query = self.text_preprocessor.keep_only_alphanumeric(user_query)
        processed_query = self.text_preprocessor.remove_newlines(processed_query)
        print(processed_query)

        processed_query = self.text_preprocessor.remove_stopwords(processed_query)
        processed_query = self.text_preprocessor.remove_newlines(processed_query)
        print(processed_query)
        
        processed_query = clean_search_query(processed_query)
        print(processed_query)
        return processed_query
    
    def remove_duplicate_entries(self, entries:List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Removes duplicate entries from the list of entries fetched by
        comparing the titles of the entries.

        Args:
            entries (List[Dict[str, Any]]): The list of entries to remove duplicates from.
        """
        known_titles = set()
        unique_entries = []

        for entry in entries:
            if entry["title"] not in known_titles:
                known_titles.add(entry["title"])
                unique_entries.append(entry)
        return entries

    def select_entries(
                        self, 
                        all_arxiv_entries:List[List[Dict[str, Any]]], 
                        all_ss_entries:List[List[Dict[str, Any]]], 
                        num_user_queries
                        ) -> List[Dict[str, Any]]:
        """
        Selects entries from the fetched entries for each user query from all of the data sources.
        
        For each entry for the final list:
        - Choose whether to use the ArXiv or Semantic Scholar entries
        - Choose which entries to use for the corresponding query and data source
        
        Args:
            all_arxiv_entries (List[List[Dict[str, Any]]]): The list of entries fetched from ArXiv for each user query.
            all_ss_entries (List[List[Dict[str, Any]]]): The list of entries fetched from Semantic Scholar for each user query.
            num_user_queries (int): The number of user queries.
        """
        
        # Total entries should be NUM_QUERIES * NUM_ENTRIES_PER_QUERY at most.
        all_entries = []
        for i in range(self.max_total_entries):

            # Choose whether to use the ArXiv or Semantic Scholar entries
            use_arxiv = np.random.choice([True, False])

            # Choose which query to use
            query_idx = np.random.choice(num_user_queries)

            print(i, use_arxiv, query_idx)

            idx = i # Set the i-th entry to fetch

            if use_arxiv:
                # Choose a random entry from the list of entries fetched for the chosen query if the index is out of bounds
                if idx >= len(all_arxiv_entries[query_idx]):
                    idx = np.random.choice(len(all_arxiv_entries[query_idx])) 

                entry = all_arxiv_entries[query_idx][idx] # The i-th entry from the list of entries fetched for the chosen query
            else:
                if idx >= len(all_ss_entries[query_idx]):
                    idx = np.random.choice(len(all_arxiv_entries[query_idx])) 

                entry = all_ss_entries[query_idx][i]
                
            all_entries.append(entry)
        return all_entries
    
    def retrieve_documents(self, user_query:str) -> List[Dict[str, Any]]:
        """
        Retrieves documents from all the data sources for the given user query.
    
        Args:
            user_query (str): The user query to fetch data for.
        """
        # ArXiv fetching
        arxiv_entries = self.arxiv_data_ingestion_pipeline.fetch_entries(
                                                                        topic=user_query, 
                                                                        max_results=self.max_total_entries
                                                                        )
        # Semantic Scholar fetching
        ss_entries = self.ss_data_ingestion_pipeline.get_entries(
                                                                topic=user_query, 
                                                                max_results=self.max_total_entries,
                                                                desired_total=self.max_total_entries
                                                                )   
        return arxiv_entries, ss_entries  


    def run(self, user_queries:List[str]) -> List[Dict[str, Any]]:
        """
        Fetches data from various sources using the user queries and processes
        the data to standardise the structure of the entries.

        Args:
            user_queries (List[str]): The list of user queries to fetch data for.
        """
        all_arxiv_entries = []
        all_ss_entries = []

        # Fetch entries from all data ingestion pipelines for each user query
        for query in user_queries:
            processed_query = self.process_query(query)

            arxiv_entries, ss_entries = self.retrieve_documents(processed_query)

            print("Num fetched from ArXiv:", len(arxiv_entries))
            print("Num fetched from Semantic Scholar:", len(ss_entries))

            all_arxiv_entries.append(arxiv_entries)
            all_ss_entries.append(ss_entries)
        
        selected_entries = self.select_entries(
                                                all_arxiv_entries=all_arxiv_entries, 
                                                all_ss_entries=all_ss_entries, 
                                                num_user_queries=len(user_queries)
                                                )

        unique_entries = self.remove_duplicate_entries(selected_entries)
        
        # Process all entries
        unique_entries = self.data_processing_pipeline.process(unique_entries)
        return unique_entries