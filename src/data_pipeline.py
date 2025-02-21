from src.data_ingestion.arxiv.topic_extractor import TopicExtractor
from src.data_processing.pipeline import DataProcessingPipeline
from src.data_ingestion.arxiv.pipeline import ArXivDataIngestionPipeline

class DataPipeline:

    def __init__(self):
        self.topic_extractor = TopicExtractor()
        self.data_processing_pipeline = DataProcessingPipeline()

        # ADD DATA INGESTION PIPELINES HERE:
        self.arxiv_data_ingestion_pipeline = ArXivDataIngestionPipeline()
        #########################################
        #########################################
        #########################################

    def run(self, user_query:str):
        all_entries = []

        # Fetch entries from all data ingestion pipelines
        topic = self.topic_extractor([user_query])[0]
        arxiv_entries = self.arxiv_data_ingestion_pipeline.fetch_entries(topic=topic, max_results=4)

        # Add entries from all data ingestion pipelines into a single list
        all_entries.extend(arxiv_entries)

        # ADD MORE DATA INGESTION PIPELINES HERE:
        #########################################
        #########################################
        #########################################

        # Process all entries
        all_entries = self.data_processing_pipeline.process(all_entries)
        return all_entries