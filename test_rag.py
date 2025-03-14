import os
import logging

from dotenv import load_dotenv

from src.RAG.RAG_mqr import RAG
from src.data_pipeline import DataPipeline
from src.RAG.query_generator import ResearchQueryGenerator

if __name__ == "__main__":

    # To see the query generated
    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise EnvironmentError("openai key not set in environment.")

    data_pipeline = DataPipeline()
    query_generator = ResearchQueryGenerator(openai_api_key=OPENAI_API_KEY)
    rag = RAG(openai_api_key=OPENAI_API_KEY)
    while True:
        user_input = input("User: ")
        
        if user_input.lower() in ["exit", "quit", "end"]:
            break

        # Generate additional queries
        additional_queries = query_generator.generate(user_input)
        print(additional_queries)
        additional_queries.append(user_input)

        # Retrieve documents from the data pipeline
        all_entries = []
        for query in additional_queries:
            print(query)
            entries = data_pipeline.run(query)
            all_entries.extend(entries)

        print("Total number of retrieved entries from data ingestion: ", len(all_entries))

        # Pass to LLM (UNFINISHED)
        final_answer = rag.answer_with_rag(all_entries)

        print("=== Final Answer ===")
        print("Researcher: ",final_answer)