import os
import logging

from dotenv import load_dotenv

from backend.src.RAG.retrieval_engine import RetrievalEngine
from backend.src.data_ingestion.data_pipeline import DataPipeline
from backend.src.RAG.query_generator import ResearchQueryGenerator
from backend.src.RAG.query_responder import QueryResponder

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
    retrieval_engine = RetrievalEngine(openai_api_key=OPENAI_API_KEY)
    query_responder = QueryResponder(openai_api_key=OPENAI_API_KEY)

    while True:
        user_input = input("User: ")
        
        if user_input.lower() in ["exit", "quit", "end"]:
            break

        # Generate additional queries
        additional_queries = query_generator.generate(user_input)
        print(additional_queries)

        if additional_queries == "ERROR":
            final_answer = query_responder.generate_answer(retrieved_docs=None, user_query=user_input)
            print(final_answer)

            # Attempt to retrieve documents via data ingestionexit

        else:
            print("No relevant documents found, searching for more documents")
            entries = data_pipeline.run(additional_queries)
            print("Total number of retrieved entries from data ingestion: ", len(entries))

            if len(entries) == 0:
                print("No entries could be found for this query, please try to rephrase your query.")
            else:
                docs = retrieval_engine.convert_entries_to_docs(entries=entries)
                retrieval_engine.split_and_add_documents(docs=docs) # Add documents to ChromaDB (save)

            # Attempt to retrieve the documents again
            responses = retrieval_engine.retrieve(user_queries=additional_queries)

            print("Responses:", responses)

            # Pass to LLM
            final_answer = query_responder.generate_answer(retrieved_docs=responses, user_query=user_input) # Use original user query

            print("=== Final Answer ===")
            print("Researcher: ", final_answer)