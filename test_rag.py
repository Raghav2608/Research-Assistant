import os
import logging

from dotenv import load_dotenv

from src.RAG.RAG_mqr import RAG
from src.data_pipeline import DataPipeline
from src.RAG.query_generator import ResearchQueryGenerator
from src.RAG.LLM import QueryResponder

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
    query_responder = QueryResponder(openai_api_key=OPENAI_API_KEY)

    while True:
        user_input = input("User: ")
        
        if user_input.lower() in ["exit", "quit", "end"]:
            break

        # Generate additional queries
        additional_queries = query_generator.generate(user_input)
        print(additional_queries)
        additional_queries.append(user_input)

        # Attempt to retrieve documents the existing database
        text_response = rag.retrieve(user_query=user_input)

        # Attempt to retrieve documents via data ingestion
        if not text_response:
            print("No relevant documents found, searching for more documents")

            all_entries = []
            for query in additional_queries:
                print(query)
                entries = data_pipeline.run(query)
                all_entries.extend(entries)
            print("Total number of retrieved entries from data ingestion: ", len(all_entries))

            if len(all_entries) == 0:
                print("No entries could be found for this query, please try to rephrase your query.")
            else:
                docs = rag.convert_entries_to_docs(entries=all_entries)
                rag.split_and_add_documents(docs=docs) # Add documents to ChromaDB (save)

                # Attempt to retrieve the documents again
                second_response = rag.retrieve(user_query=user_input)

                print("Second response:", second_response)

        # Pass to LLM
        final_answer = query_responder.generate_answer(docs=all_entries, user_query=user_input)

        print("=== Final Answer ===")
        print("Researcher: ",final_answer)