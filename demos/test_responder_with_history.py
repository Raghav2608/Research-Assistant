from backend.src.RAG.query_responder import QueryResponder
import os
import logging

from dotenv import load_dotenv
from backend.src.RAG.memory import store



if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise EnvironmentError("openai key not set in environment.")


    query_generator = QueryResponder(openai_api_key=OPENAI_API_KEY)
    
    while True:
        user_input = input("User: ")
        
        if user_input.lower() in ["exit", "quit", "end"]:
            break

        queries = query_generator.generate_answer(None,user_input)
        print(store)
        print(queries)