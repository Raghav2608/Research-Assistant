import os
import logging

from dotenv import load_dotenv

from src.RAG.RAG_mqr import RAG

if __name__ == "__main__":

    # To see the query generated
    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)

    load_dotenv()
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    if not OPENAI_API_KEY:
        raise EnvironmentError("openai key not set in environment.")

    rag = RAG(openai_api_key=OPENAI_API_KEY)
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "end"]:
            break
        final_answer = rag.answer_with_rag(user_input)
        print("=== Final Answer ===")
        print("Researcher: ",final_answer)