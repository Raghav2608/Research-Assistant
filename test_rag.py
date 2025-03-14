import getpass
import os
import logging

from dotenv import load_dotenv

from src.RAG.RAG_mqr import RAG

if __name__ == "__main__":

    # To see the query generated
    logging.basicConfig()
    logging.getLogger('langchain.retrievers.multi_query').setLevel(logging.INFO)
    load_dotenv()
    os.environ['USER_AGENT'] = 'myagent'
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

    rag = RAG()
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "end"]:
            break
        final_answer = rag.answer_with_rag(user_input)
        print("=== Final Answer ===")
        print("Researcher: ",final_answer)



