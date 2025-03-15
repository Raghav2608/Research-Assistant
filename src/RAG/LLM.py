import os

from typing import List

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document

class QueryResponder:
    """
    Class that responds to a user query by combining the context and user query and
    passing it to an LLM model for a context-aware response.    
    """
    def __init__(self, openai_api_key:str):
        os.environ["USER_AGENT"] = "myagent" # Always set a user agent
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=openai_api_key)

    def generate_answer(self, docs:List[Document], user_query):
        raise NotImplementedError()

        # Limit the number of docs if you have many
        top_docs = docs[:10]  # take top 5 if needed

        # Combine text into a single context string
        context_text = "\n\n".join(doc.page_content for doc in top_docs)

        # Run the final LLM chain
        print("=== CONTEXT TEXT ===")
        print(context_text)
        print("=== END CONTEXT ===")

        combined_query = self.combine_context_and_question(context_text, user_query)
        answer = self.qa_chain.run(combined_query)

        return 