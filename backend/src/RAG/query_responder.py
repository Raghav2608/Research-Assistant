import os

from typing import List, Dict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryMemory, CombinedMemory
from langchain_core.runnables.history import RunnableWithMessageHistory
from .memory import get_by_session_id
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class QueryResponder:
    """
    Class that responds to a user query by combining the context and user query and
    passing it to an LLM model for a context-aware response.    
    """
    def __init__(self, openai_api_key:str,session_id:str):
        os.environ["USER_AGENT"] = "myagent" # Always set a user agent
        
        self.session_id  = session_id
        
        answer_prompt_template = """
        You are a helpful research assistant. 
        Please provide a concise, well-structured answer **and include direct quotes or references** from the provided context. 
        Use the format [Source: link] (link will be given to you with every paper right after word source).
        Below are relevant excepts from academic papers:
        {context}
        The user has asked the following question:
        {question}

        If you find no context then just answer general user queries. if the user query is information specific ask for context
        """
        query_template = ChatPromptTemplate([
            MessagesPlaceholder(variable_name="history"),
            ("system", answer_prompt_template)
            # Equivalently:
            # MessagesPlaceholder(variable_name="conversation", optional=True)
        ])
        
        chain = query_template | ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
        
        self.qa_chain = RunnableWithMessageHistory(
            chain,
            get_by_session_id,
            input_messages_key="question",
            history_messages_key="history"
            )

    def format_documents(self, retrieved_docs:List[str]) -> str:
        """
        Formats the retrieved documents into a single string for the LLM model.

        Args:
            retrieved_docs (List[str]): A list of retrieved documents.
        """
        formatted_content = "\n\n".join(
                                f"Source: {doc['metadata']['link']}\nContent: {doc['page_content']}"
                                for doc in retrieved_docs
                                )
        return formatted_content
    
    def combine_context_and_question(self, context_text:str, user_query:str) -> Dict[str, str]:
        """
        Combines the context and user query into a single dictionary for the LLM model.

        Args:
            context_text (str): The context text to be passed to the LLM model.
            user_query (str): The user query to be passed to the LLM model.
        """
        return {"context": context_text, "question": user_query}
    
    def generate_answer(self, retrieved_docs:List[str], user_query:str) -> str:
        """
        Generates an answer based on the retrieved documents and user query by
        prompting the LLM model.

        Args:
            retrieved_docs (List[str]): A list of retrieved documents.
            user_query (str): The user query.
        """
        if len(retrieved_docs) == 0 or retrieved_docs is None or retrieved_docs[0] == "NONE":
            answer = self.qa_chain.invoke({"context":"","question":user_query},config={"configurable":{"session_id":self.session_id}}).content
            return answer


        formatted_content = self.format_documents(retrieved_docs)
        prompt = self.combine_context_and_question(formatted_content, user_query)
        answer = self.qa_chain.invoke(prompt,config={"configurable":{"session_id":self.session_id}}).content
        return answer