import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from backend.src.RAG.memory import get_by_session_id
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mongodb import MongoDBChatMessageHistory
from dotenv import load_dotenv
from .memory import Memory
load_dotenv()
class QueryResponder:
    """
    Class that responds to a user query by combining the context and user query and
    passing it to an LLM model for a context-aware response.    
    """
    def __init__(self, openai_api_key:str,session_id:str):
        self.memory = Memory()
        self.session_id  = session_id
        
        answer_prompt_template = """
        You are a helpful research assistant. 
        Please provide a concise, well-structured answer **and include direct quotes or references** from the provided context. 
        Use the format [Source: link] (link will be given to you with every paper right after word source).
        Below are relevant excepts from academic papers:
        {context}
        
        The user has asked the following question:
        {question}

        make as many points as necessary and make sure the answer is in numbered boullet points and ensure you conver as many papers 
        as possible 

        if the question is not clear ask questions to understand it better
        If there is not context available DO NOT PROVIDE ANY LINKS OR INFORMATION and do not answer any questions UNLESSits a general question like
        hi, hello, my name is Raghav etc.
        """
        query_template = ChatPromptTemplate([
            MessagesPlaceholder(variable_name="history"),
            ("system", answer_prompt_template)
        ])

        self.temperature=temperature
        self.max_tokens=max_tokens
        self.top_p=top_p
        self.frequency_penalty=frequency_penalty
        self.presence_penalty=presence_penalty
        
        # Initialize the ChatOpenAI model with the given hyperparameters
        self.model = ChatOpenAI(
            model="gpt-4o-mini",  # Use the appropriate model
            api_key=openai_api_key,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
        )
        
        self.qa_chain = RunnableWithMessageHistory(
            chain,
            self.memory.get_session_query_responder,
            input_messages_key="question",
            history_messages_key="history"
        )

    def generate_answer(self, retrieved_docs: List[str], user_query: str) -> str:
        """
        Generates an answer based on the retrieved documents and user query.
        """
        if len(retrieved_docs) == 0:
            answer = self.qa_chain.invoke({"context": "", "question": user_query}, config={"configurable": {"session_id": self.session_id}}).content
            return answer

        formatted_content = self.format_documents(retrieved_docs)
        prompt = self.combine_context_and_question(formatted_content, user_query)
        answer = self.qa_chain.invoke(prompt, config={"configurable": {"session_id": self.session_id}}).content
        return answer

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
        if len(retrieved_docs) == 0:
            formatted_content = ""
        else:
            formatted_content = self.format_documents(retrieved_docs)
        prompt = self.combine_context_and_question(context_text=formatted_content, user_query=user_query)
        answer = self.qa_chain.invoke(prompt,config={"configurable":{"session_id":self.session_id}}).content
        return answer

