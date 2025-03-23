import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.runnables.history import RunnableWithMessageHistory
from backend.src.RAG.memory import get_by_session_id
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


        
class QueryResponder:
    def __init__(self, openai_api_key: str, session_id: str, temperature: float = 0.7, max_tokens: int = 100, top_p: float = 1.0, frequency_penalty: float = 0.0, presence_penalty: float = 0.0):
        """
        Initialize the QueryResponder with the given hyperparameters.
        
        Args:
            openai_api_key (str): The OpenAI API key.
            session_id (str): A unique session ID for tracking.
            temperature (float): Sampling temperature for the model.
            max_tokens (int): Maximum number of tokens to generate.
            top_p (float): Nucleus sampling parameter.
            frequency_penalty (float): Penalty for frequent tokens.
            presence_penalty (float): Penalty for new tokens.
        """
        os.environ["USER_AGENT"] = "myagent"  # Always set a user agent
        
        self.session_id = session_id
        
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
        ])
        
        # Initialize the ChatOpenAI model with the given hyperparameters
        self.model = ChatOpenAI(
            model="gpt-4o-mini",  # Use the appropriate model
            api_key=openai_api_key,
            self.temperature=temperature,
            self.max_tokens=max_tokens,
            self.top_p=top_p,
            self.frequency_penalty=frequency_penalty,
            self.presence_penalty=presence_penalty,
        )
        
        self.qa_chain = RunnableWithMessageHistory(
            query_template | self.model,
            get_by_session_id,
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
    
