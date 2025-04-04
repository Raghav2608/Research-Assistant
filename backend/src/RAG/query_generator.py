import json

from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.runnables.history import RunnableWithMessageHistory
from backend.src.RAG.utils import clean_search_query
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mongodb import MongoDBChatMessageHistory
import os
from dotenv import load_dotenv
from .memory import Memory


load_dotenv()
class ResearchQueryGenerator:
    """
    A class to generate multiple variations of a research query while handling edge cases.
    """
    def __init__(self, openai_api_key:str,session_id:str):
        self.memory = Memory()
        self.session_id = session_id
        self.system_prompt_template = """
        
        You are a research assistant specializing in refining user queries for recent research retrieval or retrieval based on given papers.
        
        Even if someone asks a question like "Are there any recent developments in "topic"" or adjacent questions generate a query on what you 
        think is best
        
        Your ONLY output should be valid JSON with no extra text, in the format:
        ["query_variation_1", "query_variation_2", "query_variation_3"]

        If the question is a follow up question check history and answer the question
        in the same format as above

        If you cannot produce a valid list, return:
        ["ERROR: Failed to generate valid queries. Please try again 2."]


        **Your Responsibilities:**
        1. If the query is **too broad** (e.g., "AI"), make it more specific.
        2. If the query is **ambiguous** (e.g., "bias"), provide different possible meanings.
        3. If the query is **too narrow**, generalize it slightly while keeping it relevant.
        4. If the query is **invalid** (too short, gibberish), return: `"I don't understand. ERROR: Invalid query. Please provide more details."`
        5. If the query is telling you to find "any research" or 'cool research" or anything vague use your creativity to generate search query
        6. use history only when relevant to the current question otherwise don't

        """
        
        query_template = ChatPromptTemplate([
            ("system", self.system_prompt_template),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}")
            # Equivalently:
            # MessagesPlaceholder(variable_name="conversation", optional=True)
        ])
        
        chain = query_template | ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
        
        self.query_chain = RunnableWithMessageHistory(
            runnable=chain,
            get_session_history=self.memory.get_session_query_generator,
            input_messages_key="question",
            history_messages_key="history"
            )

    def generate(self, user_prompt:str) -> List[str]:
        """
        Generates multiple variations of a research query while handling edge cases.
        - Returns a JSON list of possible queries or an error message.
        
        Args:
            user_prompt (str): The user's query.
        """
        generated_query = self.query_chain.invoke({"question":user_prompt},config={"configurable":{"session_id":self.session_id}}).content
        print(generated_query)
        if "error" in generated_query.lower():
            return "ERROR"

        try:
            query_variations = json.loads(generated_query)
            query_variations = [query for query in query_variations]

            print("Q", query_variations)
            if isinstance(query_variations, str) and "ERROR" in query_variations:
            # Return a dict with 'content' and 'artifact' keys
                return query_variations
            
            elif isinstance(query_variations, list):
                # If the entire list is just a single error
                if len(query_variations) == 1 and "ERROR" in query_variations[0]:
                    return query_variations[0]
                else:
                    return query_variations
            else:
                # If we got something weird (not a list, not a string)
                raise ValueError("ERROR: Invalid query generation output.")
        except json.JSONDecodeError:
            return ["ERROR: Failed to generate valid queries. Please try again."]
        except ValueError:
            return ["ERROR: Failed to generate valid queries. Please try again."]