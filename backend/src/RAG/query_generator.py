import json

from langchain_openai import ChatOpenAI
from typing import List
from langchain_core.runnables.history import RunnableWithMessageHistory
from backend.src.RAG.utils import clean_search_query
from .memory import get_by_session_id
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class ResearchQueryGenerator:
    """
    A class to generate multiple variations of a research query while handling edge cases.
    """
    def __init__(self, openai_api_key:str):
        self.system_prompt_template = """
        
        You are a research assistant specializing in refining user queries for recent research retrieval or retrieval based on given papers.
        
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
            chain,
            get_by_session_id,
            input_messages_key="question",
            history_messages_key="history"
            )

    def embed_query_in_prompt(self, query:str) -> str:
        """
        Embeds the query using the LLM model.
        
        Args:
            query (str): The query to embed.
        """
        return self.system_prompt_template.format(query=query)

    def generate(self, user_prompt:str) -> List[str]:
        """
        Generates multiple variations of a research query while handling edge cases.
        - Returns a JSON list of possible queries or an error message.
        
        Args:
            user_prompt (str): The user's query.
        """
        generated_query = self.query_chain.invoke({"question":user_prompt},config={"configurable":{"session_id":"foo"}}).content
        print(generated_query)
        if "error" in generated_query.lower():
            return "ERROR"

        try:
            query_variations = json.loads(generated_query)
            query_variations = [clean_search_query(query) for query in query_variations]
            
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