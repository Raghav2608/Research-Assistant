import json
from langchain_openai import ChatOpenAI

class ResearchQueryGenerator:
    """
    A class to generate multiple variations of a research query while handling edge cases.
    """
    def __init__(self, openai_api_key:str):
        self.llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)

    def generate(self, user_prompt:str):
        """Generate multiple variations of a research query while handling edge cases.
        Returns a JSON list of possible queries or an error message."""

        system_prompt = f"""
        You are a research assistant specializing in refining user queries for recent research retrieval or retrieval based on given papers.
        
        Your ONLY output should be valid JSON with no extra text, in the format:
        ["query_variation_1", "query_variation_2", "query_variation_3"]

        If you cannot produce a valid list, return:
        ["ERROR: Failed to generate valid queries. Please try again 2."]


        **Your Responsibilities:**
        1. If the query is **too broad** (e.g., "AI"), make it more specific.
        2. If the query is **ambiguous** (e.g., "bias"), provide different possible meanings.
        3. If the query is **too narrow**, generalize it slightly while keeping it relevant.
        4. If the query is **invalid** (too short, gibberish), return: `"I don't understand. ERROR: Invalid query. Please provide more details."`

        **User Query:** "{user_prompt}"
        """

        generated_query = self.llm.invoke(system_prompt).content

        try:
            query_variations = json.loads(generated_query)
        except json.JSONDecodeError:
            return ["ERROR: Failed to generate valid queries. Please try again."]
        return query_variations