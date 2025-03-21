from pydantic import BaseModel

class ResearchPaperQuery(BaseModel):
    user_query: str # E.g., "Are there any recent advancements in transformer models?"

class DataIngestionQuery(BaseModel):
    user_queries: list[str] # E.g., "Are there any recent advancements in transformer models?"

class LLMInferenceQuery(BaseModel):
    user_query: str # E.g., "Are there any recent advancements in transformer models?"
    responses: list # List of dictionaries containing retrieved documents