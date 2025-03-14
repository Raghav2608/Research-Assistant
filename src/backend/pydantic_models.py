from pydantic import BaseModel

class ResearchPaperQuery(BaseModel):
    message: str # E.g., "Are there any recent advancements in transformer models?"

class LLMInferenceQuery(BaseModel):
    prompt: str