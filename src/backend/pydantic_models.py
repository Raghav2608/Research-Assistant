from pydantic import BaseModel

class ResearchPaperQuery(BaseModel):
    message: str

class LLMInferenceQuery(BaseModel):
    prompt: str