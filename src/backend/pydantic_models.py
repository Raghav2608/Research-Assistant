from pydantic import BaseModel

class ResearchPaperQuery(BaseModel):
    message: str