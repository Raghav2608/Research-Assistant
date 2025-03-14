import uvicorn
import logging

from fastapi import FastAPI, HTTPException
from src.backend.pydantic_models import LLMInferenceQuery
from src.constants import ENDPOINT_URLS

app = FastAPI()
logger = logging.getLogger('uvicorn.error')

@app.post(ENDPOINT_URLS['llm_inference']['path'], description="Handles LLM inference.")
async def llm_inference(inference_request:LLMInferenceQuery):
    try:
        answer = "Successfully called LLM inference pipeline"
        logger.info(answer)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("app_llm_inference:app", host="localhost", port=8003, reload=True)
