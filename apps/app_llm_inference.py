import uvicorn
import logging
import os

from fastapi import FastAPI, HTTPException, Body
from src.backend.pydantic_models import LLMInferenceQuery
from src.constants import ENDPOINT_URLS
from src.RAG.query_responder import QueryResponder

app = FastAPI()
logger = logging.getLogger('uvicorn.error')

if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("openai key not set in environment.")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise EnvironmentError("openai key not set in environment.")
query_responder = QueryResponder(openai_api_key=OPENAI_API_KEY)

@app.post(ENDPOINT_URLS['llm_inference']['path'], description="Handles LLM inference.")
async def llm_inference(inference_request:LLMInferenceQuery=Body(...)):
    try:
        answer = "Successfully called LLM inference pipeline"
        logger.info(answer)
        responses = inference_request.responses
        user_query = inference_request.user_query
        final_answer = query_responder.generate_answer(
                                                        retrieved_docs=responses, 
                                                        user_query=user_query
                                                        ) # Use original user query
        return {"answer": final_answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("app_llm_inference:app", host="localhost", port=8003, reload=True)
