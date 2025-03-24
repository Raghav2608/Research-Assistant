import uvicorn
import logging
import os

from fastapi import FastAPI, HTTPException, Body, Depends, Request
from fastapi.responses import JSONResponse
from fastapi import status
from backend.src.backend.pydantic_models import LLMInferenceQuery
from backend.src.constants import ENDPOINT_URLS
from backend.src.RAG.query_responder import QueryResponder
from backend.src.backend.user_authentication.utils import validate_request,verify_token
from dotenv import load_dotenv
import httpx
import traceback

app = FastAPI()
logger = logging.getLogger('uvicorn.error')
load_dotenv()

if "OPENAI_API_KEY" not in os.environ:
    raise EnvironmentError("openai key not set in environment.")

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise EnvironmentError("openai key not set in environment.")
query_responder = QueryResponder(openai_api_key=OPENAI_API_KEY,session_id="foo")
logger.info(query_responder)

@app.post(
        ENDPOINT_URLS['llm_inference']['path'], 
        description="Handles LLM inference.",
        dependencies=[Depends(validate_request)]
        )
async def llm_inference(request:Request,inference_request:LLMInferenceQuery=Body(...)) -> JSONResponse:
    """
    Handles passing the user query along with any additional context to the LLM model
    for a context-aware response.

    Args:
        inference_request (LLMInferenceQuery): The request containing the user query
                                               and retrieved documents.
    """
    try:
        answer = "Successfully called LLM inference pipeline"
        logger.info(answer)
    
        payload = verify_token(request)
        username = payload.get("user_id")
        if not username:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token."
            )
        
        query_responder.session_id = username
        logger.info(query_responder.session_id)
        responses = inference_request.responses
        user_query = inference_request.user_query
        final_answer = query_responder.generate_answer(
                                                        retrieved_docs=responses, 
                                                        user_query=user_query
                                                        ) # Use original user query
        logger.info(final_answer)
        return JSONResponse(content={"answer": final_answer}, status_code=status.HTTP_200_OK)
    except Exception as e:
        logger.error(f"Error in llm_inference: {traceback.format_exc()}") 
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("app_llm_inference:app", host="localhost", port=8003, reload=True)
