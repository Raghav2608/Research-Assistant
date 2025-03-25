import uvicorn
import requests
import logging

from fastapi import FastAPI, HTTPException, Body, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi import status
from fastapi.middleware.cors import CORSMiddleware
from backend.src.backend.pydantic_models import ResearchPaperQuery
from backend.src.constants import ENDPOINT_URLS
from backend.src.backend.user_authentication.utils import validate_request
from backend.src.backend.user_authentication.authentication_service import UserAuthenticationService
from backend.src.backend.user_authentication.token_manager import verify_token

app = FastAPI(title="Research Assistant API")
logger = logging.getLogger('uvicorn.error')

templates = Jinja2Templates(directory="frontend/templates_temp")

user_authentication_service = UserAuthenticationService(is_testing=True)


# Add CORS middleware to allow requests from the frontend (localhost)
origins = [
            "http://localhost:3000",
            "http://localhost:8080",
            "http://127.0.0.1",
            ]
app.add_middleware(
                    CORSMiddleware,
                    allow_origins=origins,
                    allow_credentials=True, # Allows cookies to be sent to the frontend, so that they can make authenticated requests
                    allow_methods=["GET", "POST", "OPTIONS"],
                    allow_headers=["Content-Type", "Authorization"],
                    )

@app.get(
        ENDPOINT_URLS['web_app']['path'], 
        response_class=HTMLResponse, 
        dependencies=[Depends(validate_request)]
        )
async def root(request:Request) -> HTMLResponse:
    """
    Displays the home page (temp)

    Args:
        request (Request): The request object containing information
                           that can be used/displayed in the template.
    """
    return templates.TemplateResponse(
                                    "chat.html", 
                                    {"request": request}
                                    )

@app.get(ENDPOINT_URLS['web_app']['additional_paths']['login'], response_class=HTMLResponse)
async def login(request:Request) -> HTMLResponse:
    """
    Displays the login page.

    Args:
        request (Request): The request object containing information
                           that can be used/displayed in the template.
    """
    return templates.TemplateResponse("login.html", {"request": request})

@app.get(ENDPOINT_URLS['web_app']['additional_paths']['register'], response_class=HTMLResponse)
async def register(request:Request) -> HTMLResponse:
    """
    Displays the registration page.

    Args:
        request (Request): The request object containing information
                           that can be used/displayed in the template.
    """
    return templates.TemplateResponse("register.html", {"request": request})

@app.get("/whoami", dependencies=[Depends(validate_request)])
async def whoami(request: Request) -> JSONResponse:
    """
    Returns the username of the authenticated user.
    """
    payload = verify_token(request)
    username = payload.get("user_id")
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found in token."
        )
    return JSONResponse(content={"username": username}, status_code=status.HTTP_200_OK)

@app.post(ENDPOINT_URLS['web_app']['additional_paths']['user_authentication'], response_class=JSONResponse)
async def user_authentication(
                            request:Request, 
                            username:str=Body(...), 
                            password:str=Body(...),
                            confirm_password:str=Body(None)
                            ) -> JSONResponse:
    """
    Authenticates the user by checking the username and password provided.
    - If the user is authenticated, a token is generated and set in the cookie.
    - The token is used for making authenticated requests to the rest of the system.

    Args:
        request (Request): The request object containing information
                           that can be used to authenticate the user.
        username (str): The username of the user.
        password (str): The password of the user.

    Returns:
        JSONResponse: A JSON response containing the authentication token if the user is authenticated.
    """
    logger.info(f"User authentication request received for username '{username}' ...")
    is_rate_limited, message = user_authentication_service.handle_rate_limiting(request=request, username=username)
    if is_rate_limited:
        return JSONResponse(content={"message": message}, status_code=status.HTTP_429_TOO_MANY_REQUESTS)
    
    status_code, message = user_authentication_service.handle_authentication(
                                                                            username=username, 
                                                                            password=password, 
                                                                            request=request, 
                                                                            confirm_password=confirm_password,
                                                                            )
    
    if not (status_code == status.HTTP_200_OK or status_code == status.HTTP_201_CREATED):
        return JSONResponse(content={"message": message}, status_code=status_code)
    logger.info("Successfully authenticated user ...")
    return user_authentication_service.get_token_response(username=username, status_code=status_code, message=message)

# Handles research queries.
@app.post(
        ENDPOINT_URLS['web_app']['additional_paths']['query'], 
        summary="Submit a research query", 
        description="Returns an answer generated by the system.",
        dependencies=[Depends(validate_request)]
        )
async def query_system(request:Request, query_request:ResearchPaperQuery=Body(...)) -> JSONResponse:
    """
    Submits the user query to the system and returns the answer generated by the system.
    
    Args:
        request (Request): The request object containing information that can be used to 
                           authenticate the user.
        query_request (ResearchPaperQuery): The user query to be submitted to the system.
    """
    try:
        # Retrieve authorisation token to make authenticated requests
        if request.headers.get("Authorization") is None:
            token = request.cookies.get("token")
            headers = {"Authorization": token}
        else:
            headers = {"Authorization": request.headers.get("Authorization")}

        # Call the retrieval endpoint
        logger.info("Calling retrieval endpoint")
        RETRIEVAL_URL = f"http://{ENDPOINT_URLS['retrieval']['base_url']}{ENDPOINT_URLS['retrieval']['path']}"
        retrieval_response = requests.post(
                                            url=RETRIEVAL_URL, 
                                            json={"user_query": query_request.user_query, "mode": query_request.mode}, 
                                            headers=headers
                                            )
        responses = retrieval_response.json()["responses"]
        logger.info(responses)
        LLM_INFERENCE_URL = f"http://{ENDPOINT_URLS['llm_inference']['base_url']}{ENDPOINT_URLS['llm_inference']['path']}"
        
        if responses == "ERROR":
            logger.info("received unqueriable user response answering generally")
            logger.info("Calling LLM inference endpoint")
            llm_response = requests.post(url=LLM_INFERENCE_URL, json={"user_query": query_request.user_query, "responses":[]},headers=headers)
            logger.info("Successfully called the system.")
            llm_response = llm_response.json()["answer"]
            logger.info(llm_response)
            return {"answer": llm_response,"papers":[]}
        
        else:
            logger.info(f"Successfully called the retrieval endpoint. Received {len(responses)} responses.")
            logger.info("Calling LLM inference endpoint")
            
            llm_response = requests.post(url=LLM_INFERENCE_URL, json={"user_query": query_request.user_query, "responses": responses},headers=headers)
            logger.info("Successfully called the system.")
            llm_response = llm_response.json()["answer"]
            return {"answer": llm_response,"papers":responses}
        
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    uvicorn.run("app_webapp:app", host="localhost", port=8000, reload=True)
