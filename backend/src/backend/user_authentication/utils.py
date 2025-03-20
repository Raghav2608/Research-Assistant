from fastapi import Request, HTTPException, status
from src.constants import ENDPOINT_URLS
from src.backend.user_authentication.token_manager import verify_token

def validate_request(request:Request) -> None:
    """
    Checks if the request is authenticated by checking for the presence of an authentication token in the request cookies.

    Args:
        request (Request): The request object containing information
                           that can be used to check for authentication.

    Returns:
        bool: True if the request is authenticated, False otherwise.
    """
    message = "You need to login to access this page."

    try:
        verify_token(request) # Verify the token which should be present in the request
    except Exception as e:
        message += f" {e}"
        handle_unauthenticated_request(request, message)

def handle_unauthenticated_request(request:Request, message:str) -> None:
    """
    Helper function to:
    - Redirect the user if the request is a GET request.
    - Raise an Unauthorized exception if the request is a POST request.

    Args:
        request (Request): The request object containing information
                           that can be used to check for authentication.
        message (str): The message to inclue in the response.
    """
    if request.method == "GET":
        # Redirect to the login page if the user is not authenticated
        raise HTTPException(
                            status_code=status.HTTP_302_FOUND,
                            detail=message,
                            headers={"Location": ENDPOINT_URLS['web_app']['additional_paths']['login']}
                            )
    else:
        raise HTTPException(
                            status_code=status.HTTP_401_UNAUTHORIZED,
                            detail=message
                            )