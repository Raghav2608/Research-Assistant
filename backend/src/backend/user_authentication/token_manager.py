import os
import jwt
import datetime

from fastapi import HTTPException, Request
from fastapi import status
from dotenv import load_dotenv
from typing import Dict, Any, Union

class TokenManager:
    """
    Class for generating tokens for user authentication.
    - Generates a token for the user using the user ID and a secret key.
    """
    def __init__(self):
        # Load the secret key from the environment variables
        load_dotenv()
        self.secret_key = os.getenv("TOKEN_GENERATOR_SECRET_KEY")
        self.token_duration = 1 # In hours

    def generate_token(self, user_id:str) -> Dict[str, Union[str, int]]:
        """
        Generates a token for the user.

        Args:
            user_id (str): The user ID for which the token is to be generated.
        """
        expiration_time = datetime.datetime.now(datetime.UTC) + datetime.timedelta(hours=self.token_duration)
        payload = {
                "user_id": user_id, 
                "exp": expiration_time # Automatically checked with JWT.decode
                }
        token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        duration_in_seconds = self.token_duration * 3600
        return {"token": token, "duration": duration_in_seconds}

def verify_token(request:Request) -> Dict[str, Any]:
    """
    Verifies the token present in the request header or cookies
    by decoding it using the secret key.

    Args:
        request (Request): The request object containing the token in the header.
    """
    load_dotenv()
    secret_key = os.getenv("TOKEN_GENERATOR_SECRET_KEY")
    if not secret_key:
        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail="Secret key not found. Please set 'TOKEN_GENERATOR_SECRET_KEY' in the environment variables."
                            )
    
    # Attempt to get the token from the header or the cookies
    token = request.cookies.get("token")
    if not token:
        token = request.headers.get("Authorization")
    
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token. You aren't authorized to access this page.")
    
    token = token.split("Bearer ")[1] # Extract the token from the header
    
    try:
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token has expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token is invalid.")
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=f"An error occurred during token verification. {e}")