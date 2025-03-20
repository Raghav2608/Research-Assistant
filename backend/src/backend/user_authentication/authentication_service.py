from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi import status

from src.backend.user_authentication.rate_limiter import RateLimiter
from src.backend.user_authentication.token_manager import TokenManager
from src.backend.user_authentication.authenticator import UserAuthenticator

class UserAuthenticationService:
    """
    Class for handling user authentication.
    Responsible for:
    - Rate limiting
    - Password strength validation
    - Signing up a new user
    - Logging in an existing user   
    - Generating a token for the user to make authenticated requests
    """
    def __init__(self, is_testing:bool):
        """
        Initialises the UserAuthenticationService object.

        Args:
            is_testing (bool): A boolean flag to indicate if this user authentication service is being 
                            used for testing. If it is, then it will use a collection specifically 
                            for testing. Otherwise, it will use the production collection.
        """
        self.rate_limiter = RateLimiter()
        self.user_authenticator = UserAuthenticator(is_testing)
        self.token_manager = TokenManager()
    
    def _handle_signup(self, username:str, password:str):
        """
        Handles the signup process for the user.

        Args:
            username (str): The username of the user.
            password (str): The password of the user
        """
        is_strong_password = self.user_authenticator.check_password_strength(password)
        if not is_strong_password:
            message = (
                    "Password is not strong enough."
                    " Please use a password with at least 12 characters,"
                    " including a number, a special character, and an uppercase letter"
                    )
            return status.HTTP_400_BAD_REQUEST, message
        
        self.user_authenticator.create_user(username=username, password=password)
        return status.HTTP_201_CREATED, "User created successfully."
    
    def _handle_login(self, username:str, password:str, ip_identifier:str, user_identifier:str):
        """
        Handles the login process for the user.

        Args:
            username (str): The username of the user.
            password (str): The password of the user.
            ip_identifier (str): The identifier for the IP address used for rate limiting.
            user_identifier (str): The identifier for the user used for rate limiting.
        """
        self.rate_limiter.add_attempt(identifier=ip_identifier)
        self.rate_limiter.add_attempt(identifier=user_identifier)

        if not self.user_authenticator.verify_user(username=username, password=password):
            remaining_attempts = self.rate_limiter.get_remaining_attempts(identifier=user_identifier)
            message = f"Incorrect password, please try again. {remaining_attempts} attempts remaining."
            return status.HTTP_401_UNAUTHORIZED, message
        return status.HTTP_200_OK, "User authenticated successfully."
    
    def handle_rate_limiting(self, request:Request, username:str):
        """
        Handles rate limiting for the user and IP address. Based on
        both login and signup attempts.

        Args:
            request (Request): The request object.
            username (str): The username of the user.
        """
        ip_address = request.client.host
        ip_identifier = f"ip:{ip_address}"
        ip_r_limited, ip_time_remaining = self.rate_limiter.is_rate_limited(identifier=ip_identifier)
        if ip_r_limited:
            message = f"Too many login attempts. Please try again in {ip_time_remaining} seconds."
            return True, message

        user_identifier = f"user:{username}"
        username_r_limited, username_time_remaining = self.rate_limiter.is_rate_limited(identifier=user_identifier)
        if username_r_limited:
            message = f"Too many login attempts for this user. Please try again in {username_time_remaining} seconds."
            return True, message
        
        return False, None
    
    def handle_authentication(self, username:str, password:str, request:Request):
        """
        Handles the authentication process for the user.

        Args:
            username (str): The username of the user.
            password (str): The password of the user.
            request (Request): The request object.
        """
        is_existing_user = self.user_authenticator.user_exists(username=username)
        user_identifier = f"user:{username}"
        ip_address = request.client.host
        ip_identifier = f"ip:{ip_address}"

        if not is_existing_user:
            status_code, message = self._handle_signup(username=username, password=password)
        else:
            status_code, message = self._handle_login(
                                                    username=username,
                                                    password=password, 
                                                    ip_identifier=ip_identifier, 
                                                    user_identifier=user_identifier
                                                    )
        if status_code == status.HTTP_200_OK or status_code == status.HTTP_201_CREATED:
            self.rate_limiter.clear_attempts(identifier=user_identifier)
            self.rate_limiter.clear_attempts(identifier=ip_identifier)
        return status_code, message
    
    def get_token_response(self, username:str, status_code:int, message:str):
        """
        Generates a token for the user and returns a response with the token.

        Args:
            username (str): The username of the user.
            status_code (int): The status code of the response.
            message (str): The message to be included in the response
        """
        token_dict = self.token_manager.generate_token(user_id=username)
        token = token_dict["token"]
        duration = token_dict["duration"]
        
        # Add 'Bearer' to the token
        token = f"Bearer {token}"

        # Set the token in the cookie (Used for making authenticated requests)
        response = JSONResponse(   
                            content={
                                    "message": message,
                                    "token": token
                                    },
                            status_code=status_code
                            )
        response.set_cookie(
                        key="token", 
                        value=token,
                        httponly=True, # Prevents JavaScript from accessing the cookie
                        secure=True, # Ensures that the cookie is only sent over HTTPS
                        samesite="strict", # Ensures that the cookie is only sent to the same site that set it
                        max_age=duration #  Set the cookie to expire after a certain duration
                        )
        return response