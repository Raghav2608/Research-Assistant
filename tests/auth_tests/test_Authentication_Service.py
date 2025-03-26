import pytest
from unittest.mock import MagicMock, patch
from fastapi import Request, status
from backend.src.backend.user_authentication.authentication_service import UserAuthenticationService

@pytest.fixture
def mock_dependencies():
    """Fixture to mock dependencies used in UserAuthenticationService."""
    rate_limiter = MagicMock()
    user_authenticator = MagicMock()
    token_manager = MagicMock()
    return rate_limiter, user_authenticator, token_manager

@pytest.fixture
def auth_service(mock_dependencies):
    """Fixture to initialize UserAuthenticationService with mock dependencies."""
    rate_limiter, user_authenticator, token_manager = mock_dependencies
    service = UserAuthenticationService(is_testing=True)
    service.rate_limiter = rate_limiter
    service.user_authenticator = user_authenticator
    service.token_manager = token_manager
    return service

@pytest.fixture
def mock_request():
    """Fixture to create a mock Request object with a fake IP."""
    request = MagicMock(spec=Request)
    request.client.host = "127.0.0.1"
    return request

def test_successful_signup(auth_service, mock_request):
    """Test successful user signup."""
    auth_service.user_authenticator.user_exists.return_value = False
    auth_service.user_authenticator.check_password_strength.return_value = True
    auth_service.user_authenticator.create_user.return_value = None
    
    status_code, message = auth_service.handle_authentication(
        "test_user", "StrongPass123!", "StrongPass123!", mock_request
    )
    
    assert status_code == status.HTTP_201_CREATED
    assert message == "User created successfully."
    auth_service.user_authenticator.create_user.assert_called_once()

def test_existing_user_signup(auth_service, mock_request):
    with patch.object(auth_service.user_authenticator, "user_exists", return_value=True):
    
        status_code, message = auth_service.handle_authentication(
            username="testuser",
            password="SecurePass123!",
            confirm_password="SecurePass123!",
            request=mock_request
        )
        
        assert status_code == 400
        assert message == "User already exists. Cannot create a new user with the same username."

def test_signup_weak_password(auth_service, mock_request):
    """Test signup with a weak password."""
    auth_service.user_authenticator.user_exists.return_value = False
    auth_service.user_authenticator.check_password_strength.return_value = False
    
    status_code, message = auth_service.handle_authentication(
        "test_user", "weakpass", "weakpass", mock_request
    )
    
    assert status_code == status.HTTP_400_BAD_REQUEST
    assert "Password is not strong enough" in message

def test_signup_password_mismatch(auth_service, mock_request):
    """Test signup with mismatching passwords."""
    auth_service.user_authenticator.user_exists.return_value = False
    auth_service.user_authenticator.check_password_strength.return_value = True
    
    status_code, message = auth_service.handle_authentication(
        "test_user", "StrongPass123!", "WrongPass123!", mock_request
    )
    
    assert status_code == status.HTTP_400_BAD_REQUEST
    assert "Passwords do not match" in message

def test_successful_login(auth_service, mock_request):
    """Test successful user login."""
    auth_service.user_authenticator.user_exists.return_value = True
    auth_service.user_authenticator.verify_user.return_value = True
    
    status_code, message = auth_service.handle_authentication(
        "test_user", "StrongPass123!", None, mock_request
    )
    
    assert status_code == status.HTTP_200_OK
    assert message == "User authenticated successfully."
    auth_service.user_authenticator.verify_user.assert_called_once()

def test_login_wrong_password(auth_service, mock_request):
    """Test login with an incorrect password."""
    auth_service.user_authenticator.user_exists.return_value = True
    auth_service.user_authenticator.verify_user.return_value = False
    auth_service.rate_limiter.get_remaining_attempts.return_value = 2
    
    status_code, message = auth_service.handle_authentication(
        "test_user", "WrongPass123!", None, mock_request
    )
    
    assert status_code == status.HTTP_401_UNAUTHORIZED
    assert "Incorrect password" in message
    assert "2 attempts remaining" in message

def test_login_user_does_not_exist(auth_service, mock_request):
    """Test login for a non-existing user."""
    auth_service.user_authenticator.user_exists.return_value = False
    
    status_code, message = auth_service.handle_authentication(
        "unknown_user", "SomePass123!", None, mock_request
    )
    
    assert status_code == status.HTTP_401_UNAUTHORIZED
    assert "User does not exist" in message

def test_rate_limiting(auth_service, mock_request):
    """Test rate limiting for login attempts."""
    auth_service.rate_limiter.is_rate_limited.return_value = (True, 30)
    
    is_limited, message = auth_service.handle_rate_limiting(mock_request, "test_user")
    
    assert is_limited is True
    assert "Too many login attempts" in message
    assert "30 seconds" in message

def test_token_generation(auth_service):
    """Test token generation after successful authentication."""
    auth_service.token_manager.generate_token.return_value = {"token": "abc123", "duration": 3600}
    
    response = auth_service.get_token_response("test_user", status.HTTP_200_OK, "Success")
    
    assert response.status_code == status.HTTP_200_OK
    assert response.body is not None
    auth_service.token_manager.generate_token.assert_called_once()
    
    cookies = response.headers.get("set-cookie")
    assert 'token="Bearer abc123"' in cookies
    assert "Secure" in cookies
    assert "HttpOnly" in cookies
