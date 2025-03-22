import hashlib
import os
import logging

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi

class UserAuthenticator:
    """
    A class that authenticates the user by checking the username and password provided.
    It is responsible for:
    - Initialising the MongoDB client and connecting to the database.
    - Generating a random salt for password hashing.
    - Checking the password strength.
    - Encoding passwords with the salt using the SHA-256 algorithm.
    - Verifying if the two password hashes are the same.
    - Verifying if the user exists in the database and if the entered password is correct.
    - Creating a new user if the user does not exist in the database.
    """
    def __init__(self, is_testing:bool):
        """
        Initialises the UserAuthenticator object.

        Args:
            is_testing (bool): A boolean flag to indicate if this user authenticator is being 
                            used for testing. If it is, then it will use a collection specifically 
                            for testing. Otherwise, it will use the production collection.
        """
        # Load the environment variables
        load_dotenv()
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise Exception("MongoDB URI not found. Please set 'MONGODB_URI' in the environment variables.")

        log = logging.getLogger(self.__class__.__name__)
        
        # Initialize the MongoDB client
        self.client = MongoClient(uri, server_api=ServerApi('1'))
        self.db = self.client["user_authentication"]

        # Select the collection based on the environment
        if is_testing:
            log.info("Using test database")
            self.collection = self.db["test_users"]
        else:
            log.info("Using production database")
            self.collection = self.db["users"]
        
        self.special_characters = "!@#$%^&*()-_=+[]{}|;:'\",.<>?/\\`~"

    def _generate_salt(self) -> bytes:
        """
        Generates a random salt for password hashing.

        Returns:
            bytes: The generated salt.
        """
        return os.urandom(32)

    def _encode(self, password:str, salt:bytes) -> str:
        """
        Hashes a password with a salt.
        - Uses the SHA-256 algorithm for encoding.
        - Uses 100k iterations for encoding.

        Args:
            password (str): The password to be encoded.
            salt (bytes): The salt to be used for encoding.
        """
        return hashlib.pbkdf2_hmac(
                                "sha256", 
                                password.encode('utf-8'), 
                                salt, 
                                100000 # 100k iterations
                                ).hex()

    def _verify_password(self, password1:str, password2:str) -> bool:
        """
        Verifies if the two passwords provided are the same.

        Args:
            password1 (str): The first password.
            password2 (str): The second password.

        Returns:
            bool: True if the passwords are the same, False otherwise.
        """
        return password1 == password2
    
    def check_password_strength(self, password:str) -> bool:
        """
        Checks if the password meets the minimum requirements:
        - At least 12 characters long.
        - Contains at least one digit.
        - Contains at least one uppercase letter.
        - Contains at least one lowercase letter.
        - Contains at least one special character.
        
        Args:
            password (str): The password to be checked.
        """
        
        if len(password) < 12:
            return False
        if not any(char.isdigit() for char in password):
            return False
        if not any(char.isupper() for char in password):
            return False
        if not any(char.islower() for char in password):
            return False
        if not any(char in self.special_characters for char in password):
            return False
        return True
    
    def user_exists(self, username:str) -> bool:
        """
        Checks if the user exists in the database.

        Args:
            username (str): The username of the user.
        """
        return self.collection.find_one({"username": username}) is not None
    
    def create_user(self, username:str, password:str) -> None:
        """
        Creates a new user in the database.
        - Saves the username, password hash, and salt in the database.

        Args:
            username (str): The username of the user.
            password (str): The password of the user used for encoding.
        """
        salt = self._generate_salt()
        password_hash = self._encode(password=password, salt=salt)
        self.collection.insert_one(
                                    {
                                    "username": username, 
                                    "password_hash": password_hash,
                                    "salt": salt
                                    })
    
    def verify_user(self, username:str, password:str) -> bool:
        """
        Verifies if the user exists in the database and if the password is correct.

        Args:
            username (str): The username of the user.
            password (str): The password of the user.
        """
        # Check if the user exists in the database
        if self.collection.find_one({"username": username}) is None:
            # Create a new user if the user does not exist
            salt = self._generate_salt()
            password_hash = self._encode(password=password, salt=salt)
            self.collection.insert_one(
                                        {
                                        "username": username, 
                                        "password_hash": password_hash,
                                        "salt": salt
                                        })
        else:
            """
            Verify that the password is correct:
            1. Retrieve the user data from the database
            2. Get the actual password hash and salt
            3. Generate the password hash using the salt
            4. Verify if the generated password hash is the same as the actual password hash
            """
            userdata = self.collection.find_one({"username": username})
            actual_password_hash = userdata["password_hash"]
            salt = userdata["salt"]
            generated_password_hash = self._encode(password=password, salt=salt)
            if not self._verify_password(generated_password_hash, actual_password_hash):
                return False
        return True