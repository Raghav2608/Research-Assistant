import time

from collections import defaultdict
from typing import Tuple

class RateLimiter:
    """
    Class for server-side rate limiting to prevent brute force attacks.
    Responsible for:
    - Tracking the number of attempts made by each user/IP address.
    - Locking out the user/IP address if the number of attempts exceeds a certain limit.
    - Clearing the attempts after a certain time window.
    - Providing the remaining attempts for a user/IP address
    """
    def __init__(self):
        self.attempts = defaultdict(list) # Attempts made by each user/IP address
        self.lockouts = defaultdict(float) # Lockout duration for each user/IP address

        # Constants
        self.MAX_ATTEMPTS = 5
        self.WINDOW_SIZE = 600 # In seconds
        self.LOCKOUT_DURATION = 900 # In seconds

    def is_rate_limited(self, identifier:str) -> Tuple[bool, int]:
        """
        Checks if the user/IP address is rate limited.

        Args:
            identifier (str): The user ID or the IP address of the user.
        """
        current_time = time.time()

        # Check if the user/IP address is locked out
        if self.lockouts[identifier] > current_time:
            time_remaining = int(self.lockouts[identifier] - current_time)
            return True, time_remaining
        
        # Clean up old attempts (that are outside the window)
        self.attempts[identifier] = [time for time in self.attempts[identifier] if (current_time - time) < self.WINDOW_SIZE]

        # Check if the user/IP address has made too many attempts
        if len(self.attempts[identifier]) >= self.MAX_ATTEMPTS:
            self.lockouts[identifier] = current_time + self.LOCKOUT_DURATION
            return True, self.LOCKOUT_DURATION
        
        return False, 0
    
    def get_remaining_attempts(self, identifier:str) -> int:
        """
        Returns the remaining attempts for the user/IP address.

        Args:
            identifier (str): The user ID or the IP address of the user.
        """
        if identifier in self.attempts:
            return self.MAX_ATTEMPTS - len(self.attempts[identifier])
        return self.MAX_ATTEMPTS
            
    def add_attempt(self, identifier:str) -> None:
        """
        Adds an attempt for the user/IP address.

        Args:
            identifier (str): The user ID or the IP address of the user.
        """
        current_time = time.time()
        self.attempts[identifier].append(current_time)

    def clear_attempts(self, identifier:str) -> None:
        """
        Clears the attempts made by the user/IP address.

        Args:
            identifier (str): The user ID or the IP address of the user.
        """
        self.attempts[identifier] = []
        self.lockouts[identifier] = 0