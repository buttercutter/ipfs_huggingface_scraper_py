import time
import logging
import threading
from typing import Dict, Optional, Callable, Any

class RateLimiter:
    """
    Implements rate limiting for API calls with quota management.
    
    Features:
    - Enforces rate limits for different API endpoints
    - Implements adaptive backoff for rate limit errors
    - Provides quota distribution across multiple operations
    - Tracks rate limit usage and remaining quota
    """
    
    def __init__(self, 
                 default_rate: float = 5.0, 
                 daily_quota: int = 300000,
                 authenticated_quota: int = 1000000,
                 max_retries: int = 5):
        """
        Initialize the rate limiter.
        
        Args:
            default_rate: Default requests per second (default: 5.0)
            daily_quota: Default daily quota for anonymous requests (default: 300K)
            authenticated_quota: Daily quota for authenticated requests (default: 1M)
            max_retries: Maximum number of retries for rate limited requests
        """
        self.default_rate = default_rate
        self.daily_quota = daily_quota
        self.authenticated_quota = authenticated_quota
        self.max_retries = max_retries
        
        # Endpoint-specific rate limits
        self.endpoint_rates: Dict[str, float] = {}
        
        # Last request timestamp for each endpoint
        self.last_request_time: Dict[str, float] = {}
        
        # Quota tracking
        self.quota_used = 0
        self.quota_reset_time = self._get_next_day_timestamp()
        
        # Backoff tracking
        self.backoff_until: Dict[str, float] = {}
        self.consecutive_failures: Dict[str, int] = {}
        
        # Authentication status
        self.is_authenticated = False
        
        # Thread safety
        self.lock = threading.RLock()
    
    def set_authenticated(self, is_authenticated: bool = True) -> None:
        """Set authentication status to determine quota."""
        with self.lock:
            self.is_authenticated = is_authenticated
            logging.info(f"Rate limiter using {'authenticated' if is_authenticated else 'anonymous'} quota")
    
    def set_rate_limit(self, endpoint: str, rate: float) -> None:
        """
        Set custom rate limit for a specific endpoint.
        
        Args:
            endpoint: API endpoint identifier
            rate: Maximum requests per second
        """
        with self.lock:
            self.endpoint_rates[endpoint] = rate
            logging.debug(f"Set rate limit for {endpoint} to {rate} requests/second")
    
    def get_rate_limit(self, endpoint: str) -> float:
        """
        Get the rate limit for an endpoint.
        
        Args:
            endpoint: API endpoint identifier
            
        Returns:
            Rate limit in requests per second
        """
        with self.lock:
            return self.endpoint_rates.get(endpoint, self.default_rate)
    
    def _get_next_day_timestamp(self) -> float:
        """
        Calculate timestamp for the start of the next day (UTC).
        
        Returns:
            Timestamp for quota reset
        """
        import datetime
        now = datetime.datetime.utcnow()
        tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
        return tomorrow.timestamp()
    
    def _check_quota(self) -> bool:
        """
        Check if daily quota is exceeded.
        
        Returns:
            True if quota available, False if exceeded
        """
        with self.lock:
            # Reset quota if we crossed the reset time
            current_time = time.time()
            if current_time >= self.quota_reset_time:
                self.quota_used = 0
                self.quota_reset_time = self._get_next_day_timestamp()
                logging.info("Daily API quota has been reset")
            
            # Check if we have quota remaining
            quota_limit = self.authenticated_quota if self.is_authenticated else self.daily_quota
            return self.quota_used < quota_limit
    
    def _calculate_backoff_time(self, endpoint: str) -> float:
        """
        Calculate exponential backoff time for retries.
        
        Args:
            endpoint: API endpoint identifier
            
        Returns:
            Seconds to wait before retrying
        """
        with self.lock:
            failures = self.consecutive_failures.get(endpoint, 0)
            # Exponential backoff: 2^failures seconds, maxing out at 5 minutes
            backoff_seconds = min(2 ** failures, 300)
            return backoff_seconds
    
    def wait_if_needed(self, endpoint: str) -> None:
        """
        Wait if necessary to respect rate limits.
        
        Args:
            endpoint: API endpoint identifier
        """
        with self.lock:
            current_time = time.time()
            
            # Check if we're in a backoff period for this endpoint
            if endpoint in self.backoff_until and current_time < self.backoff_until[endpoint]:
                wait_time = self.backoff_until[endpoint] - current_time
                logging.info(f"Backing off from {endpoint} for {wait_time:.2f} seconds due to rate limiting")
                time.sleep(wait_time)
                return
            
            # Check if we need to wait based on the rate limit
            if endpoint in self.last_request_time:
                rate = self.get_rate_limit(endpoint)
                min_interval = 1.0 / rate
                elapsed = current_time - self.last_request_time[endpoint]
                
                if elapsed < min_interval:
                    wait_time = min_interval - elapsed
                    logging.debug(f"Rate limiting {endpoint}: waiting {wait_time:.3f} seconds")
                    time.sleep(wait_time)
            
            # Update last request time
            self.last_request_time[endpoint] = time.time()
            
            # Check quota
            if not self._check_quota():
                # We've hit our daily quota, this is a hard limit
                seconds_until_reset = self.quota_reset_time - time.time()
                logging.warning(f"Daily API quota exceeded. Quota will reset in {seconds_until_reset/3600:.1f} hours")
                # Sleep for a while to prevent aggressive retries
                time.sleep(min(30, seconds_until_reset))
    
    def record_success(self, endpoint: str, quota_cost: int = 1) -> None:
        """
        Record a successful API call.
        
        Args:
            endpoint: API endpoint identifier
            quota_cost: Cost against the daily quota (default: 1)
        """
        with self.lock:
            # Reset consecutive failures counter
            self.consecutive_failures[endpoint] = 0
            
            # Clear any backoff
            if endpoint in self.backoff_until:
                del self.backoff_until[endpoint]
            
            # Update quota
            self.quota_used += quota_cost
            
            # Log if we're approaching quota limits
            quota_limit = self.authenticated_quota if self.is_authenticated else self.daily_quota
            quota_percentage = (self.quota_used / quota_limit) * 100
            
            if quota_percentage >= 90:
                logging.warning(f"API quota usage: {self.quota_used}/{quota_limit} ({quota_percentage:.1f}%)")
            elif quota_percentage >= 70:
                logging.info(f"API quota usage: {self.quota_used}/{quota_limit} ({quota_percentage:.1f}%)")
    
    def record_rate_limited(self, endpoint: str) -> None:
        """
        Record a rate limited API call and apply backoff.
        
        Args:
            endpoint: API endpoint identifier
        """
        with self.lock:
            # Increment consecutive failures
            failures = self.consecutive_failures.get(endpoint, 0) + 1
            self.consecutive_failures[endpoint] = failures
            
            # Calculate and apply backoff
            backoff_time = self._calculate_backoff_time(endpoint)
            self.backoff_until[endpoint] = time.time() + backoff_time
            
            logging.warning(
                f"Rate limited on {endpoint}. Backing off for {backoff_time:.1f} seconds. "
                f"Consecutive failures: {failures}"
            )
    
    def execute_with_rate_limit(self, endpoint: str, func: Callable[..., Any], *args, **kwargs) -> Any:
        """
        Execute a function with rate limiting applied.
        
        Args:
            endpoint: API endpoint identifier
            func: Function to execute
            *args: Arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            Result of the function call
            
        Raises:
            Exception: Any exception from the function after all retries
        """
        retries = 0
        last_exception = None
        
        while retries <= self.max_retries:
            try:
                # Wait according to rate limit
                self.wait_if_needed(endpoint)
                
                # Execute the function
                result = func(*args, **kwargs)
                
                # Record success
                self.record_success(endpoint)
                
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if it's a rate limit error (adjust based on actual error type/message)
                if self._is_rate_limit_error(e):
                    self.record_rate_limited(endpoint)
                    retries += 1
                    logging.warning(f"Rate limit error on {endpoint}. Retry {retries}/{self.max_retries}")
                else:
                    # For other exceptions, just propagate
                    raise
        
        # If we've exhausted all retries
        if last_exception:
            logging.error(f"Failed after {self.max_retries} retries on {endpoint}: {last_exception}")
            raise last_exception
        
        return None  # Should never reach here
    
    def _is_rate_limit_error(self, exception: Exception) -> bool:
        """
        Determine if an exception indicates a rate limit error.
        
        Args:
            exception: The exception to check
            
        Returns:
            True if it's a rate limit error, False otherwise
        """
        # Adjust based on the actual error type/message from the API
        error_str = str(exception).lower()
        return (
            "rate limit" in error_str or 
            "too many requests" in error_str or
            "429" in error_str
        )