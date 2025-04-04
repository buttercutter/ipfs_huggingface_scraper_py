"""Test the rate limiter component."""

import time
import pytest
from unittest.mock import MagicMock, patch
from ipfs_huggingface_scraper_py.rate_limiter import RateLimiter

@pytest.fixture
def rate_limiter():
    """Create a rate limiter with test settings."""
    return RateLimiter(
        default_rate=100.0,  # High default rate for faster tests
        daily_quota=100,     # Small quota for testing
        authenticated_quota=200,
        max_retries=3
    )

def test_initialization():
    """Test that rate limiter initializes correctly."""
    limiter = RateLimiter(default_rate=5.0, daily_quota=300, max_retries=2)
    assert limiter.default_rate == 5.0
    assert limiter.daily_quota == 300
    assert limiter.max_retries == 2
    assert limiter.quota_used == 0
    assert not limiter.is_authenticated

def test_set_authenticated(rate_limiter):
    """Test setting authentication status."""
    # Initial state - not authenticated
    assert not rate_limiter.is_authenticated
    
    # Set to authenticated
    rate_limiter.set_authenticated(True)
    assert rate_limiter.is_authenticated
    
    # Set back to unauthenticated
    rate_limiter.set_authenticated(False)
    assert not rate_limiter.is_authenticated

def test_set_get_rate_limit(rate_limiter):
    """Test setting and getting rate limits."""
    # Default rate
    assert rate_limiter.get_rate_limit("unknown_endpoint") == 100.0
    
    # Set custom rate
    rate_limiter.set_rate_limit("test_endpoint", 5.0)
    assert rate_limiter.get_rate_limit("test_endpoint") == 5.0
    
    # Set another custom rate
    rate_limiter.set_rate_limit("another_endpoint", 10.0)
    assert rate_limiter.get_rate_limit("test_endpoint") == 5.0
    assert rate_limiter.get_rate_limit("another_endpoint") == 10.0

def test_quota_tracking(rate_limiter):
    """Test quota tracking."""
    # Initial quota
    assert rate_limiter.quota_used == 0
    assert rate_limiter._check_quota() is True
    
    # Record some successes
    for _ in range(5):
        rate_limiter.record_success("test_endpoint")
    
    # Check quota used
    assert rate_limiter.quota_used == 5
    assert rate_limiter._check_quota() is True
    
    # Use up the quota
    for _ in range(95):
        rate_limiter.record_success("test_endpoint")
    
    # Check quota exhausted
    assert rate_limiter.quota_used == 100
    assert rate_limiter._check_quota() is False
    
    # Switch to authenticated to get higher quota
    rate_limiter.set_authenticated(True)
    assert rate_limiter._check_quota() is True

@patch('time.time')
def test_quota_reset(mock_time, rate_limiter):
    """Test quota reset at midnight."""
    # Set current time
    current_time = 1640995200  # 2022-01-01 00:00:00
    mock_time.return_value = current_time
    
    # Set reset time to just beyond current time
    rate_limiter.quota_reset_time = current_time + 1
    
    # Use some quota
    rate_limiter.record_success("test_endpoint")
    assert rate_limiter.quota_used == 1
    
    # Move time past reset
    mock_time.return_value = current_time + 2
    
    # Check quota - should be reset
    assert rate_limiter._check_quota() is True
    assert rate_limiter.quota_used == 0
    assert rate_limiter.quota_reset_time > current_time + 2

def test_rate_limiting_wait(rate_limiter):
    """Test rate limiting wait behavior."""
    # Set a very low rate for testing
    rate_limiter.set_rate_limit("slow_endpoint", 2.0)  # 2 requests per second
    
    # First request should not wait
    start_time = time.time()
    rate_limiter.wait_if_needed("slow_endpoint")
    elapsed = time.time() - start_time
    assert elapsed < 0.1  # Should be very fast
    
    # Second immediate request should wait ~0.5 seconds
    start_time = time.time()
    rate_limiter.wait_if_needed("slow_endpoint")
    elapsed = time.time() - start_time
    assert elapsed >= 0.4  # Should wait approximately 0.5 seconds (allowing some tolerance)

def test_backoff_calculation(rate_limiter):
    """Test backoff time calculation."""
    # No previous failures
    assert rate_limiter._calculate_backoff_time("test_endpoint") == 1  # 2^0 = 1
    
    # One failure
    rate_limiter.consecutive_failures["test_endpoint"] = 1
    assert rate_limiter._calculate_backoff_time("test_endpoint") == 2  # 2^1 = 2
    
    # Three failures
    rate_limiter.consecutive_failures["test_endpoint"] = 3
    assert rate_limiter._calculate_backoff_time("test_endpoint") == 8  # 2^3 = 8
    
    # Maximum backoff
    rate_limiter.consecutive_failures["test_endpoint"] = 10
    assert rate_limiter._calculate_backoff_time("test_endpoint") == 300  # Max 5 minutes (300 seconds)

def test_record_rate_limited(rate_limiter):
    """Test recording rate limited requests."""
    # Record a rate limited request
    current_time = time.time()
    rate_limiter.record_rate_limited("test_endpoint")
    
    # Check that consecutive failures was incremented
    assert rate_limiter.consecutive_failures["test_endpoint"] == 1
    
    # Check that backoff was applied
    assert "test_endpoint" in rate_limiter.backoff_until
    assert rate_limiter.backoff_until["test_endpoint"] > current_time
    
    # Record a second rate limited request
    rate_limiter.record_rate_limited("test_endpoint")
    
    # Check that consecutive failures was incremented again
    assert rate_limiter.consecutive_failures["test_endpoint"] == 2
    
    # Record a success to reset backoff
    rate_limiter.record_success("test_endpoint")
    
    # Check that consecutive failures was reset
    assert rate_limiter.consecutive_failures["test_endpoint"] == 0

def test_execute_with_rate_limit(rate_limiter):
    """Test execute with rate limit functionality."""
    # Create a mock function
    mock_func = MagicMock(return_value="success")
    
    # Execute with rate limit
    result = rate_limiter.execute_with_rate_limit("test_endpoint", mock_func, "arg1", keyword="arg2")
    
    # Check result
    assert result == "success"
    
    # Check function was called with correct arguments
    mock_func.assert_called_once_with("arg1", keyword="arg2")

def test_execute_with_rate_limit_retries(rate_limiter):
    """Test execute with rate limit with retries."""
    # Create a mock function that fails with rate limit error twice, then succeeds
    mock_func = MagicMock(side_effect=[
        Exception("Rate limit exceeded"),
        Exception("Rate limit exceeded"),
        "success"
    ])
    
    # Override is_rate_limit_error to always return True for testing
    rate_limiter._is_rate_limit_error = lambda e: True
    
    # Set a very short backoff time for testing
    rate_limiter._calculate_backoff_time = lambda e: 0.01
    
    # Execute with rate limit
    result = rate_limiter.execute_with_rate_limit("test_endpoint", mock_func)
    
    # Check result
    assert result == "success"
    
    # Check function was called three times
    assert mock_func.call_count == 3
    
    # Check consecutive failures was reset after success
    assert rate_limiter.consecutive_failures["test_endpoint"] == 0

def test_execute_with_max_retries_exceeded(rate_limiter):
    """Test execute with rate limit with max retries exceeded."""
    # Create a mock function that always fails with rate limit error
    mock_func = MagicMock(side_effect=Exception("Rate limit exceeded"))
    
    # Override is_rate_limit_error to always return True for testing
    rate_limiter._is_rate_limit_error = lambda e: True
    
    # Set a very short backoff time for testing
    rate_limiter._calculate_backoff_time = lambda e: 0.01
    
    # Execute with rate limit - should fail after 3 retries (max_retries=3)
    with pytest.raises(Exception) as excinfo:
        rate_limiter.execute_with_rate_limit("test_endpoint", mock_func)
    
    assert "Rate limit exceeded" in str(excinfo.value)
    
    # Check function was called 4 times (initial + 3 retries)
    assert mock_func.call_count == 4
    
    # Check consecutive failures was set
    assert rate_limiter.consecutive_failures["test_endpoint"] == 4

def test_rate_limit_error_detection():
    """Test detection of rate limit errors."""
    limiter = RateLimiter()
    
    # Rate limit errors
    assert limiter._is_rate_limit_error(Exception("Rate limit exceeded"))
    assert limiter._is_rate_limit_error(Exception("429 Too Many Requests"))
    assert limiter._is_rate_limit_error(Exception("Server returned HTTP 429"))
    
    # Non-rate limit errors
    assert not limiter._is_rate_limit_error(Exception("Not Found"))
    assert not limiter._is_rate_limit_error(Exception("Server Error"))
    assert not limiter._is_rate_limit_error(Exception("Invalid request"))

def test_backoff_waiting(rate_limiter):
    """Test waiting during backoff period."""
    # Set a backoff until the future
    future_time = time.time() + 0.2  # 200ms in the future
    rate_limiter.backoff_until["test_endpoint"] = future_time
    
    # Wait - should wait for the backoff period
    start_time = time.time()
    rate_limiter.wait_if_needed("test_endpoint")
    elapsed = time.time() - start_time
    
    # Should have waited at least ~200ms
    assert elapsed >= 0.15  # Allow some tolerance

def test_non_rate_limit_exception(rate_limiter):
    """Test handling of non-rate limit exceptions."""
    # Create a mock function that fails with a non-rate limit error
    mock_func = MagicMock(side_effect=Exception("Not a rate limit error"))
    
    # Execute with rate limit - should propagate the exception
    with pytest.raises(Exception) as excinfo:
        rate_limiter.execute_with_rate_limit("test_endpoint", mock_func)
    
    assert "Not a rate limit error" in str(excinfo.value)
    
    # Check function was called only once (no retries)
    assert mock_func.call_count == 1