# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
HTTP client for Kiro API with retry logic support.

Handles:
- 403: automatic token refresh and retry
- 429: exponential backoff
- 5xx: exponential backoff
- Timeouts: exponential backoff

Supports both per-request clients and shared application-level client
with connection pooling for better resource management.
"""

import asyncio
import json
import random
from typing import Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro.config import (
    MAX_RETRIES,
    BASE_RETRY_DELAY,
    MAX_RETRY_DELAY,
    THROTTLE_BASE_DELAY,
    THROTTLE_STEP_DELAY,
    THROTTLE_MAX_DELAY,
    FIRST_TOKEN_MAX_RETRIES,
    STREAMING_READ_TIMEOUT,
)
from kiro.endpoints import endpoint_candidates, endpoint_registry
from kiro.auth import KiroAuthManager
from kiro.utils import get_kiro_headers
from kiro.network_errors import classify_network_error, get_short_error_message, NetworkErrorInfo


def backoff_delay(attempt: int) -> float:
    """
    Exponential backoff with +/-25% jitter for 5xx/408/network errors.

    Args:
        attempt: 0-based attempt index

    Returns:
        Delay in seconds
    """
    base = min(BASE_RETRY_DELAY * (2 ** attempt), MAX_RETRY_DELAY)
    return max(0.0, base * random.uniform(0.75, 1.25))


def throttle_delay(attempt: int, retry_after: Optional[str] = None) -> float:
    """
    Backoff for 429: linear growth with jitter; honours Retry-After if present.

    Args:
        attempt: 0-based attempt index
        retry_after: Raw Retry-After header value (seconds), if any

    Returns:
        Delay in seconds
    """
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), THROTTLE_MAX_DELAY)
        except ValueError:
            pass
    base = min(THROTTLE_BASE_DELAY + attempt * THROTTLE_STEP_DELAY, THROTTLE_MAX_DELAY)
    return base + random.uniform(0.0, 1.5)


# Models for which Kiro rejected additionalModelRequestFields (learned at runtime)
_ADDITIONAL_FIELDS_REJECTED: set = set()

ADDITIONAL_FIELDS_KEY = "additionalModelRequestFields"


def _payload_model_id(json_data: Optional[dict]) -> Optional[str]:
    """Returns the Kiro modelId from a generateAssistantResponse payload, if any."""
    try:
        return json_data["conversationState"]["currentMessage"]["userInputMessage"]["modelId"]
    except (KeyError, TypeError):
        return None


def strip_rejected_additional_fields(json_data: Optional[dict]) -> Optional[dict]:
    """
    Drops additionalModelRequestFields for models Kiro already rejected it for.

    Args:
        json_data: Kiro request payload

    Returns:
        Payload (a shallow copy without the field when stripped)
    """
    if isinstance(json_data, dict) and ADDITIONAL_FIELDS_KEY in json_data:
        if _payload_model_id(json_data) in _ADDITIONAL_FIELDS_REJECTED:
            return {k: v for k, v in json_data.items() if k != ADDITIONAL_FIELDS_KEY}
    return json_data


def agent_mode_for_payload(json_data: Optional[dict]) -> str:
    """
    Picks the x-amzn-kiro-agent-mode header from the payload's agentTaskType.

    Args:
        json_data: Kiro request payload

    Returns:
        "spectask" or "vibe"
    """
    if isinstance(json_data, dict):
        task_type = (json_data.get("conversationState") or {}).get("agentTaskType")
        if task_type == "spectask":
            return "spectask"
    return "vibe"


class KiroHttpClient:
    """
    HTTP client for Kiro API with retry logic support.
    
    Automatically handles errors and retries requests:
    - 403: refreshes token and retries
    - 429: waits with exponential backoff
    - 5xx: waits with exponential backoff
    - Timeouts: waits with exponential backoff
    
    Supports two modes of operation:
    1. Per-request client: Creates and owns its own httpx.AsyncClient
    2. Shared client: Uses an application-level shared client (recommended)
    
    Using a shared client reduces memory usage and enables connection pooling,
    which is especially important for handling concurrent requests.
    
    Attributes:
        auth_manager: Authentication manager for obtaining tokens
        client: httpx HTTP client (owned or shared)
    
    Example:
        >>> # Per-request client (legacy mode)
        >>> client = KiroHttpClient(auth_manager)
        >>> response = await client.request_with_retry(...)
        
        >>> # Shared client (recommended)
        >>> shared = httpx.AsyncClient(limits=httpx.Limits(...))
        >>> client = KiroHttpClient(auth_manager, shared_client=shared)
        >>> response = await client.request_with_retry(...)
    """
    
    def __init__(
        self,
        auth_manager: KiroAuthManager,
        shared_client: Optional[httpx.AsyncClient] = None
    ):
        """
        Initializes the HTTP client.
        
        Args:
            auth_manager: Authentication manager
            shared_client: Optional shared httpx.AsyncClient for connection pooling.
                          If provided, this client will be used instead of creating
                          a new one. The shared client will NOT be closed by close().
        """
        self.auth_manager = auth_manager
        self._shared_client = shared_client
        self._owns_client = shared_client is None
        self.client: Optional[httpx.AsyncClient] = shared_client
    
    async def _get_client(self, stream: bool = False) -> httpx.AsyncClient:
        """
        Returns or creates an HTTP client with proper timeouts.
        
        If a shared client was provided at initialization, it is returned as-is.
        Otherwise, creates a new client with appropriate timeout configuration.
        
        httpx timeouts:
        - connect: TCP handshake (DNS + TCP SYN/ACK)
        - read: waiting for data from server between chunks
        - write: sending data to server
        - pool: waiting for free connection from pool
        
        IMPORTANT: FIRST_TOKEN_TIMEOUT is NOT used here!
        It is applied in streaming_openai.py via asyncio.wait_for() to control
        the wait time for the first token from the model (retry business logic).
        
        Args:
            stream: If True, uses STREAMING_READ_TIMEOUT for read (only for new clients)
        
        Returns:
            Active HTTP client
        """
        # If using shared client, return it directly
        # Shared client should be pre-configured with appropriate timeouts
        if self._shared_client is not None:
            return self._shared_client
        
        # Create new client if needed (per-request mode)
        if self.client is None or self.client.is_closed:
            if stream:
                # For streaming:
                # - connect: 30 sec (TCP connection, usually < 1 sec)
                # - read: STREAMING_READ_TIMEOUT (300 sec) - model may "think" between chunks
                # - write/pool: standard values
                timeout_config = httpx.Timeout(
                    connect=30.0,
                    read=STREAMING_READ_TIMEOUT,
                    write=30.0,
                    pool=30.0
                )
                logger.debug(f"Creating streaming HTTP client (read_timeout={STREAMING_READ_TIMEOUT}s)")
            else:
                # For regular requests: single timeout of 300 sec
                timeout_config = httpx.Timeout(timeout=300.0)
                logger.debug("Creating non-streaming HTTP client (timeout=300s)")
            
            self.client = httpx.AsyncClient(timeout=timeout_config, follow_redirects=True)
        return self.client
    
    async def close(self) -> None:
        """
        Closes the HTTP client if this instance owns it.
        
        If using a shared client, this method does nothing - the shared client
        should be closed by the application lifecycle manager.
        
        Uses graceful exception handling to prevent errors during cleanup
        from masking the original exception in finally blocks.
        """
        # Don't close shared clients - they're managed by the application
        if not self._owns_client:
            return
        
        if self.client and not self.client.is_closed:
            try:
                await self.client.aclose()
            except Exception as e:
                # Log but don't propagate - we're in cleanup code
                # Propagating here could mask the original exception
                logger.warning(f"Error closing HTTP client: {e}")
    
    async def request_with_retry(
        self,
        method: str,
        url: str,
        json_data: Optional[dict] = None,
        params: Optional[dict] = None,
        stream: bool = False
    ) -> httpx.Response:
        """
        Executes an HTTP request with retry logic.
        
        Automatically handles various error types:
        - 403: refreshes token via auth_manager.force_refresh() and retries
        - 429: rotates to the next configured endpoint (KIRO_ENDPOINTS), otherwise
          waits with linear backoff + jitter (Retry-After honoured)
        - 408/5xx: waits with capped exponential backoff + jitter
        - Timeouts: waits with exponential backoff
        
        For streaming, STREAMING_READ_TIMEOUT is used for waiting between chunks.
        First token timeout is controlled separately in streaming_openai.py via asyncio.wait_for().
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            json_data: Optional JSON body (for POST/PUT/PATCH)
            params: Optional query parameters (for GET)
            stream: Use streaming (default False)
        
        Returns:
            httpx.Response with successful response
        
        Raises:
            HTTPException: On failure after all attempts (502/504)
        """
        # Determine the number of retry attempts
        # FIRST_TOKEN_TIMEOUT is used in streaming_openai.py, not here
        max_retries = FIRST_TOKEN_MAX_RETRIES if stream else MAX_RETRIES
        
        client = await self._get_client(stream=stream)
        last_error = None
        last_error_info: Optional[NetworkErrorInfo] = None
        last_response: Optional[httpx.Response] = None  # Для сохранения последнего 429/5xx
        
        json_data = strip_rejected_additional_fields(json_data)
        endpoints = endpoint_candidates(url)
        account_key = id(self.auth_manager)
        agent_mode = agent_mode_for_payload(json_data)

        for attempt in range(max_retries):
            try:
                # Rotate across endpoints: throttled buckets are moved to the back
                endpoint = endpoint_registry.order(account_key, endpoints)[0]
                request_url = endpoint.url if endpoint.name != "custom" else url

                # Get current token
                token = await self.auth_manager.get_access_token()
                headers = get_kiro_headers(
                    self.auth_manager,
                    token,
                    amz_target=endpoint.amz_target if endpoint.name != "custom" else
                    "AmazonCodeWhispererStreamingService.GenerateAssistantResponse",
                    attempt=attempt + 1,
                    agent_mode=agent_mode,
                )
                
                # Build request kwargs based on parameters
                request_kwargs = {"headers": headers}
                
                if json_data is not None:
                    request_kwargs["content"] = json.dumps(json_data).encode()
                
                if params is not None:
                    request_kwargs["params"] = params
                
                if stream:
                    # Prevent CLOSE_WAIT connection leak (issue #38)
                    headers["Connection"] = "close"
                    req = client.build_request(method, request_url, **request_kwargs)
                    logger.debug(f"Sending request to Kiro API ({endpoint.name})...")
                    response = await client.send(req, stream=True)
                else:
                    logger.debug(f"Sending request to Kiro API ({endpoint.name})...")
                    response = await client.request(method, request_url, **request_kwargs)
                
                # Check status
                if response.status_code == 200:
                    return response
                
                # 403 - token expired, refresh and retry
                if response.status_code == 403:
                    logger.warning(f"Received 403, refreshing token (attempt {attempt + 1}/{MAX_RETRIES})")
                    await self.auth_manager.force_refresh()
                    continue
                
                # 429 - rate limit: mark endpoint bucket, rotate, back off
                if response.status_code == 429:
                    last_response = response  # Keep for caller after exhaustion
                    if len(endpoints) > 1:
                        endpoint_registry.throttle(account_key, endpoint.name)
                    retry_after = None
                    resp_headers = getattr(response, "headers", None)
                    if isinstance(resp_headers, (dict, httpx.Headers)):
                        retry_after = resp_headers.get("retry-after")
                    # Other endpoints are separate buckets: switch without waiting
                    free_left = len(endpoints) > 1 and any(
                        not endpoint_registry.is_throttled(account_key, e.name) for e in endpoints
                    )
                    if free_left:
                        logger.warning(
                            f"Received 429 on '{endpoint.name}', switching endpoint "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        continue
                    if attempt < max_retries - 1:
                        delay = throttle_delay(attempt, retry_after)
                        logger.warning(f"Received 429, waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                    continue
                
                # 408 / 5xx - transient server error, wait and retry
                if response.status_code == 408 or 500 <= response.status_code < 600:
                    last_response = response  # Keep for caller after exhaustion
                    if attempt < max_retries - 1:
                        delay = backoff_delay(attempt)
                        logger.warning(
                            f"Received {response.status_code}, waiting {delay:.1f}s "
                            f"(attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(delay)
                    continue
                
                # 400 because this model doesn't accept additionalModelRequestFields:
                # remember it, strip the field and retry once
                if (
                    response.status_code == 400
                    and isinstance(json_data, dict)
                    and ADDITIONAL_FIELDS_KEY in json_data
                ):
                    body = await response.aread()
                    if ADDITIONAL_FIELDS_KEY.encode() in body:
                        model_id = _payload_model_id(json_data)
                        # Remember the model only when the field itself is unsupported;
                        # a rejected value (e.g. an unknown effort) affects this request only,
                        # so later requests still honour the client's effort.
                        if model_id and b"not supported" in body.lower():
                            _ADDITIONAL_FIELDS_REJECTED.add(model_id)
                        logger.warning(
                            f"Kiro rejected {ADDITIONAL_FIELDS_KEY} for model '{model_id}': "
                            f"{body[:200].decode('utf-8', errors='replace')} - retrying without it"
                        )
                        await response.aclose()
                        json_data = {k: v for k, v in json_data.items() if k != ADDITIONAL_FIELDS_KEY}
                        continue
                
                # Other errors - return as is
                return response
                
            except httpx.TimeoutException as e:
                last_error = e
                
                # Classify timeout error for user-friendly messaging
                error_info = classify_network_error(e)
                last_error_info = error_info
                
                # Log with user-friendly message
                short_msg = get_short_error_message(error_info)
                
                if error_info.is_retryable and attempt < max_retries - 1:
                    delay = backoff_delay(attempt)
                    logger.warning(f"{short_msg} - waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"{short_msg} - no more retries (attempt {attempt + 1}/{max_retries})")
                    if not error_info.is_retryable:
                        break  # Don't retry non-retryable errors
                
            except httpx.RequestError as e:
                last_error = e
                
                # Classify the error for user-friendly messaging
                error_info = classify_network_error(e)
                last_error_info = error_info
                
                # Log with user-friendly message
                short_msg = get_short_error_message(error_info)
                
                if error_info.is_retryable and attempt < max_retries - 1:
                    delay = backoff_delay(attempt)
                    logger.warning(f"{short_msg} - waiting {delay:.1f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"{short_msg} - no more retries (attempt {attempt + 1}/{max_retries})")
                    if not error_info.is_retryable:
                        break  # Don't retry non-retryable errors
        
        # If we have a last_response (429/5xx retry exhausted), return it
        # This allows the caller to see the real status code and error body
        if last_response is not None:
            logger.warning(
                f"Retries exhausted for HTTP {last_response.status_code}, "
                f"returning response to caller for classification"
            )
            return last_response
        
        # All attempts exhausted - provide detailed, user-friendly error message
        if last_error_info:
            # Use classified error information
            error_message = last_error_info.user_message
            
            # Add troubleshooting steps
            if last_error_info.troubleshooting_steps:
                error_message += "\n\nTroubleshooting:\n"
                for i, step in enumerate(last_error_info.troubleshooting_steps, 1):
                    error_message += f"{i}. {step}\n"
            
            # Add technical details for debugging
            error_message += f"\nTechnical details: {last_error_info.technical_details}"
            
            raise HTTPException(
                status_code=last_error_info.suggested_http_code,
                detail=error_message.strip()
            )
        else:
            # Fallback if no error was captured (shouldn't happen)
            if stream:
                raise HTTPException(
                    status_code=504,
                    detail=f"Streaming failed after {max_retries} attempts. Unknown error."
                )
            else:
                raise HTTPException(
                    status_code=502,
                    detail=f"Request failed after {max_retries} attempts. Unknown error."
                )
    
    async def __aenter__(self) -> "KiroHttpClient":
        """Async context manager support."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Closes the client when exiting context."""
        await self.close()