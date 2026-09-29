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
Kiro API error enhancement and user-friendly message formatting.

This module provides a centralized system for enhancing cryptic Kiro API errors
with clear, actionable, user-friendly messages.

Architecture:
- KiroErrorReason: Enum of known error reasons from Kiro API
- KiroErrorInfo: Structured information about an enhanced error
- enhance_kiro_error(): Analyzes error JSON and returns enhanced message

Example:
    >>> error_json = {"message": "Input is too long.", "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD"}
    >>> error_info = enhance_kiro_error(error_json)
    >>> print(error_info.user_message)
    "Model context limit reached. Conversation size exceeds model capacity."
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any

from loguru import logger


@dataclass
class KiroErrorInfo:
    """
    Structured information about a Kiro API error.
    
    Contains both the enhanced user-friendly message and the original
    error details for logging and debugging.
    
    Attributes:
        reason: Error reason code from Kiro API (as string, e.g. "CONTENT_LENGTH_EXCEEDS_THRESHOLD")
        user_message: Enhanced, user-friendly message for end users
        original_message: Original message from Kiro API (for logging)
    """
    reason: str
    user_message: str
    original_message: str


def enhance_kiro_error(error_json: Dict[str, Any]) -> KiroErrorInfo:
    """
    Enhances Kiro API error with user-friendly message.
    
    Takes raw error JSON from Kiro API and returns structured information
    with enhanced, user-friendly messages that help users understand what
    went wrong without technical jargon.
    
    Args:
        error_json: Parsed JSON from Kiro API error response
                   Expected format: {"message": "...", "reason": "..."}
                   The "reason" field is optional.
    
    Returns:
        KiroErrorInfo with enhanced message and original details
    
    Example:
        >>> error_json = {"message": "Input is too long.", "reason": "CONTENT_LENGTH_EXCEEDS_THRESHOLD"}
        >>> error_info = enhance_kiro_error(error_json)
        >>> print(error_info.user_message)
        "Model context limit reached. Conversation size exceeds model capacity."
        >>> print(error_info.original_message)
        "Input is too long."
    
    Example (unknown error):
        >>> error_json = {"message": "Something went wrong.", "reason": "UNKNOWN_REASON"}
        >>> error_info = enhance_kiro_error(error_json)
        >>> print(error_info.user_message)
        "Something went wrong. (reason: UNKNOWN_REASON)"
    """
    # Extract original message and reason from Kiro API response
    # Handle None values explicitly (preserve empty strings)
    original_message = error_json.get("message")
    if original_message is None:
        original_message = "Unknown error"
    
    reason = error_json.get("reason")
    if reason is None:
        reason = "UNKNOWN"
    
    # Map known reasons to user-friendly messages
    if reason == "CONTENT_LENGTH_EXCEEDS_THRESHOLD":
        # Context limit exceeded - conversation is too long
        user_message = "Model context limit reached. Conversation size exceeds model capacity."
    
    elif reason == "MONTHLY_REQUEST_COUNT":
        # Monthly request limit exceeded - account quota exhausted
        user_message = "Monthly request limit exceeded. Account has reached its monthly quota."
    
    elif reason == "INVALID_MODEL_ID":
        # Invalid model name or subscription tier insufficient
        user_message = "Invalid model ID or insufficient subscription level to use it."

    elif original_message == "Improperly formed request." and reason in (None, "UNKNOWN", "null"):
        # Generic 400 error
        user_message = (
            "Kiro API rejected the request. If problem persists, open issue with info and attached debug logs at:"
            "https://github.com/jwadow/kiro-gateway/issues"
        )

    # Future error enhancements can be added here:
    # elif reason == "RATE_LIMIT_EXCEEDED":
    #     user_message = "Rate limit exceeded. Too many requests in a short time."
    # elif reason == "INVALID_MODEL":
    #     user_message = "Invalid model specified. The requested model is not available."
    
    else:
        # Unknown error or no enhancement available
        # Keep original message and append reason if present
        if "reason" in error_json and reason != "UNKNOWN":
            user_message = f"{original_message} (reason: {reason})"
        else:
            user_message = original_message
    
    return KiroErrorInfo(
        reason=reason,
        user_message=user_message,
        original_message=original_message
    )


@dataclass
class AnthropicErrorMapping:
    """
    How an upstream Kiro error should be presented to an Anthropic client.

    Attributes:
        status_code: HTTP status to return
        error_type: Anthropic error type (invalid_request_error, rate_limit_error, ...)
        message: Error message
        headers: Extra response headers (e.g. Retry-After)
    """
    status_code: int
    error_type: str
    message: str
    headers: Dict[str, str]


def format_prompt_too_long(estimated_tokens: int, max_tokens: int) -> str:
    """
    Formats a context overflow message in Anthropic's official wording.

    Claude Code recognises "prompt is too long: N tokens > M maximum" and can
    compact + retry; custom wording is treated as a hard failure. N is forced
    above M so the message is never self-contradictory.

    Args:
        estimated_tokens: Local estimate of the prompt size
        max_tokens: Model's max input tokens

    Returns:
        Formatted message
    """
    n = max(int(estimated_tokens or 0), int(max_tokens) + 1)
    return f"prompt is too long: {n} tokens > {int(max_tokens)} maximum"


def map_upstream_error_for_anthropic(
    status_code: int,
    reason: str,
    message: str,
    estimated_tokens: int = 0,
    max_input_tokens: int = 200000,
) -> AnthropicErrorMapping:
    """
    Maps a Kiro HTTP error to an Anthropic error response.

    - Context overflow -> 400 invalid_request_error with official wording
    - Monthly quota    -> 402 billing_error (not retryable)
    - 429              -> 429 rate_limit_error + Retry-After so clients back off
    - 5xx / 408        -> original status, overloaded_error (retryable)
    - Everything else  -> original status, api_error / invalid_request_error for 400

    Args:
        status_code: Upstream HTTP status
        reason: Kiro error reason (may be "UNKNOWN")
        message: Already-enhanced user message
        estimated_tokens: Local prompt size estimate (for overflow wording)
        max_input_tokens: Model's max input tokens

    Returns:
        AnthropicErrorMapping
    """
    if reason == "CONTENT_LENGTH_EXCEEDS_THRESHOLD":
        return AnthropicErrorMapping(
            400, "invalid_request_error",
            format_prompt_too_long(estimated_tokens, max_input_tokens), {},
        )
    if reason == "MONTHLY_REQUEST_COUNT":
        return AnthropicErrorMapping(402, "billing_error", message, {})
    if status_code == 429:
        return AnthropicErrorMapping(429, "rate_limit_error", message, {"Retry-After": "5"})
    if status_code == 408 or status_code >= 500:
        return AnthropicErrorMapping(status_code, "overloaded_error", message, {})
    if status_code == 400:
        return AnthropicErrorMapping(400, "invalid_request_error", message, {})
    return AnthropicErrorMapping(status_code, "api_error", message, {})
