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
Utility functions for Kiro Gateway.

Contains functions for fingerprint generation, header formatting,
and other common utilities.
"""

import hashlib
import re
import json
import uuid
from typing import TYPE_CHECKING, List, Dict, Any, Optional

from loguru import logger

if TYPE_CHECKING:
    from kiro.auth import KiroAuthManager


def get_machine_fingerprint() -> str:
    """
    Generates a unique machine fingerprint based on hostname and username.
    
    Used for User-Agent formation to identify a specific gateway installation.
    
    Returns:
        SHA256 hash of the string "{hostname}-{username}-kiro-gateway"
    """
    try:
        import socket
        import getpass
        
        hostname = socket.gethostname()
        username = getpass.getuser()
        unique_string = f"{hostname}-{username}-kiro-gateway"
        
        return hashlib.sha256(unique_string.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to get machine fingerprint: {e}")
        return hashlib.sha256(b"default-kiro-gateway").hexdigest()


def get_kiro_headers(
    auth_manager: "KiroAuthManager",
    token: str,
    amz_target: Optional[str] = "AmazonCodeWhispererStreamingService.GenerateAssistantResponse",
    attempt: int = 1,
    agent_mode: str = "vibe",
) -> dict:
    """
    Builds headers for Kiro API requests.
    
    Includes all necessary headers for authentication and identification:
    - Authorization with Bearer token
    - User-Agent with fingerprint
    - AWS CodeWhisperer specific headers
    
    Args:
        auth_manager: Authentication manager for obtaining fingerprint
        token: Access token for authorization
        amz_target: x-amz-target value, or None to omit the header (endpoint-specific)
        attempt: 1-based attempt number reported in amz-sdk-request
        agent_mode: x-amzn-kiro-agent-mode value ("vibe" or "spectask")
    
    Returns:
        Dictionary with headers for HTTP request
    """
    fingerprint = auth_manager.fingerprint
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-amz-json-1.0",
        "User-Agent": f"aws-sdk-js/1.0.27 ua/2.1 os/win32#10.0.19044 lang/js md/nodejs#22.21.1 api/codewhispererstreaming#1.0.27 m/E KiroIDE-0.7.45-{fingerprint}",
        "x-amz-user-agent": f"aws-sdk-js/1.0.27 KiroIDE-0.7.45-{fingerprint}",
        "x-amzn-codewhisperer-optout": "true",
        "x-amzn-kiro-agent-mode": agent_mode,
        "amz-sdk-invocation-id": str(uuid.uuid4()),
        "amz-sdk-request": f"attempt={attempt}; max=3",
    }
    if amz_target:
        headers["x-amz-target"] = amz_target
    return headers


def generate_completion_id() -> str:
    """
    Generates a unique ID for chat completion.
    
    Returns:
        ID in format "chatcmpl-{uuid_hex}"
    """
    return f"chatcmpl-{uuid.uuid4().hex}"


def generate_conversation_id(messages: List[Dict[str, Any]] = None) -> str:
    """
    Generates a stable conversation ID based on message history.
    
    For truncation recovery, we need a stable ID that persists across requests
    in the same conversation. This is generated from a hash of key messages.
    
    If no messages provided, falls back to random UUID (for backward compatibility).
    
    Args:
        messages: List of messages in the conversation (optional)
    
    Returns:
        Stable conversation ID (16-char hex) or random UUID
    
    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello"},
        ...     {"role": "assistant", "content": "Hi there!"}
        ... ]
        >>> conv_id = generate_conversation_id(messages)
        >>> # Same messages will always produce same ID
    """
    if not messages:
        # Fallback to random UUID for backward compatibility
        return str(uuid.uuid4())
    
    # Use first 3 messages + last message for stability
    # This ensures the ID stays the same as conversation grows,
    # but changes if the conversation history is different
    if len(messages) <= 3:
        key_messages = messages
    else:
        key_messages = messages[:3] + [messages[-1]]
    
    # Extract role and first 100 chars of content for hashing
    # This makes the hash stable even if content has minor formatting differences
    simplified_messages = []
    for msg in key_messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        # Handle different content formats (string, list, dict)
        if isinstance(content, str):
            content_str = content[:100]
        elif isinstance(content, list):
            # For Anthropic-style content blocks
            content_str = json.dumps(content, sort_keys=True)[:100]
        else:
            content_str = str(content)[:100]
        
        simplified_messages.append({
            "role": role,
            "content": content_str
        })
    
    # Generate stable hash
    content_json = json.dumps(simplified_messages, sort_keys=True)
    hash_digest = hashlib.sha256(content_json.encode()).hexdigest()
    
    # Return first 16 chars for readability (still 64 bits of entropy)
    return hash_digest[:16]


def generate_tool_call_id() -> str:
    """
    Generates a unique ID for tool call.
    
    Returns:
        ID in format "call_{uuid_hex[:8]}"
    """
    return f"call_{uuid.uuid4().hex[:8]}"

_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _is_uuid(value: str) -> bool:
    """Returns True if value is a canonical UUID string."""
    return bool(_UUID_RE.match(value or ""))


def extract_session_id(user_id: Optional[str]) -> Optional[str]:
    """
    Extracts a client session UUID from Anthropic metadata.user_id.

    Supports the formats Claude Code sends:
    - plain UUID
    - JSON string {"session_id": "..."} or {"id": "..."}
    - legacy "user_<hash>_account__session_<uuid>"

    Args:
        user_id: metadata.user_id value

    Returns:
        Lower-case session UUID, or None if none found
    """
    if not user_id or not isinstance(user_id, str):
        return None
    if _is_uuid(user_id):
        return user_id.lower()
    if user_id.lstrip().startswith("{"):
        try:
            parsed = json.loads(user_id)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            for key in ("session_id", "id"):
                candidate = parsed.get(key)
                if isinstance(candidate, str) and _is_uuid(candidate):
                    return candidate.lower()
    pos = user_id.find("session_")
    if pos != -1:
        candidate = user_id[pos + 8:pos + 8 + 36]
        if _is_uuid(candidate):
            return candidate.lower()
    return None


def derive_conversation_id(
    metadata: Optional[Dict[str, Any]],
    system: Any = None,
    tool_names: Optional[List[str]] = None,
    first_message: Any = None,
) -> str:
    """
    Derives a stable Kiro conversationId for a client conversation.

    Kiro can reuse server-side state across requests of the same conversation,
    which a random ID per request defeats. Priority:
    1. session UUID from metadata.user_id (Claude Code)
    2. SHA-256 of system + sorted tool names + first message (first 4096 chars)

    Args:
        metadata: Anthropic request metadata
        system: System prompt (str or list of blocks)
        tool_names: Declared tool names
        first_message: Content of the first message

    Returns:
        UUID-formatted conversation ID
    """
    user_id = metadata.get("user_id") if isinstance(metadata, dict) else None
    session_id = extract_session_id(user_id)
    if session_id:
        return session_id

    def _text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(value)

    # Claude Code's billing header line changes per request; keep it out of the hash
    system_text = "\n".join(
        line for line in _text(system).splitlines()
        if "x-anthropic-billing-header" not in line
    )
    hasher = hashlib.sha256()
    hasher.update(system_text.encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(",".join(sorted(tool_names or [])).encode("utf-8"))
    hasher.update(b"\x00")
    hasher.update(_text(first_message)[:4096].encode("utf-8"))
    return str(uuid.UUID(bytes=hasher.digest()[:16], version=4))
