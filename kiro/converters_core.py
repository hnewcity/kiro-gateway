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
Core converters for transforming API formats to Kiro format.

This module contains shared logic used by both OpenAI and Anthropic converters:
- Text content extraction from various formats
- Message merging and processing
- Kiro payload building
- Tool processing and sanitization

The core layer provides a unified interface that API-specific adapters use
to convert their formats to Kiro API format.
"""

import base64
import hashlib
import json
import re
import os
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from kiro.config import (
    TOOL_DESCRIPTION_MAX_LENGTH,
    FAKE_REASONING_ENABLED,
    FAKE_REASONING_MODE,
    FAKE_REASONING_MAX_TOKENS,
    FAKE_REASONING_BUDGET_CAP,
    KIRO_MAX_PAYLOAD_BYTES,
    AUTO_TRIM_PAYLOAD,
    KIRO_ADDITIONAL_MODEL_FIELDS,
    KIRO_DEFAULT_CLAUDE_EFFORT,
    KIRO_DEFAULT_GPT_EFFORT,
)
from kiro.payload_guards import (
    EMPTY_CONTENT_FALLBACK,
    TOOL_RESULT_ONLY_CONTENT,
    check_payload_size,
    repair_tool_pairing,
    trim_payload_to_limit,
)


# ==================================================================================================
# Converter-local Configuration
# ==================================================================================================

# Assistant reply paired with the system prompt history entry (matches Kiro CLI / reference)
SYSTEM_PROMPT_ACKNOWLEDGEMENT = "I will follow these instructions."

# Description of placeholder tool specs generated for tools referenced only in history
PLACEHOLDER_TOOL_DESCRIPTION = "Tool used in conversation history"

# Default efforts when the client does not send one (configurable in config.py)
DEFAULT_CLAUDE_EFFORT = KIRO_DEFAULT_CLAUDE_EFFORT
DEFAULT_GPT_EFFORT = KIRO_DEFAULT_GPT_EFFORT

# Kiro schema enforces max_tokens >= 1024 in additionalModelRequestFields
MIN_ADDITIONAL_MAX_TOKENS = 1024


# ==================================================================================================
# Data Classes for Unified Message Format
# ==================================================================================================

@dataclass
class ThinkingConfig:
    """
    Unified thinking configuration for fake reasoning.
    
    This configuration is created by API-specific adapters (OpenAI, Anthropic)
    and passed to the core layer for thinking tag injection.
    
    Attributes:
        enabled: Whether to inject thinking tags into the request
        budget_tokens: Token budget for thinking (None = use FAKE_REASONING_MAX_TOKENS default)
    
    Examples:
        >>> # Default configuration (enabled with default budget)
        >>> ThinkingConfig()
        ThinkingConfig(enabled=True, budget_tokens=None)
        
        >>> # Disabled by client (reasoning_effort="none" or thinking.type="disabled")
        >>> ThinkingConfig(enabled=False, budget_tokens=None)
        ThinkingConfig(enabled=False, budget_tokens=None)
        
        >>> # Custom budget from client
        >>> ThinkingConfig(enabled=True, budget_tokens=8000)
        ThinkingConfig(enabled=True, budget_tokens=8000)
    """
    enabled: bool = True
    budget_tokens: Optional[int] = None


@dataclass
class UnifiedMessage:
    """
    Unified message format used internally by converters.
    
    This format is API-agnostic and can be created from both OpenAI and Anthropic formats.
    Serves as the canonical representation for all message data before conversion to Kiro API.
    
    Attributes:
        role: Message role (user, assistant, system)
        content: Text content or list of content blocks
        tool_calls: List of tool calls (for assistant messages)
        tool_results: List of tool results (for user messages with tool responses)
        images: List of images in unified format (for multimodal user messages)
                Format: [{"media_type": "image/jpeg", "data": "base64..."}]
    """
    role: str
    content: Any = ""
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[Dict[str, Any]]] = None


@dataclass
class UnifiedTool:
    """
    Unified tool format used internally by converters.
    
    Attributes:
        name: Tool name
        description: Tool description
        input_schema: JSON Schema for tool parameters
    """
    name: str
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None


@dataclass
class KiroPayloadResult:
    """
    Result of building Kiro payload.
    
    Attributes:
        payload: The complete Kiro API payload
        tool_documentation: Documentation for tools with long descriptions (to add to system prompt)
    """
    payload: Dict[str, Any]
    tool_documentation: str = ""


# ==================================================================================================
# Text Content Extraction
# ==================================================================================================

def extract_text_content(content: Any) -> str:
    """
    Extracts text content from various formats.
    
    Supports multiple content formats used by different APIs:
    - String: "Hello, world!"
    - List of content blocks: [{"type": "text", "text": "Hello"}]
    - None: empty message
    
    Args:
        content: Content in any supported format
    
    Returns:
        Extracted text or empty string
    
    Example:
        >>> extract_text_content("Hello")
        'Hello'
        >>> extract_text_content([{"type": "text", "text": "World"}])
        'World'
        >>> extract_text_content(None)
        ''
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                # Skip image and tool_reference blocks - they're handled separately
                if item.get("type") in ("image", "image_url", "tool_reference"):
                    continue
                if item.get("type") in ("file", "input_file", "document"):
                    text_parts.append(extract_document_text_from_content_block(item))
                    continue
                if item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif "text" in item:
                    text_parts.append(item["text"])
            elif hasattr(item, "text"):
                # Handle Pydantic models like TextContentBlock
                text_parts.append(getattr(item, "text", ""))
            elif isinstance(item, str):
                text_parts.append(item)
        return "".join(text_parts)
    return str(content)


def _parse_data_url(data: str) -> Tuple[Optional[str], str]:
    """
    Parse a data URL and return (media_type, payload).

    If the value is not a data URL, the original string is returned as payload.
    """
    if not isinstance(data, str) or not data.startswith("data:"):
        return None, data

    try:
        header, payload = data.split(",", 1)
        media_type = header.split(";")[0].replace("data:", "") or None
        return media_type, payload
    except ValueError:
        return None, data


def _extract_document_source(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Extract filename, media type, and base64 data from common document blocks.
    """
    item_type = item.get("type")
    filename = item.get("filename")
    media_type = item.get("media_type")
    data = item.get("file_data") or item.get("data")

    # OpenAI chat-style file block:
    # {"type": "file", "file": {"filename": "...", "file_data": "data:application/pdf;base64,..."}}
    file_obj = item.get("file")
    if isinstance(file_obj, dict):
        filename = filename or file_obj.get("filename")
        media_type = media_type or file_obj.get("media_type")
        data = data or file_obj.get("file_data") or file_obj.get("data")

    # Anthropic-style document block:
    # {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "..."}}
    source = item.get("source")
    if isinstance(source, dict):
        filename = filename or source.get("filename")
        media_type = media_type or source.get("media_type")
        data = data or source.get("data")

    if isinstance(data, str):
        data_url_media_type, payload = _parse_data_url(data)
        media_type = media_type or data_url_media_type
        data = payload

    if item_type == "input_file":
        filename = filename or item.get("name")

    return filename, media_type, data


def _decode_base64_document(data: str) -> Optional[bytes]:
    try:
        return base64.b64decode(data, validate=False)
    except Exception as e:
        logger.warning(f"Failed to decode document base64 data: {e}")
        return None


def _extract_pdf_text(data: str) -> str:
    pdf_bytes = _decode_base64_document(data)
    if not pdf_bytes:
        return "[PDF document attached, but its base64 data could not be decoded.]"

    try:
        from pypdf import PdfReader
    except ImportError:
        return "[PDF document attached, but pypdf is not installed. Install requirements.txt to enable PDF text extraction.]"

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        text = "\n".join(part for part in pages if part).strip()
    except Exception as e:
        logger.warning(f"Failed to extract text from PDF document: {e}")
        return "[PDF document attached, but text extraction failed.]"

    if not text:
        return "[PDF document attached, but no extractable text was found.]"

    max_chars = 200000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[PDF text truncated by gateway.]"

    return text


def extract_document_text_from_content_block(item: Dict[str, Any]) -> str:
    """
    Extract text from supported document content blocks.

    Kiro accepts images natively but not OpenAI file/PDF blocks. For PDFs, the
    gateway extracts text and appends it to the user message.
    """
    filename, media_type, data = _extract_document_source(item)
    label = filename or "attached document"

    if not data:
        return f"[Document attached: {label}. No inline data was provided for extraction.]"

    normalized_media_type = (media_type or "").lower()
    if normalized_media_type == "application/pdf" or label.lower().endswith(".pdf"):
        text = _extract_pdf_text(data)
        return f"\n\n[PDF document: {label}]\n{text}\n[/PDF document]\n"

    if normalized_media_type.startswith("text/"):
        raw = _decode_base64_document(data)
        if raw is None:
            return f"[Text document attached: {label}. Base64 decoding failed.]"
        try:
            return f"\n\n[Text document: {label}]\n{raw.decode('utf-8', errors='replace')}\n[/Text document]\n"
        except Exception:
            return f"[Text document attached: {label}. Text decoding failed.]"

    return f"[Unsupported document attached: {label} ({media_type or 'unknown media type'}).]"


def extract_images_from_content(content: Any) -> List[Dict[str, Any]]:
    """
    Extracts images from message content in unified format.
    
    Supports multiple image formats used by different APIs:
    
    OpenAI format (image_url with data URL):
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/..."}}
    
    Anthropic format (image with source):
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "/9j/..."}}
    
    Args:
        content: Content in any supported format (usually a list of content blocks)
    
    Returns:
        List of images in unified format: [{"media_type": "image/jpeg", "data": "base64..."}]
        Empty list if no images found or content is not a list.
    
    Example:
        >>> extract_images_from_content([{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "abc123"}}])
        [{'media_type': 'image/png', 'data': 'abc123'}]
    """
    images: List[Dict[str, Any]] = []
    
    if not isinstance(content, list):
        return images
    
    for item in content:
        # Handle both dict and Pydantic model objects
        if isinstance(item, dict):
            item_type = item.get("type")
        elif hasattr(item, "type"):
            item_type = item.type
        else:
            continue
        
        # OpenAI format: {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        if item_type == "image_url":
            if isinstance(item, dict):
                image_url_obj = item.get("image_url", {})
            else:
                image_url_obj = getattr(item, "image_url", {})
            
            if isinstance(image_url_obj, dict):
                url = image_url_obj.get("url", "")
            elif hasattr(image_url_obj, "url"):
                url = image_url_obj.url
            else:
                url = ""
            
            if url.startswith("data:"):
                # Parse data URL: data:image/jpeg;base64,/9j/4AAQ...
                try:
                    header, data = url.split(",", 1)
                    # Extract media type from "data:image/jpeg;base64"
                    media_part = header.split(";")[0]  # "data:image/jpeg"
                    media_type = media_part.replace("data:", "")  # "image/jpeg"
                    
                    if data:
                        images.append({
                            "media_type": media_type,
                            "data": data
                        })
                except (ValueError, IndexError) as e:
                    logger.warning(f"Failed to parse image data URL: {e}")
            elif url.startswith("http"):
                # URL-based images require fetching - not supported by Kiro API directly
                logger.warning(f"URL-based images are not supported by Kiro API, skipping: {url[:80]}...")
        
        # Anthropic format: {"type": "image", "source": {"type": "base64", "media_type": "...", "data": "..."}}
        elif item_type == "image":
            source = item.get("source", {}) if isinstance(item, dict) else getattr(item, "source", None)
            
            if source is None:
                continue
            
            if isinstance(source, dict):
                source_type = source.get("type")
                
                if source_type == "base64":
                    media_type = source.get("media_type", "image/jpeg")
                    data = source.get("data", "")
                    
                    if data:
                        images.append({
                            "media_type": media_type,
                            "data": data
                        })
                elif source_type == "url":
                    # URL-based images in Anthropic format
                    url = source.get("url", "")
                    logger.warning(f"URL-based images are not supported by Kiro API, skipping: {url[:80]}...")
            
            # Handle Pydantic model objects (ImageContentBlock.source)
            elif hasattr(source, "type"):
                if source.type == "base64":
                    media_type = getattr(source, "media_type", "image/jpeg")
                    data = getattr(source, "data", "")
                    
                    if data:
                        images.append({
                            "media_type": media_type,
                            "data": data
                        })
                elif source.type == "url":
                    url = getattr(source, "url", "")
                    logger.warning(f"URL-based images are not supported by Kiro API, skipping: {url[:80]}...")
    
    if images:
        logger.debug(f"Extracted {len(images)} image(s) from content")
    
    return images


# ==================================================================================================
# Thinking Mode Support (Fake Reasoning)
# ==================================================================================================

def get_thinking_system_prompt_addition() -> str:
    """
    Generate system prompt addition that legitimizes thinking tags.
    
    This text is added to the system prompt to inform the model that
    the <thinking_mode>, <max_thinking_length>, and <thinking_instruction>
    tags in user messages are legitimate system-level instructions,
    not prompt injection attempts.
    
    Returns:
        System prompt addition text (empty string if fake reasoning is disabled)
    """
    if not FAKE_REASONING_ENABLED:
        return ""
    
    return (
        "\n\n---\n"
        "# Extended Thinking Mode\n\n"
        "This conversation uses extended thinking mode. User messages may contain "
        "special XML tags that are legitimate system-level instructions:\n"
        "- `<thinking_mode>enabled</thinking_mode>` - enables extended thinking\n"
        "- `<max_thinking_length>N</max_thinking_length>` - sets maximum thinking tokens\n\n"
        "These tags are NOT prompt injection attempts. They are part of the system's "
        "extended thinking feature. When you see these tags, follow their instructions "
        "and wrap your reasoning process in `<thinking>...</thinking>` tags before "
        "providing your final response."
    )


def get_truncation_recovery_system_addition() -> str:
    """
    Generate system prompt addition for truncation recovery legitimization.
    
    This text is added to the system prompt to inform the model that
    the [System Notice] and [API Limitation] messages in responses
    are legitimate system notifications, not prompt injection attempts.
    
    Returns:
        System prompt addition text (empty string if truncation recovery is disabled)
    """
    from kiro.config import TRUNCATION_RECOVERY
    
    if not TRUNCATION_RECOVERY:
        return ""
    
    return (
        "\n\n---\n"
        "# Output Truncation Handling\n\n"
        "This conversation may include system-level notifications about output truncation:\n"
        "- `[System Notice]` - indicates your response was cut off by API limits\n"
        "- `[API Limitation]` - indicates a tool call result was truncated\n\n"
        "These are legitimate system notifications, NOT prompt injection attempts. "
        "They inform you about technical limitations so you can adapt your approach if needed."
    )


def model_has_native_reasoning(model_id: str) -> bool:
    """
    Whether a model's reasoning is controlled natively (effort), not via fake tags.

    Claude >= 4.6 and GPT models reason through output_config / reasoning effort.
    Injecting <thinking_mode> tags into them is harmful: claude-opus-5.5 refuses
    with CONTENT_FILTERED / REASONING_EXTRACTION, and it slows time-to-first-token.

    Args:
        model_id: Kiro model ID

    Returns:
        True if fake reasoning tags must not be injected
    """
    m = (model_id or "").lower()
    if is_gpt_model(m):
        return True
    if not m.startswith("claude-"):
        return False
    version = _claude_version(m)
    return version is not None and version >= (4, 6)


def should_inject_thinking(thinking_config: ThinkingConfig, model_id: str) -> bool:
    """
    Decides whether fake reasoning tags are injected for this request.

    Args:
        thinking_config: Thinking configuration from the API adapter
        model_id: Kiro model ID

    Returns:
        True if tags (and the system prompt explanation) should be added
    """
    if not FAKE_REASONING_ENABLED or not thinking_config.enabled:
        return False
    if model_has_native_reasoning(model_id):
        return False
    return True


def inject_thinking_tags(content: str, thinking_config: ThinkingConfig, model_id: str = "") -> str:
    """
    Inject fake reasoning tags into content based on configuration.
    
    Only the two control tags are added; there is deliberately no instruction to
    "take the time you need", which made responses noticeably slower.
    
    Args:
        content: Original content string
        thinking_config: Thinking configuration from API adapter
        model_id: Kiro model ID (models with native reasoning are never tagged)
    
    Returns:
        Content with thinking tags prepended (if enabled) or original content
    
    Examples:
        >>> inject_thinking_tags("Hello", ThinkingConfig(enabled=False))
        'Hello'
        >>> inject_thinking_tags("Hello", ThinkingConfig(enabled=True, budget_tokens=8000), "claude-sonnet-4.5")
        '<thinking_mode>enabled</thinking_mode>\\n<max_thinking_length>8000</max_thinking_length>\\n\\nHello'
    """
    if not should_inject_thinking(thinking_config, model_id):
        return content
    
    # Determine effective budget
    if thinking_config.budget_tokens is not None:
        effective_budget = thinking_config.budget_tokens
    else:
        effective_budget = FAKE_REASONING_MAX_TOKENS
    
    # Apply cap if enabled
    if FAKE_REASONING_BUDGET_CAP > 0 and effective_budget > FAKE_REASONING_BUDGET_CAP:
        logger.warning(
            f"Client requested thinking budget {effective_budget} exceeds cap {FAKE_REASONING_BUDGET_CAP}, "
            f"using capped value"
        )
        effective_budget = FAKE_REASONING_BUDGET_CAP
    
    thinking_prefix = (
        f"<thinking_mode>enabled</thinking_mode>\n"
        f"<max_thinking_length>{effective_budget}</max_thinking_length>\n\n"
    )
    
    logger.debug(f"Injecting thinking tags with budget={effective_budget}")
    
    return thinking_prefix + content


# ==================================================================================================
# JSON Schema Sanitization
# ==================================================================================================

_SCHEMA_BASIC_TYPES = ("object", "array", "string", "number", "integer", "boolean")
_SCHEMA_MAX_REF_DEPTH = 16
_SCHEMA_DESCRIPTION_MAX_CHARS = 2000
_SCHEMA_COMBINATOR_KEYS = ("anyOf", "oneOf", "allOf")


def _extract_schema_defs(schema: Any) -> Dict[str, Any]:
    """
    Collect top-level `$defs` / `definitions` as the `$ref` resolution table.

    Args:
        schema: Root JSON Schema

    Returns:
        Mapping of definition name to schema
    """
    defs: Dict[str, Any] = {}
    if isinstance(schema, dict):
        for key in ("$defs", "definitions"):
            table = schema.get(key)
            if isinstance(table, dict):
                defs.update(table)
    return defs


def _resolve_schema_refs(value: Any, defs: Dict[str, Any], depth: int) -> Any:
    """
    Recursively inline `#/$defs/<name>` and `#/definitions/<name>` references.

    Sibling keys next to `$ref` (e.g. description) win over the target's keys.
    Unresolvable refs (external URLs, OpenAPI paths, missing defs) and refs beyond
    the depth limit (cycles) degrade to a permissive object schema.

    Args:
        value: Schema node
        defs: Definition table from the root schema
        depth: Current `$ref` expansion depth

    Returns:
        Schema node without `$ref`
    """
    if depth > _SCHEMA_MAX_REF_DEPTH:
        return {"type": "object"}

    if isinstance(value, list):
        return [_resolve_schema_refs(item, defs, depth) for item in value]
    if not isinstance(value, dict):
        return value

    obj = dict(value)
    ref = obj.pop("$ref", None)
    if isinstance(ref, str):
        name: Optional[str] = None
        for prefix in ("#/$defs/", "#/definitions/"):
            if ref.startswith(prefix):
                name = ref[len(prefix):]
                break
        target = defs.get(name) if name is not None else None
        if isinstance(target, dict):
            resolved = _resolve_schema_refs(target, defs, depth + 1)
            if isinstance(resolved, dict):
                for key, val in resolved.items():
                    obj.setdefault(key, val)
        else:
            logger.debug(f"Unresolvable $ref in tool schema, degrading to permissive object: {ref}")
            obj.setdefault("type", "object")

    return {key: _resolve_schema_refs(val, defs, depth) for key, val in obj.items()}


def _normalize_schema_type(raw: Any) -> Optional[str]:
    """
    Normalize a JSON Schema `type` value to a single basic type.

    Args:
        raw: `type` value (string or list of strings)

    Returns:
        Basic type name, or None if no supported type is present
    """
    candidates = raw if isinstance(raw, list) else [raw]
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip() in _SCHEMA_BASIC_TYPES:
            return candidate.strip()
    return None


def _collapse_schema_combinators(obj: Dict[str, Any]) -> None:
    """
    Remove anyOf/oneOf/allOf, keeping the single real branch when there is one.

    Kiro rejects combinator schemas, so the reference drops them. As a strict
    superset, when a combinator has exactly one non-null branch (the common
    pydantic `Optional[X]` / `allOf: [{$ref}]` shapes) that branch's keys are
    merged in (existing keys win) so the constraint is not lost.

    Args:
        obj: Schema object (mutated in place)
    """
    for key in _SCHEMA_COMBINATOR_KEYS:
        variants = obj.pop(key, None)
        if not isinstance(variants, list):
            continue
        branches = [
            v for v in variants
            if isinstance(v, dict) and v.get("type") != "null"
        ]
        if len(branches) == 1:
            for branch_key, branch_value in branches[0].items():
                if branch_key not in _SCHEMA_COMBINATOR_KEYS:
                    obj.setdefault(branch_key, branch_value)


def _normalize_schema_node(schema: Any, root: bool) -> Dict[str, Any]:
    """
    Normalize one schema node into the strict subset Kiro accepts.

    Args:
        schema: Schema node (refs already resolved)
        root: Whether this is the root (always an object schema)

    Returns:
        Normalized schema with only type/properties/required/items/description/enum
    """
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    obj = {key: val for key, val in schema.items() if val is not None}
    _collapse_schema_combinators(obj)
    obj = {key: val for key, val in obj.items() if val is not None}

    normalized_type = _normalize_schema_type(obj.get("type"))
    is_object = root or normalized_type == "object" or (
        normalized_type is None and "properties" in obj
    )

    result: Dict[str, Any] = {}
    if is_object:
        result["type"] = "object"
    elif normalized_type:
        result["type"] = normalized_type

    if is_object:
        raw_props = obj.get("properties")
        properties: Dict[str, Any] = {}
        if isinstance(raw_props, dict):
            for prop_name, prop_schema in raw_props.items():
                properties[str(prop_name)] = _normalize_schema_node(prop_schema, False)
        result["properties"] = properties

        raw_required = obj.get("required")
        required: List[str] = []
        if isinstance(raw_required, list):
            for name in raw_required:
                if isinstance(name, str) and name in properties and name not in required:
                    required.append(name)
        # Kiro rejects empty required arrays - only emit when non-empty
        if required:
            result["required"] = required

    items = obj.get("items")
    if isinstance(items, list):
        first = next((item for item in items if isinstance(item, dict)), None)
        if first is not None:
            result["items"] = _normalize_schema_node(first, False)
    elif isinstance(items, dict):
        result["items"] = _normalize_schema_node(items, False)

    description = obj.get("description")
    if isinstance(description, str):
        result["description"] = description[:_SCHEMA_DESCRIPTION_MAX_CHARS]

    enum_values = obj.get("enum")
    if isinstance(enum_values, list):
        scalars = [v for v in enum_values if isinstance(v, (str, int, float, bool))]
        if scalars:
            result["enum"] = scalars

    # additionalProperties and all other keywords are dropped (Kiro strict mode)
    return result


def sanitize_json_schema(schema: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalizes a tool JSON Schema into the strict subset the Kiro API accepts.

    Kiro returns 400 "Improperly formed request" for many valid JSON Schemas.
    Port of kiro2cc-proxy schema.rs:
    - `$ref` resolved against `$defs`/`definitions` (depth limit 16); unresolvable
      refs become a permissive object; `$defs`/`definitions` dropped
    - null values stripped; `type` arrays reduced to the first non-null basic type
    - object schemas always get a dict `properties`; `required` keeps only strings
      that exist in `properties` and is omitted when empty
    - `items` must be a schema (first schema of a tuple list)
    - anyOf/oneOf/allOf removed (single non-null branch merged in)
    - keys whitelisted to type/properties/required/items/description/enum
      (so `additionalProperties` is always dropped)
    - the root is always an object schema

    Args:
        schema: JSON Schema to sanitize

    Returns:
        Sanitized copy of schema (never mutates the input)
    """
    defs = _extract_schema_defs(schema)
    resolved = _resolve_schema_refs(schema if schema is not None else {}, defs, 0)
    return _normalize_schema_node(resolved, True)


# ==================================================================================================
# Tool Processing
# ==================================================================================================

def process_tools_with_long_descriptions(
    tools: Optional[List[UnifiedTool]]
) -> Tuple[Optional[List[UnifiedTool]], str]:
    """
    Processes tools with long descriptions.
    
    Kiro API has a limit on description length in toolSpecification.
    If description exceeds the limit, full description is moved to system prompt,
    and a reference to documentation remains in the tool.
    
    Args:
        tools: List of tools in unified format
    
    Returns:
        Tuple of:
        - List of tools with processed descriptions (or None if tools is empty)
        - String with documentation to add to system prompt (empty if all descriptions are short)
    """
    if not tools:
        return None, ""
    
    # If limit is disabled (0), return tools unchanged
    if TOOL_DESCRIPTION_MAX_LENGTH <= 0:
        return tools, ""
    
    tool_documentation_parts = []
    processed_tools = []
    
    for tool in tools:
        description = tool.description or ""
        
        if len(description) <= TOOL_DESCRIPTION_MAX_LENGTH:
            # Description is short - leave as is
            processed_tools.append(tool)
        else:
            # Description is too long - move to system prompt
            logger.debug(
                f"Tool '{tool.name}' has long description ({len(description)} chars > {TOOL_DESCRIPTION_MAX_LENGTH}), "
                f"moving to system prompt"
            )
            
            # Create documentation for system prompt
            tool_documentation_parts.append(f"## Tool: {tool.name}\n\n{description}")
            
            # Create copy of tool with reference description
            reference_description = f"[Full documentation in system prompt under '## Tool: {tool.name}']"
            
            processed_tool = UnifiedTool(
                name=tool.name,
                description=reference_description,
                input_schema=tool.input_schema
            )
            processed_tools.append(processed_tool)
    
    # Form final documentation
    tool_documentation = ""
    if tool_documentation_parts:
        tool_documentation = (
            "\n\n---\n"
            "# Tool Documentation\n"
            "The following tools have detailed documentation that couldn't fit in the tool definition.\n\n"
            + "\n\n---\n\n".join(tool_documentation_parts)
        )
    
    return processed_tools if processed_tools else None, tool_documentation


def validate_tool_names(tools: Optional[List[UnifiedTool]]) -> None:
    """
    Validates tool names against Kiro API 64-character limit.
    
    Logs WARNING for each problematic tool and raises ValueError
    with complete list of violations.
    
    Args:
        tools: List of tools to validate
    
    Raises:
        ValueError: If any tool name exceeds 64 characters
    
    Example:
        >>> validate_tool_names([UnifiedTool(name="short_name", description="test")])
        # No error
        >>> validate_tool_names([UnifiedTool(name="a" * 70, description="test")])
        # Raises ValueError with detailed message
    """
    if not tools:
        return
    
    problematic_tools = []
    for tool in tools:
        if len(tool.name) > 64:
            problematic_tools.append((tool.name, len(tool.name)))
    
    if problematic_tools:
        # Build detailed error message for client (no logging here - routes will log)
        tool_list = "\n".join([
            f"  - '{name}' ({length} characters)"
            for name, length in problematic_tools
        ])
        
        raise ValueError(
            f"Tool name(s) exceed Kiro API limit of 64 characters:\n"
            f"{tool_list}\n\n"
            f"Solution: Use shorter tool names (max 64 characters).\n"
            f"Example: 'get_user_data' instead of 'get_authenticated_user_profile_data_with_extended_information_about_it'"
        )


def convert_tools_to_kiro_format(tools: Optional[List[UnifiedTool]]) -> List[Dict[str, Any]]:
    """
    Converts unified tools to Kiro API format.
    
    Args:
        tools: List of tools in unified format
    
    Returns:
        List of tools in Kiro toolSpecification format
    """
    if not tools:
        return []
    
    kiro_tools = []
    for tool in tools:
        # Sanitize parameters from fields that Kiro API doesn't accept
        sanitized_params = sanitize_json_schema(tool.input_schema)
        
        # Kiro API requires non-empty description
        description = tool.description
        if not description or not description.strip():
            description = f"Tool: {tool.name}"
            logger.debug(f"Tool '{tool.name}' has empty description, using placeholder")
        
        kiro_tools.append({
            "toolSpecification": {
                "name": tool.name,
                "description": description,
                "inputSchema": {"json": sanitized_params}
            }
        })
    
    return kiro_tools


# ==================================================================================================
# Image Conversion to Kiro Format
# ==================================================================================================

def convert_images_to_kiro_format(images: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """
    Converts unified images to Kiro API format.
    
    Unified format: [{"media_type": "image/jpeg", "data": "base64..."}]
    Kiro format: [{"format": "jpeg", "source": {"bytes": "base64..."}}]
    
    IMPORTANT: Images must be placed directly in userInputMessage.images,
    NOT in userInputMessageContext.images. This matches the native Kiro IDE format.
    
    Also handles the case where data contains a full data URL (data:image/jpeg;base64,...)
    by stripping the prefix and extracting pure base64.
    
    Args:
        images: List of images in unified format
    
    Returns:
        List of images in Kiro format, ready for userInputMessage.images
    
    Example:
        >>> convert_images_to_kiro_format([{"media_type": "image/png", "data": "abc123"}])
        [{'format': 'png', 'source': {'bytes': 'abc123'}}]
    """
    if not images:
        return []
    
    kiro_images = []
    for img in images:
        media_type = img.get("media_type", "image/jpeg")
        data = img.get("data", "")
        
        if not data:
            logger.warning("Skipping image with empty data")
            continue
        
        # Strip data URL prefix if present (some clients send "data:image/jpeg;base64,..." in data field)
        # Kiro API expects pure base64 without the prefix
        if data.startswith("data:"):
            try:
                header, actual_data = data.split(",", 1)
                # Extract media type from header if present
                media_part = header.split(";")[0]  # "data:image/jpeg"
                extracted_media_type = media_part.replace("data:", "")
                if extracted_media_type:
                    media_type = extracted_media_type
                data = actual_data
                logger.debug(f"Stripped data URL prefix, extracted media_type: {media_type}")
            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse data URL prefix: {e}")
        
        # Extract format from media_type: "image/jpeg" -> "jpeg"
        format_str = media_type.split("/")[-1] if "/" in media_type else media_type
        
        kiro_images.append({
            "format": format_str,
            "source": {
                "bytes": data
            }
        })
    
    if kiro_images:
        logger.debug(f"Converted {len(kiro_images)} image(s) to Kiro format")
    
    return kiro_images


# ==================================================================================================
# Tool Results and Tool Uses Extraction
# ==================================================================================================

def convert_tool_results_to_kiro_format(tool_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Converts unified tool results to Kiro API format.
    
    Unified format: {"type": "tool_result", "tool_use_id": "...", "content": "..."}
    Kiro format: {"content": [{"text": "..."}], "status": "success", "toolUseId": "..."}
    
    Args:
        tool_results: List of tool results in unified format
    
    Returns:
        List of tool results in Kiro format
    """
    kiro_results = []
    for tr in tool_results:
        content = tr.get("content", "")
        if isinstance(content, str):
            content_text = content
        else:
            content_text = extract_text_content(content)
        
        # Ensure content is not empty - Kiro API requires non-empty content
        if not content_text:
            content_text = "(empty result)"
        
        kiro_results.append({
            "content": [{"text": content_text}],
            "status": "success",
            "toolUseId": tr.get("tool_use_id", "")
        })
    
    return kiro_results


def extract_tool_results_from_content(content: Any) -> List[Dict[str, Any]]:
    """
    Extracts tool results from message content.
    
    Looks for content blocks with type="tool_result" and converts them
    to Kiro API format.
    
    Args:
        content: Message content (can be a list of content blocks)
    
    Returns:
        List of tool results in Kiro format
    """
    tool_results = []
    
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_result":
                tool_results.append({
                    "content": [{"text": extract_text_content(item.get("content", "")) or "(empty result)"}],
                    "status": "success",
                    "toolUseId": item.get("tool_use_id", "")
                })
    
    return tool_results


def extract_tool_uses_from_message(
    content: Any,
    tool_calls: Optional[List[Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Extracts tool uses from assistant message.
    
    Looks for tool calls in both:
    - tool_calls field (OpenAI format)
    - content blocks with type="tool_use" (Anthropic format)
    
    Args:
        content: Message content
        tool_calls: List of tool calls (OpenAI format)
    
    Returns:
        List of tool uses in Kiro format
    """
    tool_uses = []
    
    # From tool_calls field (OpenAI format or unified format from Anthropic)
    if tool_calls:
        for tc in tool_calls:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                arguments = func.get("arguments", "{}")
                # Handle both string (OpenAI) and dict (Anthropic unified) formats
                if isinstance(arguments, str):
                    input_data = json.loads(arguments) if arguments else {}
                else:
                    input_data = arguments if arguments else {}
                tool_uses.append({
                    "name": func.get("name", ""),
                    "input": input_data,
                    "toolUseId": tc.get("id", "")
                })
    
    # From content blocks (Anthropic format)
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and item.get("type") == "tool_use":
                tool_uses.append({
                    "name": item.get("name", ""),
                    "input": item.get("input", {}),
                    "toolUseId": item.get("id", "")
                })
    
    return tool_uses


# ==================================================================================================
# Tool Content to Text Conversion (for stripping when no tools defined)
# ==================================================================================================

def tool_calls_to_text(tool_calls: List[Dict[str, Any]]) -> str:
    """
    Converts tool_calls to human-readable text representation.
    
    This is used when stripping tool content from messages (when no tools are defined).
    Instead of losing the context, we convert tool calls to text so the model
    can still understand what happened in the conversation.
    
    Args:
        tool_calls: List of tool calls in unified format
    
    Returns:
        Text representation of tool calls
    
    Example:
        >>> tool_calls_to_text([{"id": "call_123", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}])
        '[Tool: bash] (call_123)\\n{"command": "ls"}'
    """
    if not tool_calls:
        return ""
    
    parts = []
    for tc in tool_calls:
        func = tc.get("function", {})
        name = func.get("name", "unknown")
        arguments = func.get("arguments", "{}")
        tool_id = tc.get("id", "")
        
        # Format: [Tool: name] (id)\narguments
        if tool_id:
            parts.append(f"[Tool: {name} ({tool_id})]\n{arguments}")
        else:
            parts.append(f"[Tool: {name}]\n{arguments}")
    
    return "\n\n".join(parts)


def tool_results_to_text(tool_results: List[Dict[str, Any]]) -> str:
    """
    Converts tool_results to human-readable text representation.
    
    This is used when stripping tool content from messages (when no tools are defined).
    Instead of losing the context, we convert tool results to text so the model
    can still understand what happened in the conversation.
    
    Args:
        tool_results: List of tool results in unified format
    
    Returns:
        Text representation of tool results
    
    Example:
        >>> tool_results_to_text([{"tool_use_id": "call_123", "content": "file1.txt\\nfile2.txt"}])
        '[Tool Result] (call_123)\\nfile1.txt\\nfile2.txt'
    """
    if not tool_results:
        return ""
    
    parts = []
    for tr in tool_results:
        content = tr.get("content", "")
        tool_use_id = tr.get("tool_use_id", "")
        
        if isinstance(content, str):
            content_text = content
        else:
            content_text = extract_text_content(content)
        
        # Use placeholder if content is empty
        if not content_text:
            content_text = "(empty result)"
        
        # Format: [Tool Result] (id)\ncontent
        if tool_use_id:
            parts.append(f"[Tool Result ({tool_use_id})]\n{content_text}")
        else:
            parts.append(f"[Tool Result]\n{content_text}")
    
    return "\n\n".join(parts)


# ==================================================================================================
# Message Merging
# ==================================================================================================

def strip_all_tool_content(messages: List[UnifiedMessage]) -> Tuple[List[UnifiedMessage], bool]:
    """
    Strips ALL tool-related content from messages, converting it to text representation.
    
    This is used when no tools are defined in the request. Kiro API rejects
    requests that have toolResults but no tools defined.
    
    Instead of simply removing tool content, this function converts tool_calls
    and tool_results to human-readable text, preserving the context for
    summarization and other use cases.
    
    Args:
        messages: List of messages in unified format
    
    Returns:
        Tuple of:
        - List of messages with tool content converted to text
        - Boolean indicating whether any tool content was converted
    """
    if not messages:
        return [], False
    
    result = []
    total_tool_calls_stripped = 0
    total_tool_results_stripped = 0
    
    for msg in messages:
        # Check if this message has any tool content
        has_tool_calls = bool(msg.tool_calls)
        has_tool_results = bool(msg.tool_results)
        
        if has_tool_calls or has_tool_results:
            if has_tool_calls:
                total_tool_calls_stripped += len(msg.tool_calls)
            if has_tool_results:
                total_tool_results_stripped += len(msg.tool_results)
            
            # Start with existing text content
            existing_content = extract_text_content(msg.content)
            content_parts = []
            
            if existing_content:
                content_parts.append(existing_content)
            
            # Convert tool_calls to text (for assistant messages)
            if has_tool_calls:
                tool_text = tool_calls_to_text(msg.tool_calls)
                if tool_text:
                    content_parts.append(tool_text)
            
            # Convert tool_results to text (for user messages)
            if has_tool_results:
                result_text = tool_results_to_text(msg.tool_results)
                if result_text:
                    content_parts.append(result_text)
            
            # Join all parts with double newline
            content = "\n\n".join(content_parts) if content_parts else "(empty placeholder)"
            
            # Create a copy of the message without tool content but with text representation
            # IMPORTANT: Preserve images from the original message (e.g., screenshots from MCP tools)
            cleaned_msg = UnifiedMessage(
                role=msg.role,
                content=content,
                tool_calls=None,
                tool_results=None,
                images=msg.images
            )
            result.append(cleaned_msg)
        else:
            result.append(msg)
    
    had_tool_content = total_tool_calls_stripped > 0 or total_tool_results_stripped > 0
    
    # Log summary once (DEBUG level - this is normal for clients like Cline/Roo/Cursor)
    if had_tool_content:
        logger.debug(
            f"Converted tool content to text (no tools defined): "
            f"{total_tool_calls_stripped} tool_calls, {total_tool_results_stripped} tool_results"
        )
    
    return result, had_tool_content


def ensure_assistant_before_tool_results(messages: List[UnifiedMessage]) -> Tuple[List[UnifiedMessage], bool]:
    """
    Ensures that messages with tool_results have a preceding assistant message with tool_calls.
    
    Kiro API requires that when toolResults are present, there must be a preceding
    assistantResponseMessage with toolUses. Some clients (like Cline/Roo/Cursor) may send
    truncated conversations where the assistant message is missing.
    
    Since we don't know the original tool name and arguments when the assistant message
    is missing, we cannot create a valid synthetic assistant message. Instead, we convert
    the tool_results to text representation and append to the message content, preserving
    the context for the model while avoiding Kiro API rejection.
    
    Args:
        messages: List of messages in unified format
    
    Returns:
        Tuple of:
        - List of messages with orphaned tool_results converted to text
        - Boolean indicating whether any tool_results were converted (used to skip thinking tag injection)
    """
    if not messages:
        return [], False
    
    result = []
    converted_any_tool_results = False
    
    for msg in messages:
        # Check if this message has tool_results
        if msg.tool_results:
            # Check if the previous message is an assistant with tool_calls
            has_preceding_assistant = (
                result and
                result[-1].role == "assistant" and
                result[-1].tool_calls
            )
            
            if not has_preceding_assistant:
                # We cannot create a valid synthetic assistant message because we don't know
                # the original tool name and arguments. Kiro API validates tool names.
                # Convert tool_results to text to preserve context for the model.
                logger.debug(
                    f"Converting {len(msg.tool_results)} orphaned tool_results to text "
                    f"(no preceding assistant message with tool_calls). "
                    f"Tool IDs: {[tr.get('tool_use_id', 'unknown') for tr in msg.tool_results]}"
                )
                
                # Convert tool_results to text representation
                tool_results_text = tool_results_to_text(msg.tool_results)
                
                # Append to existing content
                original_content = extract_text_content(msg.content) or ""
                if original_content and tool_results_text:
                    new_content = f"{original_content}\n\n{tool_results_text}"
                elif tool_results_text:
                    new_content = tool_results_text
                else:
                    new_content = original_content
                
                # Create a copy of the message with tool_results converted to text
                cleaned_msg = UnifiedMessage(
                    role=msg.role,
                    content=new_content,
                    tool_calls=msg.tool_calls,
                    tool_results=None,  # Remove orphaned tool_results (now in text)
                    images=msg.images
                )
                result.append(cleaned_msg)
                converted_any_tool_results = True
                continue
        
        result.append(msg)
    
    return result, converted_any_tool_results


def merge_adjacent_messages(messages: List[UnifiedMessage]) -> List[UnifiedMessage]:
    """
    Merges adjacent messages with the same role.
    
    Kiro API does not accept multiple consecutive messages from the same role.
    This function merges such messages into one.
    
    Args:
        messages: List of messages in unified format
    
    Returns:
        List of messages with merged adjacent messages
    """
    if not messages:
        return []
    
    merged = []
    # Statistics for summary logging
    merge_counts = {"user": 0, "assistant": 0}
    total_tool_calls_merged = 0
    total_tool_results_merged = 0
    
    for msg in messages:
        if not merged:
            merged.append(msg)
            continue
        
        last = merged[-1]
        if msg.role == last.role:
            # Merge content
            if isinstance(last.content, list) and isinstance(msg.content, list):
                last.content = last.content + msg.content
            elif isinstance(last.content, list):
                last.content = last.content + [{"type": "text", "text": extract_text_content(msg.content)}]
            elif isinstance(msg.content, list):
                last.content = [{"type": "text", "text": extract_text_content(last.content)}] + msg.content
            else:
                last_text = extract_text_content(last.content)
                current_text = extract_text_content(msg.content)
                last.content = f"{last_text}\n{current_text}"
            
            # Merge tool_calls for assistant messages
            if msg.role == "assistant" and msg.tool_calls:
                if last.tool_calls is None:
                    last.tool_calls = []
                last.tool_calls = list(last.tool_calls) + list(msg.tool_calls)
                total_tool_calls_merged += len(msg.tool_calls)
            
            # Merge tool_results for user messages
            if msg.role == "user" and msg.tool_results:
                if last.tool_results is None:
                    last.tool_results = []
                last.tool_results = list(last.tool_results) + list(msg.tool_results)
                total_tool_results_merged += len(msg.tool_results)
            
            # Count merges by role
            if msg.role in merge_counts:
                merge_counts[msg.role] += 1
        else:
            merged.append(msg)
    
    # Log summary if any merges occurred
    total_merges = sum(merge_counts.values())
    if total_merges > 0:
        parts = []
        for role, count in merge_counts.items():
            if count > 0:
                parts.append(f"{count} {role}")
        merge_summary = ", ".join(parts)
        
        extras = []
        if total_tool_calls_merged > 0:
            extras.append(f"{total_tool_calls_merged} tool_calls")
        if total_tool_results_merged > 0:
            extras.append(f"{total_tool_results_merged} tool_results")
        
        if extras:
            logger.debug(f"Merged {total_merges} adjacent messages ({merge_summary}), including {', '.join(extras)}")
        else:
            logger.debug(f"Merged {total_merges} adjacent messages ({merge_summary})")
    
    return merged


def ensure_first_message_is_user(messages: List[UnifiedMessage]) -> List[UnifiedMessage]:
    """
    Ensures that the first message in the conversation is from user role.
    
    Kiro API requires conversations to start with a user message. If the first
    message is from assistant (or any other non-user role), we prepend a minimal
    synthetic user message.
    
    This matches LiteLLM behavior for Anthropic API compatibility and fixes
    issue #60 where conversations starting with assistant messages cause
    "Improperly formed request" errors.
    
    Args:
        messages: List of messages in unified format
    
    Returns:
        List of messages with guaranteed user-first order
    
    Example:
        >>> messages = [
        ...     UnifiedMessage(role="assistant", content="Hello"),
        ...     UnifiedMessage(role="user", content="Hi")
        ... ]
        >>> result = ensure_first_message_is_user(messages)
        >>> result[0].role
        'user'
        >>> result[0].content
        '(empty placeholder)'
    """
    if not messages:
        return messages
    
    if messages[0].role != "user":
        logger.debug(
            f"First message is '{messages[0].role}', prepending synthetic user message "
            f"(Kiro API requires conversations to start with user)"
        )
        # Create minimal synthetic user message (matches LiteLLM behavior)
        # Using "(empty placeholder)" as minimal valid content to avoid disrupting conversation context
        synthetic_user = UnifiedMessage(
            role="user",
            content="(empty placeholder)"
        )
        
        return [synthetic_user] + messages
    
    return messages


def normalize_message_roles(messages: List[UnifiedMessage]) -> List[UnifiedMessage]:
    """
    Normalizes unknown message roles to 'user'.
    
    Kiro API only supports 'user' and 'assistant' roles in history.
    Any other role (e.g., 'developer', 'system') is converted to 'user'
    to maintain compatibility.
    
    This normalization MUST happen before ensure_alternating_roles()
    to ensure consecutive messages with unknown roles are properly detected
    and synthetic assistant messages are inserted between them.
    
    Args:
        messages: List of messages in unified format
    
    Returns:
        List of messages with normalized roles
    
    Example:
        >>> messages = [
        ...     UnifiedMessage(role="developer", content="Context 1"),
        ...     UnifiedMessage(role="developer", content="Context 2"),
        ...     UnifiedMessage(role="user", content="Question")
        ... ]
        >>> result = normalize_message_roles(messages)
        >>> [msg.role for msg in result]
        ['user', 'user', 'user']
    """
    if not messages:
        return messages
    
    normalized = []
    converted_count = 0
    
    for msg in messages:
        if msg.role not in ("user", "assistant"):
            logger.debug(f"Normalizing role '{msg.role}' to 'user'")
            normalized_msg = UnifiedMessage(
                role="user",
                content=msg.content,
                tool_calls=msg.tool_calls,
                tool_results=msg.tool_results,
                images=msg.images
            )
            normalized.append(normalized_msg)
            converted_count += 1
        else:
            normalized.append(msg)
    
    if converted_count > 0:
        logger.debug(f"Normalized {converted_count} message(s) with unknown roles to 'user'")
    
    return normalized


def ensure_alternating_roles(messages: List[UnifiedMessage]) -> List[UnifiedMessage]:
    """
    Ensures alternating user/assistant roles by inserting synthetic assistant messages.
    
    Kiro API requires alternating userInputMessage and assistantResponseMessage.
    When consecutive user messages are detected, synthetic assistant messages
    with "(empty placeholder)" placeholder are inserted between them to maintain alternation.
    
    This fixes multiple unknown roles (converted to user)
    create consecutive userInputMessage entries that violate Kiro API requirements.
    
    Args:
        messages: List of messages in unified format
    
    Returns:
        List of messages with synthetic assistant messages inserted where needed
    
    Example:
        >>> messages = [
        ...     UnifiedMessage(role="user", content="First"),
        ...     UnifiedMessage(role="user", content="Second"),
        ...     UnifiedMessage(role="user", content="Third")
        ... ]
        >>> result = ensure_alternating_roles(messages)
        >>> len(result)
        5  # 3 user + 2 synthetic assistant
        >>> result[1].role
        'assistant'
        >>> result[1].content
        '(empty placeholder)'
    """
    if not messages or len(messages) < 2:
        return messages
    
    result = [messages[0]]
    synthetic_count = 0
    
    for msg in messages[1:]:
        prev_role = result[-1].role
        
        # If both current and previous are user → insert synthetic assistant
        if msg.role == "user" and prev_role == "user":
            synthetic_assistant = UnifiedMessage(
                role="assistant",
                content="(empty placeholder)"  # Consistent with build_kiro_history() placeholder
            )
            result.append(synthetic_assistant)
            synthetic_count += 1
        
        result.append(msg)
    
    if synthetic_count > 0:
        logger.debug(f"Inserted {synthetic_count} synthetic assistant message(s) to ensure alternation")
    
    return result


# ==================================================================================================
# Kiro History Building
# ==================================================================================================

def build_kiro_history(messages: List[UnifiedMessage], model_id: str) -> List[Dict[str, Any]]:
    """
    Builds history array for Kiro API from unified messages.
    
    Kiro API expects alternating userInputMessage and assistantResponseMessage.
    This function converts unified format to Kiro format.
    
    All messages should have 'user' or 'assistant' roles at this point,
    as unknown roles are normalized earlier in the pipeline by normalize_message_roles().
    
    Args:
        messages: List of messages in unified format (with normalized roles)
        model_id: Internal Kiro model ID
    
    Returns:
        List of dictionaries for history field in Kiro API
    """
    history = []
    
    for msg in messages:
        if msg.role == "user":
            content = extract_text_content(msg.content)
            
            # Fallback for empty content - Kiro API requires non-empty content.
            # A user turn that only carries tool results gets a neutral hint so the
            # model answers based on the results instead of reading it as "continue".
            if not content:
                has_results = bool(msg.tool_results) or bool(
                    extract_tool_results_from_content(msg.content)
                )
                content = TOOL_RESULT_ONLY_CONTENT if has_results else EMPTY_CONTENT_FALLBACK
            
            user_input = {
                "content": content,
                "modelId": model_id,
                "origin": "AI_EDITOR",
            }
            
            # Process images - extract from message or content
            # IMPORTANT: images go directly into userInputMessage, NOT into userInputMessageContext
            # This matches the native Kiro IDE format
            images = msg.images or extract_images_from_content(msg.content)
            if images:
                kiro_images = convert_images_to_kiro_format(images)
                if kiro_images:
                    user_input["images"] = kiro_images
            
            # Build userInputMessageContext for tools and toolResults only
            user_input_context: Dict[str, Any] = {}
            
            # Process tool_results - convert to Kiro format if present
            if msg.tool_results:
                kiro_tool_results = convert_tool_results_to_kiro_format(msg.tool_results)
                if kiro_tool_results:
                    user_input_context["toolResults"] = kiro_tool_results
            else:
                # Try to extract from content (already in Kiro format)
                tool_results = extract_tool_results_from_content(msg.content)
                if tool_results:
                    user_input_context["toolResults"] = tool_results
            
            # Add context if not empty (contains toolResults only, not images)
            if user_input_context:
                user_input["userInputMessageContext"] = user_input_context
            
            history.append({"userInputMessage": user_input})
            
        elif msg.role == "assistant":
            content = extract_text_content(msg.content)
            
            # Process tool_calls
            tool_uses = extract_tool_uses_from_message(msg.content, msg.tool_calls)
            
            # Fallback for empty content - Kiro API requires non-empty content.
            # Assistant turns with only toolUses use a single space (reference behaviour).
            if not content.strip():
                content = " " if tool_uses else EMPTY_CONTENT_FALLBACK
            
            assistant_response = {"content": content}
            
            if tool_uses:
                assistant_response["toolUses"] = tool_uses
            
            history.append({"assistantResponseMessage": assistant_response})
    
    return history


# ==================================================================================================
# Envelope Helpers (agentContinuationId, agentTaskType, additionalModelRequestFields)
# ==================================================================================================

def derive_agent_continuation_id(conversation_id: str) -> str:
    """
    Derive a stable agentContinuationId from the conversationId.

    Port of kiro2cc-proxy session.rs: SHA-256 over "agent-continuation:" + id,
    first 16 bytes formatted as a UUID string. The same conversationId always
    yields the same agentContinuationId.

    Args:
        conversation_id: Conversation ID sent in conversationState

    Returns:
        UUID-formatted lowercase hex string
    """
    digest = hashlib.sha256(b"agent-continuation:" + conversation_id.encode("utf-8")).hexdigest()
    return f"{digest[0:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"


def determine_agent_task_type(tools: Optional[List[UnifiedTool]]) -> str:
    """
    Determine conversationState.agentTaskType (reference convert.rs).

    Args:
        tools: Tools declared by the client

    Returns:
        "spectask" when any tool is declared, otherwise "vibe"
    """
    return "spectask" if tools else "vibe"


def is_gpt_model(model_id: str) -> bool:
    """
    Check whether the Kiro model ID belongs to the GPT family.

    Args:
        model_id: Kiro model ID

    Returns:
        True for gpt-* models
    """
    return model_id.lower().startswith("gpt-")


def additional_fields_skipped(model_id: str) -> bool:
    """
    Check whether additionalModelRequestFields must be omitted for a model.

    Kiro rejects the field for the "4.5" generation (sonnet/opus/haiku 4.5) with
    400 REQUEST_BODY_INVALID (reference thinking.rs). Legacy claude-3.x models
    are also skipped (the reference never sends to them), as are non-Claude,
    non-GPT models whose schema is unknown.

    Args:
        model_id: Kiro model ID

    Returns:
        True if the field must not be sent
    """
    m = model_id.lower()
    if is_gpt_model(m):
        return False
    if not m.startswith("claude-"):
        return True
    # Only send to generations known to accept it (>= 4.6). Kiro rejects it for
    # claude-3.x, 4.0 (verified: claude-sonnet-4) and 4.5 with REQUEST_BODY_INVALID.
    version = _claude_version(m)
    return version is None or version < (4, 6)


def _claude_version(model_id: str) -> Optional[tuple]:
    """
    Extracts (major, minor) from a Claude model ID.

    Handles "claude-sonnet-4.5", "claude-sonnet-4-5", "claude-opus-5", "claude-3.7-sonnet".

    Args:
        model_id: Lower-case Kiro model ID

    Returns:
        (major, minor) tuple, or None if no version found
    """
    match = re.search(r"(?<![\d])(\d+)(?:[.-](\d)(?![\d]))?", model_id)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or 0)


def model_max_output_tokens(model_id: str) -> int:
    """
    Kiro max_tokens upper bound per model generation (reference fields.rs).

    Args:
        model_id: Kiro model ID

    Returns:
        128000 for opus 4.7 / 4.8 / 5.x, otherwise 64000
    """
    m = model_id.lower()
    if any(tag in m for tag in ("opus-4-7", "opus-4.7", "opus-4-8", "opus-4.8", "opus-5", "opus.5", "opus 5")):
        return 128000
    return 64000


def _extract_effort(output_config: Any) -> Optional[str]:
    """
    Read `effort` from an Anthropic output_config (dict or object).

    Args:
        output_config: output_config value from the client request

    Returns:
        Non-empty effort string, or None
    """
    if output_config is None:
        return None
    effort = output_config.get("effort") if isinstance(output_config, dict) else getattr(output_config, "effort", None)
    if isinstance(effort, str) and effort.strip():
        return effort.strip()
    return None


def build_additional_model_request_fields(
    model_id: str,
    max_tokens: Optional[int] = None,
    output_config: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build top-level additionalModelRequestFields (port of reference fields.rs).

    - Disabled entirely by KIRO_ADDITIONAL_MODEL_FIELDS=false
    - Skipped for models in additional_fields_skipped() (4.5 generation etc.)
    - GPT family: {"reasoning": {"effort": <effort or "high">}}
    - Claude: {"output_config": {"effort": <effort or "low">},
               "max_tokens": clamp(max_tokens, 1024, model cap)} (max_tokens only if > 0)
    - The `thinking` field is intentionally never sent (reference: adds TTFB)

    Args:
        model_id: Kiro model ID
        max_tokens: Client max_tokens (None/0 = omit)
        output_config: Client output_config (dict with "effort"), optional

    Returns:
        Fields dict, or None when nothing should be sent
    """
    if not KIRO_ADDITIONAL_MODEL_FIELDS or additional_fields_skipped(model_id):
        return None

    effort = _extract_effort(output_config)

    if is_gpt_model(model_id):
        return {"reasoning": {"effort": effort or DEFAULT_GPT_EFFORT}}

    fields: Dict[str, Any] = {"output_config": {"effort": effort or DEFAULT_CLAUDE_EFFORT}}
    if isinstance(max_tokens, int) and not isinstance(max_tokens, bool) and max_tokens > 0:
        capped = min(max_tokens, model_max_output_tokens(model_id))
        fields["max_tokens"] = max(capped, MIN_ADDITIONAL_MAX_TOKENS)
    return fields


def build_placeholder_tool_specs(
    history: List[Dict[str, Any]],
    kiro_tools: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Create minimal tool specs for toolUse names in history that are not declared.

    Kiro requires every tool referenced in history to be defined in
    currentMessage tools. Names are compared case-insensitively (Kiro matches
    tool names ignoring case). Reference: websearch.rs create_placeholder_tool.

    Args:
        history: Kiro-format history (after pairing cleanup)
        kiro_tools: Declared tools in Kiro toolSpecification format

    Returns:
        Placeholder specs to append to the tools list (may be empty)
    """
    known = {
        str(t.get("toolSpecification", {}).get("name", "")).lower()
        for t in kiro_tools
    }
    placeholders: List[Dict[str, Any]] = []
    for entry in history:
        assistant = entry.get("assistantResponseMessage")
        if not assistant:
            continue
        for tool_use in assistant.get("toolUses") or []:
            name = tool_use.get("name") if isinstance(tool_use, dict) else None
            if not name or name.lower() in known:
                continue
            known.add(name.lower())
            placeholders.append({
                "toolSpecification": {
                    "name": name,
                    "description": PLACEHOLDER_TOOL_DESCRIPTION,
                    "inputSchema": {"json": {"type": "object", "properties": {}}},
                }
            })
    if placeholders:
        logger.debug(
            f"Added {len(placeholders)} placeholder tool definition(s) for history-only tools: "
            f"{[p['toolSpecification']['name'] for p in placeholders]}"
        )
    return placeholders


def drop_trailing_assistant_messages(messages: List[UnifiedMessage]) -> List[UnifiedMessage]:
    """
    Drop trailing assistant messages (prefill) - Kiro does not support prefill.

    Reference convert.rs: the conversation is truncated after the last non-assistant
    message. Unknown roles count as user (they are normalized later).

    Args:
        messages: Messages in unified format

    Returns:
        New list without trailing assistant messages (input is not mutated).
        An assistant-only list is returned unchanged.
    """
    end = len(messages)
    while end > 0 and messages[end - 1].role == "assistant":
        end -= 1
    if end == 0:
        # Assistant-only conversation: nothing to answer. Keep it unchanged and let
        # build_kiro_payload move the assistant turn into history (legacy behaviour).
        return list(messages)
    if end != len(messages):
        logger.info(f"Dropping {len(messages) - end} trailing assistant message(s) (prefill is not supported by Kiro)")
    return list(messages[:end])


# ==================================================================================================
# Main Payload Building
# ==================================================================================================

def build_kiro_payload(
    messages: List[UnifiedMessage],
    system_prompt: str,
    model_id: str,
    tools: Optional[List[UnifiedTool]],
    conversation_id: str,
    profile_arn: str,
    thinking_config: ThinkingConfig,
    max_tokens: Optional[int] = None,
    output_config: Optional[Any] = None,
) -> KiroPayloadResult:
    """
    Builds complete payload for Kiro API from unified data.
    
    This is the main function that assembles the Kiro API payload from
    API-agnostic unified message and tool formats.

    Layout:
    - history[0..1]: system prompt as user {system} + assistant
      "I will follow these instructions." (only when a system prompt exists)
    - history[2..]: conversation turns (trailing assistant prefill dropped)
    - currentMessage: last user turn with tools (+ placeholders) and toolResults
    - conversationState.agentTaskType / agentContinuationId envelope fields
    - top-level additionalModelRequestFields (gated by KIRO_ADDITIONAL_MODEL_FIELDS)
    
    Args:
        messages: List of messages in unified format (without system messages)
        system_prompt: Already extracted system prompt
        model_id: Internal Kiro model ID
        tools: List of tools in unified format (or None)
        conversation_id: Unique conversation ID
        profile_arn: AWS CodeWhisperer profile ARN
        thinking_config: Thinking configuration from API adapter
        max_tokens: Client max output tokens (for additionalModelRequestFields), optional
        output_config: Client output_config ({"effort": ...}), optional
    
    Returns:
        KiroPayloadResult with payload and tool documentation
    
    Raises:
        ValueError: If there are no messages to send
    """
    # Process tools with long descriptions
    processed_tools, tool_documentation = process_tools_with_long_descriptions(tools)
    
    # Validate tool names against Kiro API 64-character limit
    validate_tool_names(processed_tools)
    
    # Add tool documentation to system prompt if present
    full_system_prompt = system_prompt
    if tool_documentation:
        full_system_prompt = full_system_prompt + tool_documentation if full_system_prompt else tool_documentation.strip()
    
    # Add thinking mode legitimization only when tags are actually injected
    thinking_system_addition = (
        get_thinking_system_prompt_addition()
        if should_inject_thinking(thinking_config, model_id) else ""
    )
    if thinking_system_addition:
        full_system_prompt = full_system_prompt + thinking_system_addition if full_system_prompt else thinking_system_addition.strip()
    
    # Add truncation recovery legitimization to system prompt if enabled
    truncation_system_addition = get_truncation_recovery_system_addition()
    if truncation_system_addition:
        full_system_prompt = full_system_prompt + truncation_system_addition if full_system_prompt else truncation_system_addition.strip()

    # Kiro does not support assistant prefill - drop trailing assistant messages
    messages = drop_trailing_assistant_messages(messages)
    
    # If no tools are defined, strip ALL tool-related content from messages
    # Kiro API rejects requests with toolResults but no tools
    if not tools:
        messages_with_assistants, _ = strip_all_tool_content(messages)
    else:
        # Tool results without a preceding assistant tool call are converted to text
        messages_with_assistants, _ = ensure_assistant_before_tool_results(messages)
    
    # Merge adjacent messages with the same role
    merged_messages = merge_adjacent_messages(messages_with_assistants)
    
    # Ensure first message is from user (Kiro API requirement, fixes issue #60)
    merged_messages = ensure_first_message_is_user(merged_messages)
    
    # Normalize unknown roles to 'user' (fixes issue #64)
    # This must happen BEFORE ensure_alternating_roles() so that consecutive
    # messages with unknown roles (e.g., 'developer') are properly detected
    merged_messages = normalize_message_roles(merged_messages)
    
    # Ensure alternating user/assistant roles (fixes issue #64)
    # Insert synthetic assistant messages between consecutive user messages
    merged_messages = ensure_alternating_roles(merged_messages)
    
    if not merged_messages:
        raise ValueError("No messages to send")
    
    # History = all messages except the last one; current = last (always user after prefill drop)
    history_messages = merged_messages[:-1]
    current_message = merged_messages[-1]
    if current_message.role == "assistant":
        # Only reachable for assistant-only conversations (see drop_trailing_assistant_messages)
        history_messages = merged_messages
        current_message = UnifiedMessage(role="user", content="")
    history = build_kiro_history(history_messages, model_id)
    current_content = extract_text_content(current_message.content)
    
    # Tool results of the current message (Kiro format)
    if current_message.tool_results:
        current_tool_results = convert_tool_results_to_kiro_format(current_message.tool_results)
    else:
        current_tool_results = extract_tool_results_from_content(current_message.content)

    # Enforce toolUse/toolResult pairing: drop duplicate results, convert orphaned
    # results to text, remove toolUses that never get a result
    current_tool_results, current_orphan_text = repair_tool_pairing(history, current_tool_results)
    if current_orphan_text:
        current_content = f"{current_content}\n\n{current_orphan_text}" if current_content else current_orphan_text

    # Empty content fallbacks
    if not current_content:
        current_content = TOOL_RESULT_ONLY_CONTENT if current_tool_results else EMPTY_CONTENT_FALLBACK

    # System prompt as a separate leading user/assistant pair (stable prefix)
    system_prefix_entries = 0
    if full_system_prompt:
        history = [
            {"userInputMessage": {"content": full_system_prompt, "modelId": model_id, "origin": "AI_EDITOR"}},
            {"assistantResponseMessage": {"content": SYSTEM_PROMPT_ACKNOWLEDGEMENT}},
        ] + history
        system_prefix_entries = 2
    
    # Process images in current message - extract from message or content
    # IMPORTANT: images go directly into userInputMessage, NOT into userInputMessageContext
    # This matches the native Kiro IDE format
    images = current_message.images or extract_images_from_content(current_message.content)
    kiro_images = None
    if images:
        kiro_images = convert_images_to_kiro_format(images)
        if kiro_images:
            logger.debug(f"Added {len(kiro_images)} image(s) to current message")
    
    # Build user_input_context for tools and toolResults only (NOT images)
    user_input_context: Dict[str, Any] = {}
    
    # Add tools if present, plus placeholder specs for tools referenced only in history
    kiro_tools = convert_tools_to_kiro_format(processed_tools)
    if kiro_tools:
        kiro_tools.extend(build_placeholder_tool_specs(history, kiro_tools))
        user_input_context["tools"] = kiro_tools
    
    if current_tool_results:
        user_input_context["toolResults"] = current_tool_results
    
    # Inject thinking tags if enabled (current user message only)
    current_content = inject_thinking_tags(current_content, thinking_config, model_id)
    
    # Build userInputMessage
    user_input_message = {
        "content": current_content,
        "modelId": model_id,
        "origin": "AI_EDITOR",
    }
    
    # Add images directly to userInputMessage (NOT to userInputMessageContext)
    if kiro_images:
        user_input_message["images"] = kiro_images
    
    # Add user_input_context if present (contains tools and toolResults only)
    if user_input_context:
        user_input_message["userInputMessageContext"] = user_input_context
    
    # Assemble final payload
    payload: Dict[str, Any] = {
        "conversationState": {
            "agentContinuationId": derive_agent_continuation_id(conversation_id),
            "agentTaskType": determine_agent_task_type(tools),
            "chatTriggerType": "MANUAL",
            "conversationId": conversation_id,
            "currentMessage": {
                "userInputMessage": user_input_message
            }
        }
    }
    
    # Add history only if not empty
    if history:
        payload["conversationState"]["history"] = history
    
    # Add profileArn
    if profile_arn:
        payload["profileArn"] = profile_arn

    # Model-specific request fields (top level, next to conversationState)
    additional_fields = build_additional_model_request_fields(model_id, max_tokens, output_config)
    if additional_fields:
        payload["additionalModelRequestFields"] = additional_fields

    # Payload size guard — auto-trim if enabled (system prompt pair is never trimmed)
    if AUTO_TRIM_PAYLOAD:
        payload_size = check_payload_size(payload)
        if payload_size > KIRO_MAX_PAYLOAD_BYTES:
            stats = trim_payload_to_limit(
                payload, KIRO_MAX_PAYLOAD_BYTES, keep_prefix_entries=system_prefix_entries
            )
            logger.info(
                f"Trimmed conversation history: {stats.original_entries} -> {stats.final_entries} messages "
                f"({stats.original_bytes} -> {stats.final_bytes} bytes)"
            )

    return KiroPayloadResult(payload=payload, tool_documentation=tool_documentation)
