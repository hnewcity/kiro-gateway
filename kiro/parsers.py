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
Parsers for AWS Event Stream format.

Contains classes and functions for:
- Parsing binary AWS SSE stream
- Extracting JSON events
- Processing tool calls
- Content deduplication
"""

import codecs
import json
import re
import struct
import zlib
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger

from kiro.utils import generate_tool_call_id


def find_matching_brace(text: str, start_pos: int) -> int:
    """
    Finds the position of the closing brace considering nesting and strings.
    
    Uses bracket counting for correct parsing of nested JSON.
    Accounts for quoted strings and escape sequences.
    
    Args:
        text: Text to search
        start_pos: Position of opening brace '{'
    
    Returns:
        Position of closing brace or -1 if not found
    
    Example:
        >>> find_matching_brace('{"a": {"b": 1}}', 0)
        14
        >>> find_matching_brace('{"a": "{}"}', 0)
        10
    """
    if start_pos >= len(text) or text[start_pos] != '{':
        return -1
    
    brace_count = 0
    in_string = False
    escape_next = False
    
    for i in range(start_pos, len(text)):
        char = text[i]
        
        if escape_next:
            escape_next = False
            continue
        
        if char == '\\' and in_string:
            escape_next = True
            continue
        
        if char == '"' and not escape_next:
            in_string = not in_string
            continue
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return i
    
    return -1


def parse_bracket_tool_calls(response_text: str) -> List[Dict[str, Any]]:
    """
    Parses tool calls in [Called func_name with args: {...}] format.
    
    Some models return tool calls in text format instead of
    structured JSON. This function extracts them.
    
    Args:
        response_text: Model response text
    
    Returns:
        List of tool calls in OpenAI format
    
    Example:
        >>> text = "[Called get_weather with args: {\"city\": \"London\"}]"
        >>> calls = parse_bracket_tool_calls(text)
        >>> calls[0]["function"]["name"]
        'get_weather'
    """
    if not response_text or "[Called" not in response_text:
        return []
    
    tool_calls = []
    pattern = r'\[Called\s+(\w+)\s+with\s+args:\s*'
    
    for match in re.finditer(pattern, response_text, re.IGNORECASE):
        func_name = match.group(1)
        args_start = match.end()
        
        # Find JSON start
        json_start = response_text.find('{', args_start)
        if json_start == -1:
            continue
        
        # Find JSON end considering nesting
        json_end = find_matching_brace(response_text, json_start)
        if json_end == -1:
            continue
        
        json_str = response_text[json_start:json_end + 1]
        
        try:
            args = json.loads(json_str)
            tool_call_id = generate_tool_call_id()
            # index will be added later when forming the final response
            tool_calls.append({
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": func_name,
                    "arguments": json.dumps(args)
                }
            })
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse tool call arguments: {json_str[:100]}")
    
    return tool_calls


def deduplicate_tool_calls(tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Removes duplicate tool calls.
    
    Deduplication occurs by two criteria:
    1. By id - if there are multiple tool calls with the same id, keep the one with
       more arguments (not empty "{}")
    2. By name+arguments - remove complete duplicates
    
    Args:
        tool_calls: List of tool calls
    
    Returns:
        List of unique tool calls
    """
    # First deduplicate by id - keep tool call with non-empty arguments
    by_id: Dict[str, Dict[str, Any]] = {}
    for tc in tool_calls:
        tc_id = tc.get("id", "")
        if not tc_id:
            # Without id - add as is (will be deduplicated by name+args)
            continue
        
        existing = by_id.get(tc_id)
        if existing is None:
            by_id[tc_id] = tc
        else:
            # Duplicate by id exists - keep the one with more arguments
            existing_args = existing.get("function", {}).get("arguments", "{}")
            current_args = tc.get("function", {}).get("arguments", "{}")
            
            # Prefer non-empty arguments
            if current_args != "{}" and (existing_args == "{}" or len(current_args) > len(existing_args)):
                logger.debug(f"Replacing tool call {tc_id} with better arguments: {len(existing_args)} -> {len(current_args)}")
                by_id[tc_id] = tc
    
    # Collect tool calls: first those with id, then without id
    result_with_id = list(by_id.values())
    result_without_id = [tc for tc in tool_calls if not tc.get("id")]
    
    # Now deduplicate by name+arguments for all
    seen = set()
    unique = []
    
    for tc in result_with_id + result_without_id:
        # Protection against None in function
        func = tc.get("function") or {}
        func_name = func.get("name") or ""
        func_args = func.get("arguments") or "{}"
        key = f"{func_name}-{func_args}"
        if key not in seen:
            seen.add(key)
            unique.append(tc)
    
    if len(tool_calls) != len(unique):
        logger.debug(f"Deduplicated tool calls: {len(tool_calls)} -> {len(unique)}")
    
    return unique


# ==================================================================================================
# AWS binary eventstream (application/vnd.amazon.eventstream) frame decoding
# ==================================================================================================
#
# Frame layout (all integers big-endian):
#   [total_len u32][headers_len u32][prelude_crc u32][headers ...][payload ...][message_crc u32]
# prelude_crc = CRC32 of the first 8 bytes, message_crc = CRC32 of everything before it.

EVENTSTREAM_PRELUDE_SIZE = 12
EVENTSTREAM_MESSAGE_CRC_SIZE = 4
EVENTSTREAM_MIN_FRAME_SIZE = EVENTSTREAM_PRELUDE_SIZE + EVENTSTREAM_MESSAGE_CRC_SIZE
EVENTSTREAM_MAX_FRAME_SIZE = 16 * 1024 * 1024
# Consecutive decode errors (corrupt frames / resync episodes) before the decoder gives up
EVENTSTREAM_MAX_CONSECUTIVE_ERRORS = 5
# Max bytes skipped in a single resync episode before the decoder gives up
EVENTSTREAM_MAX_RESYNC_BYTES = 1024 * 1024

# Header value types per the AWS eventstream spec
_HEADER_TYPE_BOOL_TRUE = 0
_HEADER_TYPE_BOOL_FALSE = 1
_HEADER_TYPE_BYTE = 2
_HEADER_TYPE_SHORT = 3
_HEADER_TYPE_INTEGER = 4
_HEADER_TYPE_LONG = 5
_HEADER_TYPE_BYTE_ARRAY = 6
_HEADER_TYPE_STRING = 7
_HEADER_TYPE_TIMESTAMP = 8
_HEADER_TYPE_UUID = 9

# Fixed-size header value types: type -> (size, struct format or None for raw bytes)
_FIXED_HEADER_TYPES: Dict[int, Tuple[int, Optional[str]]] = {
    _HEADER_TYPE_BYTE: (1, ">b"),
    _HEADER_TYPE_SHORT: (2, ">h"),
    _HEADER_TYPE_INTEGER: (4, ">i"),
    _HEADER_TYPE_LONG: (8, ">q"),
    _HEADER_TYPE_TIMESTAMP: (8, ">q"),
    _HEADER_TYPE_UUID: (16, None),
}

HeaderValue = Union[bool, int, bytes, str]


class EventStreamDecodeError(ValueError):
    """Raised when an eventstream frame (or its headers) cannot be decoded."""


def parse_eventstream_headers(data: bytes) -> Dict[str, HeaderValue]:
    """
    Parses the headers section of an AWS eventstream frame.

    Each header is: name_len (u8), name (utf-8), value_type (u8), value.
    String (7) and byte array (6) values are prefixed with a u16 length.

    Args:
        data: Raw header bytes (exactly headers_len bytes of the frame)

    Returns:
        Mapping of header name to decoded value (str for string headers)

    Raises:
        EventStreamDecodeError: If the headers are malformed or truncated
    """
    headers: Dict[str, HeaderValue] = {}
    offset = 0
    end = len(data)

    def _require(count: int) -> None:
        if offset + count > end:
            raise EventStreamDecodeError(
                f"header truncated: need {count} bytes at offset {offset}, have {end - offset}"
            )

    while offset < end:
        _require(1)
        name_len = data[offset]
        offset += 1
        if name_len == 0:
            raise EventStreamDecodeError("header name length must not be 0")
        _require(name_len)
        name = data[offset:offset + name_len].decode("utf-8", errors="replace")
        offset += name_len

        _require(1)
        value_type = data[offset]
        offset += 1

        value: HeaderValue
        if value_type == _HEADER_TYPE_BOOL_TRUE:
            value = True
        elif value_type == _HEADER_TYPE_BOOL_FALSE:
            value = False
        elif value_type in (_HEADER_TYPE_STRING, _HEADER_TYPE_BYTE_ARRAY):
            _require(2)
            (value_len,) = struct.unpack_from(">H", data, offset)
            offset += 2
            _require(value_len)
            raw_value = bytes(data[offset:offset + value_len])
            offset += value_len
            value = raw_value.decode("utf-8", errors="replace") if value_type == _HEADER_TYPE_STRING else raw_value
        elif value_type in _FIXED_HEADER_TYPES:
            size, fmt = _FIXED_HEADER_TYPES[value_type]
            _require(size)
            if fmt is None:
                value = bytes(data[offset:offset + size])
            else:
                (value,) = struct.unpack_from(fmt, data, offset)
            offset += size
        else:
            raise EventStreamDecodeError(f"invalid header value type {value_type} for header '{name}'")

        headers[name] = value

    return headers


def encode_eventstream_frame(headers: Dict[str, str], payload: bytes) -> bytes:
    """
    Encodes a single AWS eventstream frame with string headers.

    Used by tests and debugging tools to build realistic Kiro stream data.

    Args:
        headers: Header name -> string value
        payload: Frame payload bytes

    Returns:
        Complete frame bytes including prelude and both CRCs
    """
    header_bytes = bytearray()
    for name, value in headers.items():
        name_raw = name.encode("utf-8")
        value_raw = value.encode("utf-8")
        header_bytes += struct.pack(">B", len(name_raw)) + name_raw
        header_bytes += struct.pack(">BH", _HEADER_TYPE_STRING, len(value_raw)) + value_raw

    total_len = EVENTSTREAM_MIN_FRAME_SIZE + len(header_bytes) + len(payload)
    prelude = struct.pack(">II", total_len, len(header_bytes))
    prelude += struct.pack(">I", zlib.crc32(prelude))
    message = prelude + bytes(header_bytes) + payload
    return message + struct.pack(">I", zlib.crc32(message))


class AwsEventStreamParser:
    """
    Parser for AWS Event Stream format.
    
    Kiro returns application/vnd.amazon.eventstream binary frames. The parser decodes
    frames (validating prelude and message CRC32), routes them by the ``:message-type``
    and ``:event-type`` / ``:exception-type`` headers, and converts payloads into a
    convenient format. Data is buffered as bytes, so multi-byte UTF-8 characters split
    across network chunks are never lost.
    
    If the stream is clearly not binary eventstream (e.g. plain JSON text), the parser
    falls back to scanning the text for known JSON event patterns.
    
    Emitted event types:
    - content: Text content of response (data: str)
    - usage: Full metering dict, e.g. {"usage": 0.12, "unit": "credit", ...} (data: dict)
    - context_usage: Context usage percentage (data: float)
    - exception: Upstream exception/error frame
      (data: {"exception_type": str, "message": str, "raw": dict})
    
    Tool calls (tool_start / tool_input / tool_stop) are accumulated internally and
    returned by get_tool_calls().
    
    Attributes:
        buffer: Text buffer used by the plain-text fallback path
        last_content: Last processed content (for deduplication)
        current_tool_call: Current incomplete tool call
        tool_calls: List of completed tool calls
    
    Example:
        >>> parser = AwsEventStreamParser()
        >>> events = parser.feed(chunk)
        >>> for event in events:
        ...     if event["type"] == "content":
        ...         print(event["data"])
    """
    
    # Patterns for finding JSON events
    EVENT_PATTERNS = [
        ('{"content":', 'content'),
        ('{"name":', 'tool_start'),
        ('{"input":', 'tool_input'),
        ('{"stop":', 'tool_stop'),
        ('{"followupPrompt":', 'followup'),
        ('{"usage":', 'usage'),
        ('{"contextUsagePercentage":', 'context_usage'),
    ]
    
    # Maps the :event-type header to the internal event type used by _process_event
    EVENT_TYPE_MAP: Dict[str, str] = {
        "assistantResponseEvent": "content",
        "meteringEvent": "usage",
        "contextUsageEvent": "context_usage",
        "followupPromptEvent": "followup",
    }
    
    # Stream modes
    _MODE_UNKNOWN = "unknown"
    _MODE_BINARY = "binary"
    _MODE_TEXT = "text"
    
    def __init__(self) -> None:
        """Initializes the parser."""
        self.buffer = ""  # Text buffer (plain-text fallback path only)
        self.last_content: Optional[str] = None  # For deduplicating repeating content
        self.current_tool_call: Optional[Dict[str, Any]] = None
        self.tool_calls: List[Dict[str, Any]] = []
        self._init_decoder_state()
    
    def _init_decoder_state(self) -> None:
        """Initializes (or resets) binary decoder / mode detection state."""
        self._mode = self._MODE_UNKNOWN
        self._byte_buffer = bytearray()
        self._text_decoder = codecs.getincrementaldecoder("utf-8")(errors="ignore")
        self._consecutive_errors = 0
        self._resyncing = False
        self._resync_skipped = 0
        self._decoder_stopped = False
        self._stop_reported = False
        self._stop_reason = ""
        self.frames_decoded = 0
        self.bytes_skipped = 0
    
    def feed(self, chunk: bytes) -> List[Dict[str, Any]]:
        """
        Adds chunk to buffer and returns parsed events.
        
        The first bytes of the stream decide the mode: a valid eventstream prelude
        selects binary frame decoding, anything else selects the plain-text fallback.
        
        Args:
            chunk: Bytes of data from stream
        
        Returns:
            List of events in {"type": str, "data": Any} format
        """
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            logger.warning(f"AwsEventStreamParser.feed expected bytes, got {type(chunk).__name__}")
            return []
        if not chunk:
            return []
        
        if self._mode == self._MODE_BINARY:
            if self._decoder_stopped:
                return []
            self._byte_buffer.extend(chunk)
            return self._decode_frames()
        
        if self._mode == self._MODE_TEXT:
            return self._feed_text(bytes(chunk))
        
        # Mode not decided yet: accumulate until we can tell
        self._byte_buffer.extend(chunk)
        mode = self._detect_mode()
        if mode is None:
            return []
        self._mode = mode
        if mode == self._MODE_BINARY:
            logger.debug("AWS eventstream: binary frame mode")
            return self._decode_frames()
        
        logger.debug("AWS eventstream: stream is not binary eventstream, using text fallback")
        pending = bytes(self._byte_buffer)
        self._byte_buffer.clear()
        return self._feed_text(pending)
    
    def _detect_mode(self) -> Optional[str]:
        """
        Decides whether the stream is binary eventstream or plain text.
        
        A frame is at most 16MB, so its big-endian total length always starts with
        0x00 (or 0x01 for exactly 16MB). Text never starts with those bytes, so the
        first byte is enough. A corrupt first frame is then handled by binary resync
        instead of silently switching to text scanning.
        
        Returns:
            Binary or text mode, or None if no data is buffered yet
        """
        if not self._byte_buffer:
            return None
        return self._MODE_BINARY if self._byte_buffer[0] <= 0x01 else self._MODE_TEXT
    
    # ------------------------------------------------------------------------------------------
    # Binary eventstream path
    # ------------------------------------------------------------------------------------------
    
    def _decode_frames(self) -> List[Dict[str, Any]]:
        """
        Decodes all complete frames currently in the byte buffer.
        
        Recovery follows the reference implementation: a bad prelude (CRC mismatch or
        impossible lengths) means frame boundaries are lost, so bytes are skipped until
        the next plausible prelude; a bad message CRC or malformed headers means the
        boundary is right but the data is corrupt, so the whole frame is skipped.
        
        Returns:
            Parsed events from all decoded frames
        """
        events: List[Dict[str, Any]] = []
        buf = self._byte_buffer
        
        while not self._decoder_stopped and len(buf) >= EVENTSTREAM_PRELUDE_SIZE:
            total_len, headers_len, prelude_crc = struct.unpack_from(">III", buf, 0)
            
            # The prelude CRC covers both lengths, so validate it before trusting them
            actual_prelude_crc = zlib.crc32(buf[:8])
            if actual_prelude_crc != prelude_crc:
                self._resync(f"prelude CRC mismatch (expected 0x{prelude_crc:08x}, got 0x{actual_prelude_crc:08x})")
                continue
            if total_len < EVENTSTREAM_MIN_FRAME_SIZE or total_len > EVENTSTREAM_MAX_FRAME_SIZE:
                self._resync(f"invalid frame length {total_len} (allowed {EVENTSTREAM_MIN_FRAME_SIZE}..{EVENTSTREAM_MAX_FRAME_SIZE})")
                continue
            if headers_len > total_len - EVENTSTREAM_MIN_FRAME_SIZE:
                self._resync(f"headers length {headers_len} exceeds frame length {total_len}")
                continue
            
            if len(buf) < total_len:
                break  # Wait for the rest of the frame
            
            frame = bytes(buf[:total_len])
            del buf[:total_len]
            
            (message_crc,) = struct.unpack_from(">I", frame, total_len - EVENTSTREAM_MESSAGE_CRC_SIZE)
            actual_message_crc = zlib.crc32(frame[:-EVENTSTREAM_MESSAGE_CRC_SIZE])
            if actual_message_crc != message_crc:
                self.bytes_skipped += total_len
                self._register_error(
                    f"message CRC mismatch (expected 0x{message_crc:08x}, got 0x{actual_message_crc:08x}), "
                    f"skipped corrupt frame of {total_len} bytes"
                )
                continue
            
            headers_end = EVENTSTREAM_PRELUDE_SIZE + headers_len
            try:
                headers = parse_eventstream_headers(frame[EVENTSTREAM_PRELUDE_SIZE:headers_end])
            except EventStreamDecodeError as e:
                self.bytes_skipped += total_len
                self._register_error(f"header parse failed ({e}), skipped frame of {total_len} bytes")
                continue
            
            payload = frame[headers_end:-EVENTSTREAM_MESSAGE_CRC_SIZE]
            self._consecutive_errors = 0
            self._resyncing = False
            self._resync_skipped = 0
            self.frames_decoded += 1
            
            event = self._handle_frame(headers, payload)
            if event:
                events.append(event)
        
        if self._decoder_stopped and not self._stop_reported:
            self._stop_reported = True
            events.append({
                "type": "exception",
                "data": {
                    "exception_type": "EventStreamDecodeError",
                    "message": self._stop_reason,
                    "raw": {},
                },
            })
        
        return events
    
    def _resync(self, reason: str) -> None:
        """
        Skips bytes after a prelude error until the next plausible frame start.
        
        A whole resync episode counts as one error; it ends at the next valid frame.
        
        Args:
            reason: Why the current position is not a valid prelude
        """
        buf = self._byte_buffer
        if not self._resyncing:
            self._resyncing = True
            self._register_error(f"{reason}; resyncing to next frame boundary")
            if self._decoder_stopped:
                return
        
        # A valid prelude can only start with 0x00 (or 0x01 for exactly 16MB frames)
        candidates = [pos for pos in (buf.find(b"\x00", 1), buf.find(b"\x01", 1)) if pos != -1]
        skip = min(candidates) if candidates else len(buf)
        del buf[:skip]
        self._resync_skipped += skip
        self.bytes_skipped += skip
        logger.debug(f"AWS eventstream resync: skipped {skip} bytes ({self._resync_skipped} in this episode)")
        
        if self._resync_skipped > EVENTSTREAM_MAX_RESYNC_BYTES:
            self._stop_decoder(
                f"no valid frame found after skipping {self._resync_skipped} bytes (last error: {reason})"
            )
    
    def _register_error(self, message: str) -> None:
        """
        Records a decode error and stops the decoder after too many in a row.
        
        Args:
            message: Error description
        """
        self._consecutive_errors += 1
        logger.warning(
            f"AWS eventstream decode error {self._consecutive_errors}/{EVENTSTREAM_MAX_CONSECUTIVE_ERRORS}: {message}"
        )
        if self._consecutive_errors >= EVENTSTREAM_MAX_CONSECUTIVE_ERRORS:
            self._stop_decoder(f"{self._consecutive_errors} consecutive errors, last: {message}")
    
    def _stop_decoder(self, reason: str) -> None:
        """
        Stops the decoder; remaining and future data is discarded.
        
        Args:
            reason: Why decoding was abandoned
        """
        self._decoder_stopped = True
        self._stop_reason = f"AWS eventstream decoder stopped: {reason}"
        self._byte_buffer.clear()
        logger.error(self._stop_reason)
    
    def _handle_frame(self, headers: Dict[str, HeaderValue], payload: bytes) -> Optional[Dict[str, Any]]:
        """
        Routes a decoded frame by its :message-type header.
        
        Args:
            headers: Decoded frame headers
            payload: Frame payload bytes
        
        Returns:
            Parsed event or None
        """
        message_type = self._header_str(headers, ":message-type") or "event"
        
        if message_type == "event":
            return self._handle_event_frame(self._header_str(headers, ":event-type"), payload)
        if message_type == "exception":
            exception_type = self._header_str(headers, ":exception-type") or "UnknownException"
            return self._build_exception_event(exception_type, payload)
        if message_type == "error":
            error_code = self._header_str(headers, ":error-code") or "UnknownError"
            return self._build_exception_event(error_code, payload)
        
        logger.warning(f"AWS eventstream: ignoring frame with unknown :message-type '{message_type}'")
        return None
    
    @staticmethod
    def _header_str(headers: Dict[str, HeaderValue], name: str) -> Optional[str]:
        """
        Returns a header value if it is a string header.
        
        Args:
            headers: Decoded frame headers
            name: Header name
        
        Returns:
            String value or None
        """
        value = headers.get(name)
        return value if isinstance(value, str) else None
    
    def _handle_event_frame(self, event_type: Optional[str], payload: bytes) -> Optional[Dict[str, Any]]:
        """
        Decodes an event frame payload and dispatches it to the event processors.
        
        The :event-type header decides the event type; unknown or missing types fall back
        to the same JSON key detection used by the text path.
        
        Args:
            event_type: Value of the :event-type header, if any
            payload: Frame payload bytes (JSON)
        
        Returns:
            Parsed event or None
        """
        if not payload:
            return None
        try:
            data = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            logger.warning(f"Failed to decode '{event_type}' payload: {e}. Raw: {payload[:100]!r}")
            return None
        
        if event_type == "toolUseEvent":
            if not isinstance(data, dict):
                logger.warning(f"Unexpected toolUseEvent payload type: {type(data).__name__}")
                return None
            return self._process_tool_use_event(data)
        
        internal_type = self.EVENT_TYPE_MAP.get(event_type) if event_type else None
        
        if internal_type == "usage":
            # Metering payloads are normalized in _process_event (numbers are wrapped)
            return self._process_event(data, internal_type)
        
        if not isinstance(data, dict):
            logger.warning(f"Unexpected '{event_type}' payload type: {type(data).__name__}")
            return None
        
        if internal_type == "content" and "content" not in data:
            # e.g. an assistantResponseEvent carrying only metadata such as modelId
            logger.debug(f"Ignoring assistantResponseEvent without content, keys {list(data.keys())[:5]}")
            return None
        
        if internal_type is None:
            internal_type = self._detect_event_type_from_keys(data)
            if internal_type is None:
                logger.debug(f"Ignoring AWS event '{event_type}' with keys {list(data.keys())[:5]}")
                return None
        
        return self._process_event(data, internal_type)
    
    def _detect_event_type_from_keys(self, data: Dict[str, Any]) -> Optional[str]:
        """
        Detects the event type from JSON keys (fallback when :event-type is unknown).
        
        Mirrors EVENT_PATTERNS: the first key of the object decides the type.
        
        Args:
            data: Decoded payload
        
        Returns:
            Internal event type or None
        """
        key_map = {pattern[2:-2]: event_type for pattern, event_type in self.EVENT_PATTERNS}
        for key in data:
            if key in key_map:
                return key_map[key]
        return None
    
    def _process_tool_use_event(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Processes a binary toolUseEvent payload.
        
        Kiro repeats name/toolUseId on every fragment, so fragments are grouped by
        toolUseId instead of by the first JSON key.
        
        Args:
            data: Decoded toolUseEvent payload
        
        Returns:
            Always None (tool calls are collected via get_tool_calls())
        """
        tool_use_id = data.get("toolUseId")
        current_id = self.current_tool_call.get("id") if self.current_tool_call else None
        is_new_call = "name" in data and (
            self.current_tool_call is None or (tool_use_id is not None and tool_use_id != current_id)
        )
        
        if is_new_call:
            return self._process_tool_start_event(data)
        
        if "input" in data:
            self._process_tool_input_event(data)
        if data.get("stop"):
            self._process_tool_stop_event(data)
        return None
    
    def _build_exception_event(self, exception_type: str, payload: bytes) -> Dict[str, Any]:
        """
        Builds an exception event from an exception/error frame.
        
        Args:
            exception_type: Value of :exception-type (or :error-code)
            payload: Frame payload (usually JSON like {"message": "..."})
        
        Returns:
            {"type": "exception", "data": {"exception_type", "message", "raw"}}
        """
        text = payload.decode("utf-8", errors="replace")
        raw: Dict[str, Any] = {}
        parsed: Any = None
        if text.strip():
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = None
        if isinstance(parsed, dict):
            raw = parsed
        elif text:
            raw = {"payload": text}
        
        message = raw.get("message") or raw.get("Message")
        if not isinstance(message, str):
            message = text
        
        logger.warning(f"Kiro stream exception: {exception_type} - {message[:500]}")
        return {
            "type": "exception",
            "data": {"exception_type": exception_type, "message": message, "raw": raw},
        }
    
    # ------------------------------------------------------------------------------------------
    # Plain-text fallback path
    # ------------------------------------------------------------------------------------------
    
    def _feed_text(self, data: bytes) -> List[Dict[str, Any]]:
        """
        Scans decoded text for known JSON event patterns (non-binary fallback).
        
        Uses an incremental UTF-8 decoder so characters split across chunks are kept.
        
        Args:
            data: Raw bytes
        
        Returns:
            Parsed events
        """
        self.buffer += self._text_decoder.decode(data)
        
        events = []
        
        while True:
            # Find nearest pattern
            earliest_pos = -1
            earliest_type = None
            
            for pattern, event_type in self.EVENT_PATTERNS:
                pos = self.buffer.find(pattern)
                if pos != -1 and (earliest_pos == -1 or pos < earliest_pos):
                    earliest_pos = pos
                    earliest_type = event_type
            
            if earliest_pos == -1:
                break
            
            # Find JSON end
            json_end = find_matching_brace(self.buffer, earliest_pos)
            if json_end == -1:
                # JSON not complete, wait for more data
                break
            
            json_str = self.buffer[earliest_pos:json_end + 1]
            self.buffer = self.buffer[json_end + 1:]
            
            try:
                data_obj = json.loads(json_str)
                event = self._process_event(data_obj, earliest_type)
                if event:
                    events.append(event)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON: {json_str[:100]}")
        
        return events
    
    def _process_event(self, data: Any, event_type: str) -> Optional[Dict[str, Any]]:
        """
        Processes a parsed event.
        
        Args:
            data: Parsed JSON (a dict, except metering payloads which may be a bare number)
            event_type: Event type
        
        Returns:
            Processed event or None
        """
        if event_type == 'content':
            return self._process_content_event(data)
        elif event_type == 'tool_start':
            return self._process_tool_start_event(data)
        elif event_type == 'tool_input':
            return self._process_tool_input_event(data)
        elif event_type == 'tool_stop':
            return self._process_tool_stop_event(data)
        elif event_type == 'usage':
            logger.debug(f"[raw-event] metering payload: {data!r}")
            return {"type": "usage", "data": self._normalize_metering(data)}
        elif event_type == 'context_usage':
            return {"type": "context_usage", "data": data.get('contextUsagePercentage', 0)}
        
        return None
    
    @staticmethod
    def _normalize_metering(data: Any) -> Dict[str, Any]:
        """
        Normalizes a metering payload into a dict, keeping every field.
        
        Args:
            data: Metering payload, e.g. {"usage": 0.12, "unit": "credit",
                "cacheReadInputTokens": 100} or a bare number
        
        Returns:
            Metering dict; a bare number becomes {"usage": number}
        """
        if isinstance(data, dict):
            return dict(data)
        if isinstance(data, (int, float)) and not isinstance(data, bool):
            return {"usage": data}
        logger.warning(f"Unexpected metering payload type: {type(data).__name__}")
        return {}
    
    def _process_content_event(self, data: dict) -> Optional[Dict[str, Any]]:
        """Processes content event."""
        content = data.get('content', '')
        
        # Skip followupPrompt
        if data.get('followupPrompt'):
            return None
        
        # Deduplicate repeating content
        if content == self.last_content:
            return None
        
        self.last_content = content
        
        return {"type": "content", "data": content}
    
    def _process_tool_start_event(self, data: dict) -> Optional[Dict[str, Any]]:
        """Processes tool call start."""
        # Finalize previous tool call if exists
        if self.current_tool_call:
            self._finalize_tool_call()
        
        # input can be string or object
        input_data = data.get('input', '')
        if isinstance(input_data, dict):
            if input_data:
                # Non-empty dict: serialize it
                input_str = json.dumps(input_data)
            else:
                # Empty dict {}: fragments will follow, use empty string
                input_str = ''
        else:
            input_str = str(input_data) if input_data else ''
        
        self.current_tool_call = {
            "id": data.get('toolUseId', generate_tool_call_id()),
            "type": "function",
            "function": {
                "name": data.get('name', ''),
                "arguments": input_str
            }
        }
        
        if data.get('stop'):
            self._finalize_tool_call()
        
        return None
    
    def _process_tool_input_event(self, data: dict) -> Optional[Dict[str, Any]]:
        """Processes input continuation for tool call."""
        if self.current_tool_call:
            # input can be string or object
            input_data = data.get('input', '')
            if isinstance(input_data, dict):
                if input_data:
                    input_str = json.dumps(input_data)
                else:
                    input_str = ''
            else:
                input_str = str(input_data) if input_data else ''
            self.current_tool_call['function']['arguments'] += input_str
        return None
    
    def _process_tool_stop_event(self, data: dict) -> Optional[Dict[str, Any]]:
        """Processes tool call end."""
        if self.current_tool_call and data.get('stop'):
            self._finalize_tool_call()
        return None
    
    def _finalize_tool_call(self) -> None:
        """Finalizes current tool call and adds to list."""
        if not self.current_tool_call:
            return
        
        # Try to parse and normalize arguments as JSON
        args = self.current_tool_call['function']['arguments']
        tool_name = self.current_tool_call['function'].get('name', 'unknown')
        
        logger.debug(f"Finalizing tool call '{tool_name}' with raw arguments: {repr(args)[:200]}")
        
        if isinstance(args, str):
            if args.strip():
                try:
                    parsed = json.loads(args)
                    # Ensure result is a JSON string
                    self.current_tool_call['function']['arguments'] = json.dumps(parsed)
                    logger.debug(f"Tool '{tool_name}' arguments parsed successfully: {list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}")
                except json.JSONDecodeError as e:
                    # Analyze the failure to provide better diagnostics
                    truncation_info = self._diagnose_json_truncation(args)
                    
                    if truncation_info["is_truncated"]:
                        # Mark for recovery system
                        self.current_tool_call['_truncation_detected'] = True
                        self.current_tool_call['_truncation_info'] = truncation_info
                        
                        # Check if recovery is enabled
                        from kiro.config import TRUNCATION_RECOVERY
                        tool_id = self.current_tool_call.get('id', 'unknown')
                        
                        # Clear error message: this is Kiro API's fault, not ours
                        logger.error(
                            f"Tool call truncated by Kiro API: "
                            f"tool='{tool_name}', id={tool_id}, size={truncation_info['size_bytes']} bytes, "
                            f"reason={truncation_info['reason']}. "
                            f"This is a Kiro API limitation. "
                            f"{'Model will be notified automatically about truncation.' if TRUNCATION_RECOVERY else 'Set TRUNCATION_RECOVERY=true in .env to auto-notify model about truncation.'}"
                        )
                    else:
                        # Regular JSON parse error
                        logger.warning(f"Failed to parse tool '{tool_name}' arguments: {e}. Raw: {args[:200]}")
                    
                    self.current_tool_call['function']['arguments'] = "{}"
            else:
                # Empty string - use empty object
                # This is normal behavior for duplicate tool calls from Kiro
                logger.debug(f"Tool '{tool_name}' has empty arguments string (will be deduplicated)")
                self.current_tool_call['function']['arguments'] = "{}"
        elif isinstance(args, dict):
            # If already an object - serialize to string
            self.current_tool_call['function']['arguments'] = json.dumps(args)
            logger.debug(f"Tool '{tool_name}' arguments already dict with keys: {list(args.keys())}")
        else:
            # Unknown type - empty object
            logger.warning(f"Tool '{tool_name}' has unexpected arguments type: {type(args)}")
            self.current_tool_call['function']['arguments'] = "{}"
        
        self.tool_calls.append(self.current_tool_call)
        self.current_tool_call = None
    
    def _diagnose_json_truncation(self, json_str: str) -> Dict[str, Any]:
        """
        Analyzes a malformed JSON string to determine if it was truncated.
        
        This helps distinguish between upstream issues (Kiro API cutting off
        large tool call arguments) and actual malformed JSON from the model.
        
        Args:
            json_str: The raw JSON string that failed to parse
        
        Returns:
            Dictionary with diagnostic information:
            - is_truncated: True if the JSON appears to be cut off
            - reason: Human-readable explanation of why it's truncated
            - size_bytes: Size of the received data
        """
        size_bytes = len(json_str.encode('utf-8'))
        stripped = json_str.strip()
        
        # Check for obvious truncation signs
        if not stripped:
            return {"is_truncated": False, "reason": "empty string", "size_bytes": size_bytes}
        
        # Count braces and brackets (simplified, doesn't account for strings perfectly)
        open_braces = stripped.count('{')
        close_braces = stripped.count('}')
        open_brackets = stripped.count('[')
        close_brackets = stripped.count(']')
        
        # Check if JSON starts with { but doesn't end with }
        if stripped.startswith('{') and not stripped.endswith('}'):
            missing = open_braces - close_braces
            return {
                "is_truncated": True,
                "reason": f"missing {missing} closing brace(s)",
                "size_bytes": size_bytes
            }
        
        # Check if JSON starts with [ but doesn't end with ]
        if stripped.startswith('[') and not stripped.endswith(']'):
            missing = open_brackets - close_brackets
            return {
                "is_truncated": True,
                "reason": f"missing {missing} closing bracket(s)",
                "size_bytes": size_bytes
            }
        
        # Check for unbalanced braces/brackets
        if open_braces != close_braces:
            diff = open_braces - close_braces
            return {
                "is_truncated": True,
                "reason": f"unbalanced braces ({open_braces} open, {close_braces} close)",
                "size_bytes": size_bytes
            }
        
        if open_brackets != close_brackets:
            diff = open_brackets - close_brackets
            return {
                "is_truncated": True,
                "reason": f"unbalanced brackets ({open_brackets} open, {close_brackets} close)",
                "size_bytes": size_bytes
            }
        
        # Check for unclosed string (ends with backslash or inside quotes)
        # This is a heuristic - count unescaped quotes
        quote_count = 0
        i = 0
        while i < len(stripped):
            if stripped[i] == '\\' and i + 1 < len(stripped):
                i += 2  # Skip escaped character
                continue
            if stripped[i] == '"':
                quote_count += 1
            i += 1
        
        if quote_count % 2 != 0:
            return {
                "is_truncated": True,
                "reason": "unclosed string literal",
                "size_bytes": size_bytes
            }
        
        # Doesn't look truncated, probably just malformed
        return {"is_truncated": False, "reason": "malformed JSON", "size_bytes": size_bytes}
    
    def get_tool_calls(self) -> List[Dict[str, Any]]:
        """
        Returns all collected tool calls.
        
        Finalizes current tool call if not finished.
        Removes duplicates.
        
        Returns:
            List of unique tool calls
        """
        if self.current_tool_call:
            self._finalize_tool_call()
        return deduplicate_tool_calls(self.tool_calls)
    
    def reset(self) -> None:
        """Resets parser state."""
        self.buffer = ""
        self.last_content = None
        self.current_tool_call = None
        self.tool_calls = []
        self._init_decoder_state()