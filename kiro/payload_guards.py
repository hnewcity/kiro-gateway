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
Payload size guard for Kiro API requests.

The Kiro API rejects payloads exceeding ~615KB with a misleading
"Improperly formed request." (reason: null) error. This module provides:
- Pre-flight size checking
- Auto-trimming of oldest history entries to fit under the limit

It also hosts the Kiro-level toolUse/toolResult pairing cleanup, because
both the converter and the trimmer need the same rules:
- a toolResult is valid only if it answers a toolUse of the immediately
  preceding assistant message and has not been answered before
- a toolUse is valid only if it receives a toolResult

Ported from sametakofficial's payload_guards.py, simplified.
"""

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

# Content used for user messages that only carry toolResults
TOOL_RESULT_ONLY_CONTENT = "(tool result above)"

# Fallback for messages whose content is truly empty (Kiro rejects empty content)
EMPTY_CONTENT_FALLBACK = "(empty placeholder)"


@dataclass
class PayloadTrimStats:
    """Statistics from a payload trim operation."""
    original_bytes: int
    final_bytes: int
    original_entries: int
    final_entries: int
    trimmed: bool


def check_payload_size(payload: Dict[str, Any]) -> int:
    """Return the serialized byte size of the payload as UTF-8 JSON."""
    return len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))


def _strip_empty_tool_uses(history: list) -> None:
    """Remove empty toolUses arrays in-place (Kiro quirk)."""
    for entry in history:
        assistant = entry.get("assistantResponseMessage")
        if assistant and "toolUses" in assistant and assistant["toolUses"] == []:
            del assistant["toolUses"]


def _align_to_user_message(history: list) -> list:
    """Ensure history starts with a userInputMessage entry."""
    while history and "userInputMessage" not in history[0]:
        history.pop(0)
    return history


def _tool_result_text(tool_result: Dict[str, Any]) -> str:
    """
    Extract plain text from a Kiro-format toolResult.

    Args:
        tool_result: Kiro toolResult ({"content": [{"text": ...}], "toolUseId": ...})

    Returns:
        Concatenated text content (may be empty)
    """
    content = tool_result.get("content")
    parts: List[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                parts.append(str(part["text"]))
    elif isinstance(content, str) and content:
        parts.append(content)
    return "\n".join(parts)


def _orphan_results_to_text(tool_results: List[Dict[str, Any]]) -> str:
    """
    Render orphaned toolResults as text so their context is not lost.

    Uses the same "[Tool Result (id)]" format as converters_core.tool_results_to_text.

    Args:
        tool_results: Orphaned Kiro-format toolResults

    Returns:
        Text representation (empty string if nothing to render)
    """
    parts = []
    for tr in tool_results:
        text = _tool_result_text(tr) or "(empty result)"
        tool_use_id = tr.get("toolUseId", "")
        header = f"[Tool Result ({tool_use_id})]" if tool_use_id else "[Tool Result]"
        parts.append(f"{header}\n{text}")
    return "\n\n".join(parts)


def _tool_use_ids(entry: Optional[Dict[str, Any]]) -> Set[str]:
    """
    Collect toolUseIds of an assistant history entry.

    Args:
        entry: History entry (any role) or None

    Returns:
        Set of toolUseIds (empty if the entry is not an assistant message)
    """
    ids: Set[str] = set()
    if not entry:
        return ids
    assistant = entry.get("assistantResponseMessage")
    if not assistant:
        return ids
    for tu in assistant.get("toolUses") or []:
        tool_use_id = tu.get("toolUseId") if isinstance(tu, dict) else None
        if tool_use_id:
            ids.add(tool_use_id)
    return ids


def _filter_tool_results(
    tool_results: List[Dict[str, Any]],
    valid_ids: Set[str],
    answered_ids: Set[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split toolResults into kept and orphaned, dropping duplicates.

    Args:
        tool_results: Kiro-format toolResults of one user message
        valid_ids: toolUseIds of the immediately preceding assistant message
        answered_ids: toolUseIds already answered (mutated: kept ids are added)

    Returns:
        Tuple of (kept results, orphaned results). Duplicates are in neither.
    """
    kept: List[Dict[str, Any]] = []
    orphaned: List[Dict[str, Any]] = []
    for tr in tool_results:
        tool_use_id = tr.get("toolUseId")
        if tool_use_id in answered_ids:
            logger.warning(f"Dropping duplicate toolResult for already answered toolUseId={tool_use_id}")
        elif tool_use_id in valid_ids:
            kept.append(tr)
            answered_ids.add(tool_use_id)
        else:
            logger.warning(f"Converting orphaned toolResult to text (no matching toolUse): toolUseId={tool_use_id}")
            orphaned.append(tr)
    return kept, orphaned


def repair_tool_pairing(
    history: List[Dict[str, Any]],
    current_tool_results: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Enforce Kiro toolUse/toolResult pairing on an assembled history (in place).

    Rules (reference: kiro2cc-proxy tools.rs validate_tool_pairing /
    remove_orphaned_tool_uses):
    - duplicate toolResults (toolUseId already answered) are dropped
    - orphaned toolResults (toolUseId not in the immediately preceding assistant
      message) are converted to text instead of silently dropped, so the model
      keeps their content
    - toolUses that never receive a toolResult (in history or in the current
      message) are removed from their assistant message

    Args:
        history: Kiro-format history entries (mutated in place)
        current_tool_results: Kiro-format toolResults of the current message,
            which may answer the toolUses of the last history entry

    Returns:
        Tuple of (kept current toolResults, text of orphaned current toolResults)
    """
    answered_ids: Set[str] = set()

    for i, entry in enumerate(history):
        user_msg = entry.get("userInputMessage")
        if not user_msg:
            continue
        ctx = user_msg.get("userInputMessageContext")
        if not ctx or not ctx.get("toolResults"):
            continue

        valid_ids = _tool_use_ids(history[i - 1]) if i > 0 else set()
        original = ctx["toolResults"]
        kept, orphaned = _filter_tool_results(original, valid_ids, answered_ids)
        if len(kept) == len(original):
            continue

        if kept:
            ctx["toolResults"] = kept
        else:
            del ctx["toolResults"]
            if not ctx:
                del user_msg["userInputMessageContext"]

        orphan_text = _orphan_results_to_text(orphaned)
        content = user_msg.get("content", "")
        if content == TOOL_RESULT_ONLY_CONTENT and not kept:
            # Message no longer carries results; replace the "results above" hint
            user_msg["content"] = orphan_text or EMPTY_CONTENT_FALLBACK
        elif orphan_text:
            user_msg["content"] = f"{content}\n\n{orphan_text}" if content else orphan_text

    kept_current: List[Dict[str, Any]] = []
    current_orphan_text = ""
    if current_tool_results:
        last_ids = _tool_use_ids(history[-1]) if history else set()
        kept_current, orphaned = _filter_tool_results(current_tool_results, last_ids, answered_ids)
        current_orphan_text = _orphan_results_to_text(orphaned)

    # Remove toolUses that never got a result
    removed = 0
    for entry in history:
        assistant = entry.get("assistantResponseMessage")
        if not assistant or not assistant.get("toolUses"):
            continue
        tool_uses = assistant["toolUses"]
        remaining = [tu for tu in tool_uses if tu.get("toolUseId") in answered_ids]
        if len(remaining) == len(tool_uses):
            continue
        removed += len(tool_uses) - len(remaining)
        if remaining:
            assistant["toolUses"] = remaining
        else:
            del assistant["toolUses"]
            if not str(assistant.get("content", "")).strip():
                assistant["content"] = EMPTY_CONTENT_FALLBACK
    if removed:
        logger.warning(f"Removed {removed} toolUse(s) without matching toolResult from history")

    return kept_current, current_orphan_text


def _repair_orphaned_tool_results(history: list) -> None:
    """
    Remove orphaned toolResults that reference toolUseIds not present
    in the preceding assistant message. Preserve orphaned text content
    inline with a marker.
    """
    for i, entry in enumerate(history):
        user_msg = entry.get("userInputMessage")
        if not user_msg:
            continue

        ctx = user_msg.get("userInputMessageContext")
        if not ctx or "toolResults" not in ctx:
            continue

        # Collect toolUseIds from the preceding assistant message
        valid_ids = set()
        if i > 0:
            prev_assistant = history[i - 1].get("assistantResponseMessage")
            if prev_assistant:
                for tu in prev_assistant.get("toolUses", []):
                    tool_use_id = tu.get("toolUseId")
                    if tool_use_id:
                        valid_ids.add(tool_use_id)

        kept = []
        orphaned_text_parts = []
        for tr in ctx["toolResults"]:
            if tr.get("toolUseId") in valid_ids:
                kept.append(tr)
            else:
                # Preserve text content from orphaned results
                content = tr.get("content")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("text"):
                            orphaned_text_parts.append(part["text"])
                elif isinstance(content, str) and content:
                    orphaned_text_parts.append(content)

        if len(kept) != len(ctx["toolResults"]):
            if kept:
                ctx["toolResults"] = kept
            else:
                del ctx["toolResults"]
                if not ctx:
                    del user_msg["userInputMessageContext"]

            # Append orphaned text to user message content
            if orphaned_text_parts:
                marker = "\n[trimmed tool result] " + "; ".join(orphaned_text_parts)
                current_content = user_msg.get("content", "")
                user_msg["content"] = current_content + marker


def trim_payload_to_limit(
    payload: Dict[str, Any],
    max_bytes: int,
    keep_prefix_entries: int = 0,
) -> PayloadTrimStats:
    """
    Trim oldest history entries so the serialized payload fits under max_bytes.

    Trims in user/assistant pairs (2 entries at a time), aligns start to
    userInputMessage, and repairs orphaned toolResults after trimming.

    Args:
        payload: Kiro payload (mutated in place)
        max_bytes: Target maximum serialized size in bytes
        keep_prefix_entries: Number of leading history entries that are never
            trimmed (e.g. 2 for the system prompt user/assistant pair)

    Returns:
        PayloadTrimStats describing the operation
    """
    original_bytes = check_payload_size(payload)
    history = payload.get("conversationState", {}).get("history")

    if not history:
        return PayloadTrimStats(
            original_bytes=original_bytes,
            final_bytes=original_bytes,
            original_entries=0,
            final_entries=0,
            trimmed=False,
        )

    original_entries = len(history)

    # Strip empty toolUses before measuring
    _strip_empty_tool_uses(history)

    keep = max(0, min(keep_prefix_entries, len(history)))

    # Trim pairs after the protected prefix until under limit (keep at least 2 trimmable entries)
    while len(history) - keep > 2 and check_payload_size(payload) > max_bytes:
        # Remove 2 entries (a user/assistant pair)
        history.pop(keep)
        history.pop(keep)

    # Align to userInputMessage boundary
    if keep:
        while len(history) > keep and "userInputMessage" not in history[keep]:
            history.pop(keep)
    else:
        _align_to_user_message(history)

    # Repair orphaned tool results after trimming
    _repair_orphaned_tool_results(history)

    final_bytes = check_payload_size(payload)
    return PayloadTrimStats(
        original_bytes=original_bytes,
        final_bytes=final_bytes,
        original_entries=original_entries,
        final_entries=len(history),
        trimmed=original_entries != len(history),
    )
