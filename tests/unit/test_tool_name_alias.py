# -*- coding: utf-8 -*-

"""Unit tests for the tool-name aliasing extension."""

import pytest

from extensions.tool_name_alias import (
    MAX_KIRO_TOOL_NAME_LENGTH,
    alias_for_tool_name,
    install_tool_name_aliasing,
    needs_alias,
    original_for_tool_name,
    uninstall_tool_name_aliasing,
)
from kiro import converters_core, parsers, streaming_anthropic, streaming_core, streaming_openai
from kiro.converters_core import UnifiedTool
from kiro.streaming_core import KiroEvent


LONG_NAME = "mcp__some-aggregator__service-x__do_something_extremely_long_for_kiro_limit"
ANOTHER_LONG_NAME = "mcp__other__svc__a" * 6  # also exceeds 64 chars


@pytest.fixture
def aliasing_installed():
    install_tool_name_aliasing()
    yield
    uninstall_tool_name_aliasing()


def test_needs_alias_detects_long_or_invalid_names():
    assert needs_alias(LONG_NAME)
    assert needs_alias("has spaces")
    assert not needs_alias("short_ok-1")


def test_alias_is_within_limit_and_stable(aliasing_installed):
    alias = alias_for_tool_name(LONG_NAME)
    assert len(alias) <= MAX_KIRO_TOOL_NAME_LENGTH
    assert alias != LONG_NAME
    # Same input must yield the same alias (stable mapping)
    assert alias_for_tool_name(LONG_NAME) == alias
    # Distinct inputs must produce distinct aliases
    assert alias_for_tool_name(ANOTHER_LONG_NAME) != alias
    # Round-trip resolution
    assert original_for_tool_name(alias) == LONG_NAME


def test_short_names_are_passthrough(aliasing_installed):
    assert alias_for_tool_name("short_ok") == "short_ok"
    assert original_for_tool_name("short_ok") == "short_ok"


def test_validate_tool_names_no_longer_raises_for_long_names(aliasing_installed):
    converters_core.validate_tool_names([UnifiedTool(name=LONG_NAME, description="x")])


def test_convert_tools_to_kiro_format_uses_alias(aliasing_installed):
    tools = [UnifiedTool(name=LONG_NAME, description="d", input_schema={"type": "object"})]
    result = converters_core.convert_tools_to_kiro_format(tools)
    assert len(result) == 1
    kiro_name = result[0]["toolSpecification"]["name"]
    assert kiro_name == alias_for_tool_name(LONG_NAME)
    assert len(kiro_name) <= MAX_KIRO_TOOL_NAME_LENGTH


def test_extract_tool_uses_aliases_outgoing_names(aliasing_installed):
    tool_calls = [{
        "id": "call_1",
        "type": "function",
        "function": {"name": LONG_NAME, "arguments": "{\"x\": 1}"},
    }]
    tool_uses = converters_core.extract_tool_uses_from_message(content="", tool_calls=tool_calls)
    assert tool_uses
    assert tool_uses[0]["name"] == alias_for_tool_name(LONG_NAME)


def test_parse_bracket_tool_calls_restores_original_name(aliasing_installed):
    alias = alias_for_tool_name(LONG_NAME)
    text = f"[Called {alias} with args: {{\"x\": 1}}]"
    calls = parsers.parse_bracket_tool_calls(text)
    assert calls
    assert calls[0]["function"]["name"] == LONG_NAME


@pytest.mark.asyncio
async def test_parse_kiro_stream_restores_event_tool_name(aliasing_installed, monkeypatch):
    alias = alias_for_tool_name(LONG_NAME)

    async def fake_original_stream(*_args, **_kwargs):
        yield KiroEvent(type="tool_use", tool_use={"id": "1", "name": alias, "function": {"name": alias}})

    # Replace the captured original; the wrapper still delegates through _originals dict,
    # so swap the entry directly to avoid running real Kiro stream parsing.
    from extensions import tool_name_alias as ext

    ext._originals["parse_kiro_stream"] = fake_original_stream

    events = [event async for event in streaming_core.parse_kiro_stream(None)]
    assert len(events) == 1
    assert events[0].tool_use["name"] == LONG_NAME
    assert events[0].tool_use["function"]["name"] == LONG_NAME


def test_streaming_modules_share_patched_bindings(aliasing_installed):
    # All three modules must use the patched parse_bracket_tool_calls so the
    # wrapper applies regardless of which streaming layer is invoked.
    assert streaming_core.parse_bracket_tool_calls is parsers.parse_bracket_tool_calls
    assert streaming_openai.parse_bracket_tool_calls is parsers.parse_bracket_tool_calls
    assert streaming_anthropic.parse_bracket_tool_calls is parsers.parse_bracket_tool_calls
    assert streaming_core.parse_kiro_stream is streaming_openai.parse_kiro_stream
    assert streaming_core.parse_kiro_stream is streaming_anthropic.parse_kiro_stream


def test_uninstall_restores_originals():
    install_tool_name_aliasing()
    # Snapshot originals
    original_validate = converters_core.validate_tool_names
    uninstall_tool_name_aliasing()
    # After uninstall, validate_tool_names must reject long names again
    with pytest.raises(ValueError):
        converters_core.validate_tool_names([UnifiedTool(name=LONG_NAME, description="x")])
    # And the function reference should differ from the patched one
    assert converters_core.validate_tool_names is not original_validate
