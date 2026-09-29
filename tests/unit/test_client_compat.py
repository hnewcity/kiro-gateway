# -*- coding: utf-8 -*-

"""Tests for client-compatibility helpers: error mapping, SSE pings, stable conversation IDs."""

import asyncio
import json

import pytest

from kiro.kiro_errors import format_prompt_too_long, map_upstream_error_for_anthropic
from kiro.streaming_anthropic import PING_SSE_EVENT, stream_error_event, with_sse_pings
from kiro.utils import derive_conversation_id, extract_session_id


class TestErrorMapping:
    def test_prompt_too_long_official_wording(self):
        assert format_prompt_too_long(250000, 200000) == "prompt is too long: 250000 tokens > 200000 maximum"

    def test_prompt_too_long_never_self_contradictory(self):
        assert format_prompt_too_long(0, 200000) == "prompt is too long: 200001 tokens > 200000 maximum"

    def test_context_overflow(self):
        m = map_upstream_error_for_anthropic(400, "CONTENT_LENGTH_EXCEEDS_THRESHOLD", "x", max_input_tokens=1000)
        assert (m.status_code, m.error_type) == (400, "invalid_request_error")
        assert m.message.startswith("prompt is too long:")

    def test_429_rate_limit_with_retry_after(self):
        m = map_upstream_error_for_anthropic(429, "UNKNOWN", "slow down")
        assert (m.status_code, m.error_type) == (429, "rate_limit_error")
        assert m.headers == {"Retry-After": "5"}

    def test_quota_is_billing_error(self):
        m = map_upstream_error_for_anthropic(402, "MONTHLY_REQUEST_COUNT", "quota")
        assert (m.status_code, m.error_type) == (402, "billing_error")

    @pytest.mark.parametrize("status", [500, 502, 503, 408])
    def test_server_errors_overloaded(self, status):
        m = map_upstream_error_for_anthropic(status, "UNKNOWN", "boom")
        assert (m.status_code, m.error_type) == (status, "overloaded_error")

    def test_plain_400_invalid_request(self):
        assert map_upstream_error_for_anthropic(400, "UNKNOWN", "bad").error_type == "invalid_request_error"

    def test_other_status_api_error(self):
        assert map_upstream_error_for_anthropic(404, "UNKNOWN", "nf").error_type == "api_error"


class TestStreamErrorEvent:
    def test_forwards_structured_error(self):
        err = Exception(json.dumps({"type": "error", "error": {"type": "rate_limit_error", "message": "m"}}))
        event = stream_error_event(err)
        assert event.startswith("event: error\n")
        assert '"rate_limit_error"' in event

    def test_plain_exception_is_overloaded(self):
        event = stream_error_event(RuntimeError("connection reset"))
        assert '"overloaded_error"' in event
        assert "connection reset" in event


class TestSsePings:
    @pytest.mark.asyncio
    async def test_ping_emitted_while_idle(self):
        async def slow():
            await asyncio.sleep(0.05)
            yield "a"

        out = [c async for c in with_sse_pings(slow(), 0.01)]
        assert out[-1] == "a"
        assert PING_SSE_EVENT in out

    @pytest.mark.asyncio
    async def test_no_ping_when_busy(self):
        async def fast():
            for c in ("a", "b", "c"):
                yield c

        assert [c async for c in with_sse_pings(fast(), 1.0)] == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_disabled(self):
        async def fast():
            yield "a"

        assert [c async for c in with_sse_pings(fast(), 0)] == ["a"]

    @pytest.mark.asyncio
    async def test_source_error_propagates(self):
        async def broken():
            yield "a"
            raise ValueError("upstream")

        out = []
        with pytest.raises(ValueError):
            async for c in with_sse_pings(broken(), 1.0):
                out.append(c)
        assert out == ["a"]

    @pytest.mark.asyncio
    async def test_consumer_close_closes_source(self):
        closed = asyncio.Event()

        async def endless():
            try:
                while True:
                    await asyncio.sleep(0.2)
                    yield "x"
            finally:
                closed.set()

        gen = with_sse_pings(endless(), 0.01)
        assert await gen.__anext__() == PING_SSE_EVENT
        await gen.aclose()
        assert closed.is_set()


class TestConversationId:
    SESSION = "1b2c3d4e-0000-4000-8000-000000000000"

    @pytest.mark.parametrize("user_id", [
        SESSION,
        SESSION.upper(),
        json.dumps({"session_id": SESSION, "device_id": "d"}),
        json.dumps({"id": SESSION}),
        f"user_abc_account__session_{SESSION}",
    ])
    def test_extract_session(self, user_id):
        assert extract_session_id(user_id) == self.SESSION

    @pytest.mark.parametrize("user_id", [None, "", "user_abc", '{"session_id": "nope"}', "{bad json"])
    def test_extract_none(self, user_id):
        assert extract_session_id(user_id) is None

    def test_session_wins(self):
        assert derive_conversation_id({"user_id": self.SESSION}, "sys", ["a"], "hi") == self.SESSION

    def test_fallback_stable_and_order_independent(self):
        a = derive_conversation_id(None, "sys", ["b", "a"], "hi")
        b = derive_conversation_id({}, "sys", ["a", "b"], "hi")
        assert a == b
        assert len(a) == 36

    def test_fallback_ignores_billing_header(self):
        a = derive_conversation_id(None, "x-anthropic-billing-header: cc_version=1;\nYou are X", [], "hi")
        b = derive_conversation_id(None, "x-anthropic-billing-header: cc_version=2;\nYou are X", [], "hi")
        assert a == b

    def test_fallback_differs_by_first_message(self):
        assert derive_conversation_id(None, "s", [], "hi") != derive_conversation_id(None, "s", [], "yo")


# ==================================================================================================
# Upstream exception frames -> Anthropic output
# ==================================================================================================

from unittest.mock import AsyncMock, MagicMock, patch

from kiro.streaming_anthropic import collect_anthropic_response, stream_kiro_to_anthropic
from kiro.streaming_core import KiroEvent, KiroStreamError, StreamResult


def _cache():
    cache = MagicMock()
    cache.get_max_input_tokens.return_value = 200000
    return cache


def _response():
    response = AsyncMock()
    response.status_code = 200
    response.aclose = AsyncMock()
    return response


async def _run_stream(events):
    async def fake_parse(*args, **kwargs):
        for e in events:
            yield e

    out = []
    with patch("kiro.streaming_anthropic.parse_kiro_stream", fake_parse):
        async for chunk in stream_kiro_to_anthropic(_response(), "claude-sonnet-4", _cache(), MagicMock()):
            out.append(chunk)
    return out


class TestAnthropicExceptionFrames:
    @pytest.mark.asyncio
    async def test_content_length_exceeded_maps_to_max_tokens(self):
        out = await _run_stream([
            KiroEvent(type="content", content="partial"),
            KiroEvent(type="exception", exception_type="ContentLengthExceededException", exception_message="too long"),
        ])
        joined = "".join(out)
        assert '"stop_reason": "max_tokens"' in joined
        assert "message_stop" in joined

    @pytest.mark.asyncio
    async def test_other_exception_is_not_disguised_as_end_turn(self):
        out = []

        async def fake_parse(*args, **kwargs):
            yield KiroEvent(type="content", content="partial")
            yield KiroEvent(type="exception", exception_type="ThrottlingException", exception_message="slow")

        with patch("kiro.streaming_anthropic.parse_kiro_stream", fake_parse):
            with pytest.raises(KiroStreamError):
                async for chunk in stream_kiro_to_anthropic(_response(), "claude-sonnet-4", _cache(), MagicMock()):
                    out.append(chunk)
        joined = "".join(out)
        assert "message_delta" not in joined
        assert "end_turn" not in joined
        # Route layer turns the exception into a retryable error event
        assert '"overloaded_error"' in stream_error_event(KiroStreamError("ThrottlingException", "slow"))

    @pytest.mark.asyncio
    async def test_non_streaming_content_length_exceeded(self):
        result = StreamResult(content="partial", exception_type="ContentLengthExceededException")
        with patch("kiro.streaming_anthropic.collect_stream_to_result", AsyncMock(return_value=result)):
            data = await collect_anthropic_response(_response(), "claude-sonnet-4", _cache(), MagicMock())
        assert data["stop_reason"] == "max_tokens"

    @pytest.mark.asyncio
    async def test_non_streaming_other_exception_raises(self):
        result = StreamResult(content="partial", exception_type="InternalServerException", exception_message="x")
        with patch("kiro.streaming_anthropic.collect_stream_to_result", AsyncMock(return_value=result)):
            with pytest.raises(KiroStreamError):
                await collect_anthropic_response(_response(), "claude-sonnet-4", _cache(), MagicMock())


class TestAdditionalFieldsGating:
    @pytest.mark.parametrize("model,skipped", [
        ("claude-sonnet-4", True),        # verified live: Kiro rejects with REQUEST_BODY_INVALID
        ("claude-sonnet-4.5", True),
        ("claude-sonnet-4-5-20250929", True),
        ("claude-haiku-4.5", True),
        ("claude-3.7-sonnet", True),
        ("claude-opus-4.6", False),
        ("claude-opus-4.7", False),
        ("claude-sonnet-5", False),
        ("auto-kiro", True),
        ("deepseek-3.2", True),
    ])
    def test_version_gate(self, model, skipped):
        from kiro.converters_core import additional_fields_skipped

        assert additional_fields_skipped(model) is skipped
