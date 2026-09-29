# -*- coding: utf-8 -*-

"""Tests for kiro/endpoints.py and endpoint rotation / backoff in http_client."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest

from kiro import endpoints as ep_mod
from kiro.endpoints import (
    AMAZONQ_TARGET,
    CODEWHISPERER_TARGET,
    EndpointThrottleRegistry,
    build_endpoint,
    endpoint_candidates,
    endpoint_registry,
    region_from_url,
)
from kiro.http_client import (
    KiroHttpClient,
    agent_mode_for_payload,
    backoff_delay,
    throttle_delay,
)
from kiro.utils import get_kiro_headers


@pytest.fixture(autouse=True)
def _clear_registry():
    endpoint_registry.clear()
    yield
    endpoint_registry.clear()


class TestBuildEndpoint:
    def test_runtime_keeps_legacy_target(self):
        ep = build_endpoint("runtime", "US-EAST-1")
        assert ep.url == "https://runtime.us-east-1.kiro.dev/generateAssistantResponse"
        assert ep.amz_target == CODEWHISPERER_TARGET

    def test_ide_has_no_target(self):
        ep = build_endpoint("ide", "eu-central-1")
        assert ep.base_url == "https://q.eu-central-1.amazonaws.com"
        assert ep.amz_target is None

    def test_codewhisperer_host_only_in_us_east_1(self):
        assert build_endpoint("codewhisperer", "us-east-1").base_url == "https://codewhisperer.us-east-1.amazonaws.com"
        assert build_endpoint("codewhisperer", "eu-central-1").base_url == "https://q.eu-central-1.amazonaws.com"

    def test_amazonq_target(self):
        assert build_endpoint("amazonq", "us-east-1").amz_target == AMAZONQ_TARGET

    def test_unknown_raises(self):
        with pytest.raises(ValueError):
            build_endpoint("nope", "us-east-1")


class TestCandidates:
    def test_region_from_url(self):
        assert region_from_url("https://runtime.eu-central-1.kiro.dev/generateAssistantResponse") == "eu-central-1"
        assert region_from_url("https://api.example.com/test") is None

    def test_default_is_runtime_only(self, monkeypatch):
        monkeypatch.setattr(ep_mod, "KIRO_ENDPOINTS", ["runtime"])
        eps = endpoint_candidates("https://runtime.us-east-1.kiro.dev/generateAssistantResponse")
        assert [e.name for e in eps] == ["runtime"]

    def test_configured_order_and_unknown_filtered(self, monkeypatch):
        monkeypatch.setattr(ep_mod, "KIRO_ENDPOINTS", ["runtime", "bogus", "ide", "amazonq"])
        eps = endpoint_candidates("https://runtime.us-east-1.kiro.dev/generateAssistantResponse")
        assert [e.name for e in eps] == ["runtime", "ide", "amazonq"]

    def test_non_kiro_url_passthrough(self):
        eps = endpoint_candidates("https://api.example.com/test")
        assert len(eps) == 1 and eps[0].name == "custom"


class TestRegistry:
    def test_throttled_moves_to_back(self):
        reg = EndpointThrottleRegistry()
        eps = [build_endpoint("runtime", "us-east-1"), build_endpoint("ide", "us-east-1")]
        reg.throttle(1, "runtime", 30)
        assert [e.name for e in reg.order(1, eps)] == ["ide", "runtime"]
        # other accounts unaffected
        assert [e.name for e in reg.order(2, eps)] == ["runtime", "ide"]

    def test_expiry(self):
        reg = EndpointThrottleRegistry()
        reg.throttle(1, "runtime", -1)
        assert not reg.is_throttled(1, "runtime")


class TestDelays:
    def test_backoff_capped_with_jitter(self):
        for attempt in range(10):
            d = backoff_delay(attempt)
            assert 0 <= d <= 5 * 1.25

    def test_throttle_honours_retry_after(self):
        assert throttle_delay(0, "3") == 3.0
        assert throttle_delay(0, "999") == 8.0

    def test_throttle_bad_retry_after_falls_back(self):
        d = throttle_delay(0, "soon")
        assert 2.0 <= d <= 3.5


class TestHeaders:
    def _auth(self):
        auth = Mock()
        auth.fingerprint = "fp"
        return auth

    def test_defaults_unchanged(self):
        h = get_kiro_headers(self._auth(), "t")
        assert h["x-amz-target"] == CODEWHISPERER_TARGET
        assert h["amz-sdk-request"] == "attempt=1; max=3"
        assert h["x-amzn-kiro-agent-mode"] == "vibe"

    def test_target_omitted_and_attempt(self):
        h = get_kiro_headers(self._auth(), "t", amz_target=None, attempt=3, agent_mode="spectask")
        assert "x-amz-target" not in h
        assert h["amz-sdk-request"] == "attempt=3; max=3"
        assert h["x-amzn-kiro-agent-mode"] == "spectask"

    def test_agent_mode_from_payload(self):
        assert agent_mode_for_payload({"conversationState": {"agentTaskType": "spectask"}}) == "spectask"
        assert agent_mode_for_payload({"conversationState": {}}) == "vibe"
        assert agent_mode_for_payload(None) == "vibe"


class TestRotation:
    @pytest.mark.asyncio
    async def test_429_switches_endpoint_without_sleep(self, monkeypatch):
        monkeypatch.setattr(ep_mod, "KIRO_ENDPOINTS", ["runtime", "ide"])
        auth = Mock()
        auth.fingerprint = "fp"
        auth.get_access_token = AsyncMock(return_value="tok")
        client = KiroHttpClient(auth)

        r429 = Mock(status_code=429, headers={})
        r200 = Mock(status_code=200)
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.request = AsyncMock(side_effect=[r429, r200])

        with patch.object(client, "_get_client", return_value=mock_client):
            with patch("kiro.http_client.asyncio.sleep", new_callable=AsyncMock) as sleep:
                resp = await client.request_with_retry(
                    "POST",
                    "https://runtime.us-east-1.kiro.dev/generateAssistantResponse",
                    {"conversationState": {"agentTaskType": "spectask"}},
                )

        assert resp is r200
        sleep.assert_not_called()
        urls = [c.args[1] for c in mock_client.request.call_args_list]
        assert urls == [
            "https://runtime.us-east-1.kiro.dev/generateAssistantResponse",
            "https://q.us-east-1.amazonaws.com/generateAssistantResponse",
        ]
        second_headers = mock_client.request.call_args_list[1].kwargs["headers"]
        assert "x-amz-target" not in second_headers
        assert second_headers["amz-sdk-request"] == "attempt=2; max=3"
        assert second_headers["x-amzn-kiro-agent-mode"] == "spectask"

    @pytest.mark.asyncio
    async def test_408_is_retried(self):
        auth = Mock()
        auth.fingerprint = "fp"
        auth.get_access_token = AsyncMock(return_value="tok")
        client = KiroHttpClient(auth)
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.request = AsyncMock(side_effect=[Mock(status_code=408), Mock(status_code=200)])
        with patch.object(client, "_get_client", return_value=mock_client):
            with patch("kiro.http_client.asyncio.sleep", new_callable=AsyncMock) as sleep:
                resp = await client.request_with_retry("POST", "https://api.example.com/x", {})
        assert resp.status_code == 200
        sleep.assert_called_once()


class TestAdditionalFieldsFallback:
    @pytest.mark.asyncio
    async def test_400_on_additional_fields_strips_and_retries(self):
        from kiro import http_client as hc

        hc._ADDITIONAL_FIELDS_REJECTED.clear()
        auth = Mock()
        auth.fingerprint = "fp"
        auth.get_access_token = AsyncMock(return_value="tok")
        client = KiroHttpClient(auth)
        r400 = Mock(status_code=400)
        r400.aread = AsyncMock(return_value=b'{"message":"additionalModelRequestFields is not supported for this model"}')
        r400.aclose = AsyncMock()
        r200 = Mock(status_code=200)
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.request = AsyncMock(side_effect=[r400, r200])
        payload = {
            "conversationState": {"currentMessage": {"userInputMessage": {"modelId": "claude-x"}}},
            "additionalModelRequestFields": {"output_config": {"effort": "low"}},
        }
        with patch.object(client, "_get_client", return_value=mock_client):
            resp = await client.request_with_retry("POST", "https://api.example.com/x", payload)
        assert resp is r200
        sent = [json.loads(c.kwargs["content"]) for c in mock_client.request.call_args_list]
        assert "additionalModelRequestFields" in sent[0]
        assert "additionalModelRequestFields" not in sent[1]
        # learned: next request for the same model is stripped up front
        assert "additionalModelRequestFields" not in hc.strip_rejected_additional_fields(payload)
        assert "additionalModelRequestFields" in payload  # caller's dict untouched
        hc._ADDITIONAL_FIELDS_REJECTED.clear()

    @pytest.mark.asyncio
    async def test_other_400_returned_as_is(self):
        auth = Mock()
        auth.fingerprint = "fp"
        auth.get_access_token = AsyncMock(return_value="tok")
        client = KiroHttpClient(auth)
        r400 = Mock(status_code=400)
        r400.aread = AsyncMock(return_value=b'{"message":"Improperly formed request."}')
        mock_client = AsyncMock()
        mock_client.is_closed = False
        mock_client.request = AsyncMock(return_value=r400)
        payload = {"additionalModelRequestFields": {}, "conversationState": {}}
        with patch.object(client, "_get_client", return_value=mock_client):
            resp = await client.request_with_retry("POST", "https://api.example.com/x", payload)
        assert resp is r400
        assert mock_client.request.call_count == 1
