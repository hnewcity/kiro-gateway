# -*- coding: utf-8 -*-

"""
Unit tests for profile ARN selection.
"""

from unittest.mock import MagicMock, patch

from kiro.auth import AuthType
from kiro.profile_arn import profile_arn_for_payload


class TestProfileArnForPayload:
    """Tests for profileArn payload selection on runtime.kiro.dev."""

    def test_returns_profile_arn_for_kiro_desktop(self):
        """
        What it does: Verifies Kiro Desktop accounts send their profile ARN.
        Purpose: Runtime requires profileArn for Kiro Desktop accounts.
        """
        print("Setup: Creating Kiro Desktop auth manager with profile ARN...")
        auth_manager = MagicMock()
        auth_manager.auth_type = AuthType.KIRO_DESKTOP
        auth_manager.profile_arn = "arn:aws:codewhisperer:us-east-1:123456789:profile/test"

        print("Action: Selecting profileArn for payload...")
        result = profile_arn_for_payload(auth_manager)

        print(f"Comparing result: Expected profile ARN, Got '{result}'")
        assert result == auth_manager.profile_arn

    def test_returns_profile_arn_for_aws_sso(self):
        """
        What it does: Verifies kiro-cli AWS SSO accounts also send profileArn.
        Purpose: After the runtime.kiro.dev migration, profileArn is required
            for all auth types — including plain kiro-cli SSO OIDC, which used
            to be excluded under the legacy q.amazonaws.com endpoint.
        """
        print("Setup: Creating AWS SSO auth manager with profile ARN...")
        auth_manager = MagicMock()
        auth_manager.auth_type = AuthType.AWS_SSO_OIDC
        auth_manager.profile_arn = "arn:aws:codewhisperer:us-east-1:123456789:profile/test"

        print("Action: Selecting profileArn for payload...")
        result = profile_arn_for_payload(auth_manager)

        print(f"Comparing result: Expected profile ARN, Got '{result}'")
        assert result == auth_manager.profile_arn

    def test_falls_back_to_env_profile_arn(self):
        """
        What it does: Verifies PROFILE_ARN env var is used when auth manager has none.
        Purpose: Allow operators to provide a profileArn out-of-band when the
            credential source does not carry one.
        """
        print("Setup: Auth manager without profile ARN, PROFILE_ARN env set...")
        auth_manager = MagicMock()
        auth_manager.auth_type = AuthType.AWS_SSO_OIDC
        auth_manager.profile_arn = None

        env_arn = "arn:aws:codewhisperer:us-east-1:999999999:profile/env-fallback"
        with patch("kiro.profile_arn.PROFILE_ARN", env_arn):
            print("Action: Selecting profileArn for payload...")
            result = profile_arn_for_payload(auth_manager)

        print(f"Comparing result: Expected env ARN, Got '{result}'")
        assert result == env_arn

    def test_returns_empty_when_no_source_available(self):
        """
        What it does: Verifies an empty string is returned when no ARN is available.
        Purpose: Without an auth-manager ARN or PROFILE_ARN env var, payload
            sends an empty profileArn (runtime will still reject it, but the
            gateway should not invent a value).
        """
        print("Setup: Auth manager without profile ARN, PROFILE_ARN env empty...")
        auth_manager = MagicMock()
        auth_manager.auth_type = AuthType.AWS_SSO_OIDC
        auth_manager.profile_arn = None

        with patch("kiro.profile_arn.PROFILE_ARN", ""):
            print("Action: Selecting profileArn for payload...")
            result = profile_arn_for_payload(auth_manager)

        print(f"Comparing result: Expected empty string, Got '{result}'")
        assert result == ""
