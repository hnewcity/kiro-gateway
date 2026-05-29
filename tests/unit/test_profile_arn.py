# -*- coding: utf-8 -*-

"""
Unit tests for profile ARN selection.
"""

from unittest.mock import MagicMock

from kiro.auth import AuthType
from kiro.profile_arn import profile_arn_for_payload


class TestProfileArnForPayload:
    """Tests for profileArn payload selection."""

    def test_returns_profile_arn_for_kiro_desktop(self):
        """
        What it does: Verifies Kiro Desktop accounts send their profile ARN.
        Purpose: Ensure Enterprise/Kiro IDE runtime requests include profileArn.
        """
        print("Setup: Creating Kiro Desktop auth manager with profile ARN...")
        auth_manager = MagicMock()
        auth_manager.auth_type = AuthType.KIRO_DESKTOP
        auth_manager.profile_arn = "arn:aws:codewhisperer:us-east-1:123456789:profile/test"

        print("Action: Selecting profileArn for payload...")
        result = profile_arn_for_payload(auth_manager)

        print(f"Comparing result: Expected profile ARN, Got '{result}'")
        assert result == auth_manager.profile_arn

    def test_returns_empty_string_for_kiro_desktop_without_profile_arn(self):
        """
        What it does: Verifies Kiro Desktop accounts without profile ARN send nothing.
        Purpose: Avoid adding empty profileArn fields to Kiro payloads.
        """
        print("Setup: Creating Kiro Desktop auth manager without profile ARN...")
        auth_manager = MagicMock()
        auth_manager.auth_type = AuthType.KIRO_DESKTOP
        auth_manager.profile_arn = None

        print("Action: Selecting profileArn for payload...")
        result = profile_arn_for_payload(auth_manager)

        print(f"Comparing result: Expected empty string, Got '{result}'")
        assert result == ""

    def test_returns_empty_string_for_aws_sso_even_with_profile_arn(self):
        """
        What it does: Verifies AWS SSO OIDC accounts do not send profileArn.
        Purpose: Preserve the known-good behavior from the runtime migration.
        """
        print("Setup: Creating AWS SSO auth manager with discovered profile ARN...")
        auth_manager = MagicMock()
        auth_manager.auth_type = AuthType.AWS_SSO_OIDC
        auth_manager.profile_arn = "arn:aws:codewhisperer:us-east-1:123456789:profile/test"

        print("Action: Selecting profileArn for payload...")
        result = profile_arn_for_payload(auth_manager)

        print(f"Comparing result: Expected empty string, Got '{result}'")
        assert result == ""
