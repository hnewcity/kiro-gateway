# -*- coding: utf-8 -*-

"""
Profile ARN selection for Kiro runtime payloads.

Kiro Desktop credentials use a CodeWhisperer profile ARN in
generateAssistantResponse payloads. AWS SSO OIDC credentials must not send
profileArn, even if a local credential source exposes one.
"""

from typing import Optional, Protocol

from kiro.auth import AuthType


class ProfileArnCarrier(Protocol):
    """Auth object fields needed to decide whether profileArn is allowed."""

    @property
    def auth_type(self) -> AuthType:
        """Authentication type used by the account."""
        ...

    @property
    def profile_arn(self) -> Optional[str]:
        """AWS CodeWhisperer profile ARN if available."""
        ...


def profile_arn_for_payload(auth_manager: ProfileArnCarrier) -> str:
    """
    Return the profile ARN that should be sent to Kiro runtime.

    Args:
        auth_manager: Auth manager for the selected account.

    Returns:
        The Kiro Desktop profile ARN, or an empty string when the selected
        account should not send profileArn.
    """
    if auth_manager.auth_type != AuthType.KIRO_DESKTOP:
        return ""

    return auth_manager.profile_arn or ""
