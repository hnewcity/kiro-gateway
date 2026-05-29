# -*- coding: utf-8 -*-

"""
Profile ARN selection for Kiro runtime payloads.

Kiro Desktop credentials use a CodeWhisperer profile ARN in
generateAssistantResponse payloads. Enterprise Kiro IDE uses AWS SSO OIDC for
token refresh but still requires profileArn in runtime payloads. Plain kiro-cli
AWS SSO OIDC requests should not send profileArn.
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

    @property
    def is_enterprise_ide(self) -> bool:
        """Whether the account is Enterprise Kiro IDE."""
        ...


def profile_arn_for_payload(auth_manager: ProfileArnCarrier) -> str:
    """
    Return the profile ARN that should be sent to Kiro runtime.

    Args:
        auth_manager: Auth manager for the selected account.

    Returns:
        The profile ARN for Kiro Desktop or Enterprise Kiro IDE accounts, or an
        empty string when the selected account should not send profileArn.
    """
    should_send_profile_arn = (
        auth_manager.auth_type == AuthType.KIRO_DESKTOP
        or getattr(auth_manager, "is_enterprise_ide", False)
    )
    if not should_send_profile_arn:
        return ""

    return auth_manager.profile_arn or ""
