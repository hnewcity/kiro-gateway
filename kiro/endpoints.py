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
Kiro upstream endpoints for generateAssistantResponse and per-endpoint 429 tracking.

Different hosts/targets are reported to sit behind independent rate-limit buckets,
so on 429 the HTTP client can rotate to the next configured endpoint instead of
only backing off in place. Enabled endpoints come from KIRO_ENDPOINTS (default:
runtime only, i.e. the historical behaviour).
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from loguru import logger

from kiro.config import ENDPOINT_THROTTLE_SECONDS, KIRO_ENDPOINTS

GENERATE_PATH = "/generateAssistantResponse"

# Target sent on the runtime endpoint historically; kept to avoid behaviour change.
CODEWHISPERER_TARGET = "AmazonCodeWhispererStreamingService.GenerateAssistantResponse"
AMAZONQ_TARGET = "AmazonQDeveloperStreamingService.SendMessage"

KNOWN_ENDPOINTS = ("runtime", "ide", "codewhisperer", "amazonq")


@dataclass(frozen=True)
class Endpoint:
    """
    A concrete upstream endpoint.

    Attributes:
        name: Endpoint name (runtime / ide / codewhisperer / amazonq)
        base_url: Scheme + host, e.g. "https://runtime.us-east-1.kiro.dev"
        amz_target: Value for the x-amz-target header, or None to omit it
    """
    name: str
    base_url: str
    amz_target: Optional[str]

    @property
    def url(self) -> str:
        """Full generateAssistantResponse URL."""
        return f"{self.base_url}{GENERATE_PATH}"


def build_endpoint(name: str, region: str) -> Endpoint:
    """
    Builds an endpoint for the given name and region.

    Args:
        name: One of KNOWN_ENDPOINTS
        region: AWS region (case-insensitive)

    Returns:
        Endpoint instance

    Raises:
        ValueError: If name is unknown
    """
    region = region.lower()
    if name == "runtime":
        return Endpoint(name, f"https://runtime.{region}.kiro.dev", CODEWHISPERER_TARGET)
    if name == "ide":
        return Endpoint(name, f"https://q.{region}.amazonaws.com", None)
    if name == "codewhisperer":
        # codewhisperer.* host only exists in us-east-1
        host = (
            "codewhisperer.us-east-1.amazonaws.com"
            if region == "us-east-1"
            else f"q.{region}.amazonaws.com"
        )
        return Endpoint(name, f"https://{host}", CODEWHISPERER_TARGET)
    if name == "amazonq":
        return Endpoint(name, f"https://q.{region}.amazonaws.com", AMAZONQ_TARGET)
    raise ValueError(f"Unknown Kiro endpoint: {name}")


def region_from_url(url: str) -> Optional[str]:
    """
    Extracts the region from a runtime/q/codewhisperer generateAssistantResponse URL.

    Args:
        url: Request URL

    Returns:
        Region string, or None if the URL is not a recognised Kiro endpoint
    """
    if not url.endswith(GENERATE_PATH):
        return None
    host = url.split("://", 1)[-1].split("/", 1)[0]
    parts = host.split(".")
    if len(parts) >= 3 and parts[0] in ("runtime", "q", "codewhisperer"):
        return parts[1]
    return None


def endpoint_candidates(url: str) -> List[Endpoint]:
    """
    Returns the ordered list of endpoints to try for a request URL.

    The first entry always serves the original URL so that callers passing a
    non-Kiro URL (or when only one endpoint is enabled) keep the exact old
    behaviour.

    Args:
        url: URL the caller asked for

    Returns:
        List of endpoints (at least one)
    """
    region = region_from_url(url)
    if region is None:
        return [Endpoint("custom", url[: -len(GENERATE_PATH)] if url.endswith(GENERATE_PATH) else url, None)]

    names = [n for n in KIRO_ENDPOINTS if n in KNOWN_ENDPOINTS] or ["runtime"]
    endpoints: List[Endpoint] = []
    seen: set = set()
    for name in names:
        ep = build_endpoint(name, region)
        key = (ep.base_url, ep.amz_target)
        if key not in seen:
            seen.add(key)
            endpoints.append(ep)
    return endpoints


class EndpointThrottleRegistry:
    """
    Remembers which (account, endpoint) buckets recently returned 429.

    Entries expire on their own; no background cleanup needed.
    """

    def __init__(self) -> None:
        """Initializes an empty registry."""
        self._until: Dict[Tuple[int, str], float] = {}

    def throttle(self, account_key: int, endpoint_name: str, seconds: float = ENDPOINT_THROTTLE_SECONDS) -> None:
        """
        Marks an endpoint as throttled for an account.

        Args:
            account_key: Stable key for the account (id of its auth manager)
            endpoint_name: Endpoint name
            seconds: Throttle duration
        """
        self._until[(account_key, endpoint_name)] = time.monotonic() + seconds
        logger.debug(f"Endpoint '{endpoint_name}' throttled for {seconds:.0f}s")

    def is_throttled(self, account_key: int, endpoint_name: str) -> bool:
        """
        Checks whether an endpoint is currently throttled for an account.

        Args:
            account_key: Stable key for the account
            endpoint_name: Endpoint name

        Returns:
            True if still inside the throttle window
        """
        until = self._until.get((account_key, endpoint_name))
        return until is not None and until > time.monotonic()

    def order(self, account_key: int, endpoints: List[Endpoint]) -> List[Endpoint]:
        """
        Orders endpoints so non-throttled ones come first (stable otherwise).

        Args:
            account_key: Stable key for the account
            endpoints: Candidate endpoints

        Returns:
            Reordered list
        """
        free = [e for e in endpoints if not self.is_throttled(account_key, e.name)]
        busy = [e for e in endpoints if self.is_throttled(account_key, e.name)]
        return free + busy

    def clear(self) -> None:
        """Clears all throttle state (used in tests)."""
        self._until.clear()


endpoint_registry = EndpointThrottleRegistry()
