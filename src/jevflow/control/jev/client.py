"""Port and adapter for the TypeSafe System One API (Ports & Adapters).

The strategy depends on the tiny :class:`DecisionModel` protocol, not on the SDK, so tests and
offline demos can inject a fake while production uses :class:`TypeSafeDecisionModel`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

from typesafe_sdk import AsyncTypeSafeClient, Question, RetryPolicy, SystemOneResponse

if TYPE_CHECKING:
    import httpx2


class DecisionModel(Protocol):
    model: str

    async def evaluate(
        self, state: Mapping[str, Any], questions: Mapping[str, Question]
    ) -> SystemOneResponse: ...

    async def aclose(self) -> None: ...


class TypeSafeDecisionModel:
    """Async adapter over ``AsyncTypeSafeClient`` tuned for a real-time control loop.

    A signal controller cannot wait long, so the time budget is short and retries are few. If
    the budget is blown, the strategy falls back to deterministic control.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout_s: float,
        max_retries: int,
        transport: httpx2.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        self._client = AsyncTypeSafeClient(
            api_key=api_key,
            model=model,
            timeout=timeout_s,
            retry=RetryPolicy(
                max_retries=max_retries, backoff_initial=0.2, backoff_max=0.5, timeout=timeout_s
            ),
            transport=transport,
        )

    async def evaluate(
        self, state: Mapping[str, Any], questions: Mapping[str, Question]
    ) -> SystemOneResponse:
        return await self._client.system_one(state=dict(state), questions=questions)

    async def aclose(self) -> None:
        await self._client.aclose()
