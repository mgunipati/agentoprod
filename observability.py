import logging
from typing import Any

from strands.hooks import HookRegistry
from strands.hooks.events import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)


class ConsoleObservationHooks:
    """Log lifecycle metadata without including model or tool content."""

    def __init__(self) -> None:
        self.logger = logging.getLogger("agentoprod.observe")

    def register_hooks(self, registry: HookRegistry, **kwargs: Any) -> None:
        registry.add_callback(BeforeModelCallEvent, self.before_model)
        registry.add_callback(AfterModelCallEvent, self.after_model)
        registry.add_callback(BeforeToolCallEvent, self.before_tool)
        registry.add_callback(AfterToolCallEvent, self.after_tool)

    def before_model(self, event: BeforeModelCallEvent) -> None:
        self.logger.info(
            "model_call_started",
            extra={"request_id": event.invocation_state.get("request_id")},
        )

    def after_model(self, event: AfterModelCallEvent) -> None:
        self.logger.info(
            "model_call_completed",
            extra={
                "request_id": event.invocation_state.get("request_id"),
                "status": "error" if event.exception else "completed",
                "error_type": type(event.exception).__name__ if event.exception else None,
                "stop_reason": event.stop_response.stop_reason if event.stop_response else None,
            },
        )

    def before_tool(self, event: BeforeToolCallEvent) -> None:
        self.logger.info(
            "tool_call_started",
            extra={
                "request_id": event.invocation_state.get("request_id"),
                "tool_name": event.tool_use.get("name"),
                "tool_use_id": event.tool_use.get("toolUseId"),
            },
        )

    def after_tool(self, event: AfterToolCallEvent) -> None:
        self.logger.info(
            "tool_call_completed",
            extra={
                "request_id": event.invocation_state.get("request_id"),
                "tool_name": event.tool_use.get("name"),
                "tool_use_id": event.tool_use.get("toolUseId"),
                "status": (
                    "cancelled" if event.cancel_message is not None
                    else "error" if event.exception or event.result.get("status") == "error"
                    else "completed"
                ),
                "duration_ms": round(event.duration * 1000) if event.duration is not None else None,
                "error_type": type(event.exception).__name__ if event.exception else None,
            },
        )
