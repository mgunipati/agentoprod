import logging
from typing import Any

from strands.hooks import HookRegistry
from strands.hooks.events import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)

# Fields the hooks and tools pass as `extra`. JSON formatters (the CLI) print them
# as keys; plain-text formats (AgentCore Runtime) print only the message.
_FIELDS = ("request_id", "tool_name", "tool_use_id", "status", "stop_reason", "duration_ms", "error_type")


class FieldsInMessageFilter(logging.Filter):
    """Append `extra` fields to the message as key=value for plain-text logs."""

    def filter(self, record: logging.LogRecord) -> bool:
        parts = [f"{key}={getattr(record, key)}" for key in _FIELDS if getattr(record, key, None) is not None]
        if parts:
            record.msg = f"{record.msg} {' '.join(parts)}"
        return True


def log_fields_in_message() -> None:
    """Make the agent's own loggers show their fields in plain-text log formats."""
    for name in ("agentoprod.observe", "agentoprod.tools"):
        logger = logging.getLogger(name)
        if not any(isinstance(f, FieldsInMessageFilter) for f in logger.filters):
            logger.addFilter(FieldsInMessageFilter())


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
