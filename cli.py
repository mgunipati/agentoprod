import argparse
import contextlib
import json
import logging
import sys
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any

from strands.models import Model
from strands.multiagent.base import MultiAgentBase
from strands.multiagent.swarm import SwarmResult

from app.agendev.agent_factory import (
    AgentSettings,
    create_agent,
    create_multi_agent_model,
    prompt_error,
    reply_text,
    rollback_on_failure,
)
from app.agendev.report_graph import create_report_graph, report_text
from app.agendev.research_swarm import create_research_swarm, handoff_path, swarm_answer


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        for key in (
            "model_id", "region", "duration_ms", "stop_reason", "request_id",
            "status", "error_type", "tool_name", "tool_use_id", "token_usage",
            "cycles", "path",
        ):
            if hasattr(record, key):
                entry[key] = getattr(record, key)
        return json.dumps(entry, ensure_ascii=True)


def configure_logging(level: int = logging.WARNING) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonLogFormatter())
    logging.basicConfig(
        level=level,
        handlers=[handler],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a general-purpose Amazon Bedrock assistant.")
    parser.add_argument("--prompt", help="Send one prompt and exit; otherwise start an interactive chat.")
    parser.add_argument("--model-id", help="Override BEDROCK_MODEL_ID.")
    parser.add_argument("--region", help="Override AWS_REGION / AWS_DEFAULT_REGION.")
    parser.add_argument("--max-tokens", type=int, help="Override BEDROCK_MAX_TOKENS.")
    parser.add_argument("--observe", action="store_true", help="Log local Strands lifecycle and usage telemetry.")
    once = parser.add_mutually_exclusive_group()
    once.add_argument(
        "--report", metavar="REQUEST",
        help="Run the fetch → analyse → review stock report Graph once and exit.",
    )
    once.add_argument(
        "--swarm", metavar="REQUEST",
        help="Run the researcher/analyst/writer Swarm once and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        settings = AgentSettings.from_env(
            model_id=args.model_id,
            region=args.region,
            max_tokens=args.max_tokens,
        )
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    configure_logging(logging.INFO if args.observe else logging.WARNING)
    logger = logging.getLogger("agentoprod")
    logger.info("agent_started", extra={"model_id": settings.model_id, "region": settings.region})
    if args.report is not None:
        return run_once("report", args.report, settings, logger, create_report_graph, report_text)
    if args.swarm is not None:
        return run_once("swarm", args.swarm, settings, logger, create_research_swarm, swarm_answer)
    assistant = create_agent(settings, observe=args.observe)
    # Run each request on a worker thread so Ctrl+C can cancel it cleanly.
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        return run(args, settings, assistant, executor, logger)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def run_once(
    name: str,
    request: str,
    settings: AgentSettings,
    logger: logging.Logger,
    build: Callable[[Model], MultiAgentBase],
    answer: Callable[[Any], str],
) -> int:
    """Run a multi-agent pattern (Graph or Swarm) once; print its answer or why it failed."""
    error = prompt_error(request, settings)
    if error:
        print(error, file=sys.stderr)
        return 1
    started = time.monotonic()
    try:
        result = build(create_multi_agent_model(settings))(request)
        text = answer(result)
    except KeyboardInterrupt:
        print(f"\n{name.capitalize()} cancelled.", file=sys.stderr)
        return 130
    except Exception as exc:
        logger.error(f"{name}_failed", extra={"error_type": type(exc).__name__})
        print(f"The {name} failed ({type(exc).__name__}).", file=sys.stderr)
        return 1
    extra = {
        "duration_ms": round((time.monotonic() - started) * 1000),
        "token_usage": dict(result.accumulated_usage),
    }
    if isinstance(result, SwarmResult):
        extra["path"] = handoff_path(result)  # which agents the swarm chose
    logger.info(f"{name}_completed", extra=extra)
    print(text)
    return 0


def run(args, settings, assistant, executor, logger) -> int:

    def respond(prompt: str) -> bool:
        error = prompt_error(prompt, settings)
        if error:
            print(error, file=sys.stderr)
            return False

        started = time.monotonic()
        request_id = str(uuid.uuid4()) if args.observe else None
        try:
            with rollback_on_failure(assistant):
                future = executor.submit(
                    assistant, prompt,
                    invocation_state={"request_id": request_id} if request_id else None,
                )
                try:
                    result = future.result()
                except KeyboardInterrupt:
                    # Stop the running turn and wait for it before the rollback.
                    assistant.cancel()
                    with contextlib.suppress(Exception):
                        future.result()
                    raise
        except KeyboardInterrupt:
            logger.warning(
                "agent_invocation_cancelled",
                extra={
                    "request_id": request_id,
                    "duration_ms": round((time.monotonic() - started) * 1000),
                },
            )
            print("\nRequest cancelled.", file=sys.stderr)
            return False
        except Exception as exc:
            logger.error(
                "agent_invocation_failed",
                extra={
                    "request_id": request_id,
                    "duration_ms": round((time.monotonic() - started) * 1000),
                    "error_type": type(exc).__name__,
                },
            )
            print(f"The request failed ({type(exc).__name__}).", file=sys.stderr)
            return False

        invocation = result.metrics.latest_agent_invocation if args.observe else None
        logger.info(
            "agent_invocation_completed",
            extra={
                "duration_ms": round((time.monotonic() - started) * 1000),
                "stop_reason": result.stop_reason,
                **(
                    {
                        "request_id": request_id,
                        "token_usage": dict(invocation.usage) if invocation is not None else None,
                        "cycles": len(invocation.cycles) if invocation is not None else None,
                    }
                    if args.observe else {}
                ),
            },
        )
        print(reply_text(result))
        return True

    if args.prompt is not None:
        return 0 if respond(args.prompt) else 1

    print("Assistant ready. Type /exit to end the conversation.")
    while True:
        try:
            prompt = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if prompt.strip().lower() in {"/exit", "/quit"}:
            return 0
        respond(prompt)


if __name__ == "__main__":
    raise SystemExit(main())
