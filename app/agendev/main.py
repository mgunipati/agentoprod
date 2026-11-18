"""AgentCore Runtime entrypoint for the general-purpose assistant.

The agent itself comes from agent_factory.py (shared with the local CLI). This
file only adapts it to the AgentCore HTTP contract: POST /invocations and
GET /ping on port 8080, both provided by BedrockAgentCoreApp.

The payload's optional "mode" picks one of three multi-agent patterns:

- "chat" (default): one agent with tools and tutor agents; remembers the session
- "report": the fetch → analyse → review Graph from report_graph.py
- "swarm": the researcher/analyst/writer Swarm from research_swarm.py

AgentCore gives every runtime session its own microVM, so chat history can live
in this process's memory, keyed by session ID. It is lost when the session ends
or goes idle; AgentCore Memory is the way to keep it longer. Reports and swarms
are one-shot: each request builds new agents and keeps no history.

Who the user is: the deployed runtime uses a CUSTOM_JWT (Cognito) authorizer, so
AgentCore verifies the caller's token before the request arrives and forwards
its Authorization header. The token's `sub` is the user for both AgentCore
Memory and AgentCore Identity. Without a token (agentcore dev, the IAM setup
before JWT), the payload's optional "user_id" is used instead.
"""

import base64
import binascii
import json
import time
import uuid
from functools import lru_cache

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.types.exceptions import ConcurrencyException

from agents.agent_factory import (
    AgentSettings,
    blocked_by_guardrail,
    create_agent,
    create_multi_agent_model,
    prompt_error,
    reply_text,
    rollback_on_failure,
)
from agents.report_graph import create_report_graph, report_text
from agents.research_swarm import create_research_swarm, handoff_path, swarm_answer
from memory.memory_session import valid_memory_id
from observability.observability import log_fields_in_message

app = BedrockAgentCoreApp()
# AgentCore's log format prints only the message text, so put the fields in it.
log_fields_in_message()
log = app.logger
settings = AgentSettings.from_env()

# mode -> (build the pattern from a model, extract the final answer from its result)
ONE_SHOT_MODES = {
    "report": (create_report_graph, report_text),
    "swarm": (create_research_swarm, swarm_answer),
}
MODES = ("chat", *ONE_SHOT_MODES)


def token_user(headers: dict | None) -> str | None:
    """Return the `sub` claim of the forwarded bearer token, or None without one.

    The signature is not checked here: AgentCore's JWT authorizer has already
    validated the token (issuer, signature, expiry, allowed client) before the
    request reaches this container. Raises ValueError for a malformed token.
    """
    authorization = next((v for k, v in (headers or {}).items() if k.lower() == "authorization"), None)
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    parts = token.split(".")
    if scheme.lower() != "bearer" or len(parts) != 3:
        raise ValueError("Authorization header is not a bearer JWT")
    try:
        payload = parts[1] + "=" * (-len(parts[1]) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (binascii.Error, ValueError) as exc:
        raise ValueError("JWT payload is not valid base64 JSON") from exc
    sub = claims.get("sub") if isinstance(claims, dict) else None
    if not valid_memory_id(sub):
        raise ValueError("JWT has no usable sub claim")
    return sub


@lru_cache(maxsize=16)
def agent_for_session(session_id: str, user_id: str) -> Agent:
    """One chat agent per (session, user); with AgentCore Memory it reloads saved history."""
    if not valid_memory_id(session_id):
        log.warning("memory_skipped reason=session_id_format")
        return create_agent(settings, observe=True)
    return create_agent(settings, observe=True, session_id=session_id, actor_id=user_id)


@app.entrypoint
async def invoke(payload, context):
    if not isinstance(payload, dict) or "prompt" not in payload:
        return {"error": "Send a JSON body like {\"prompt\": \"...\"}."}
    error = prompt_error(payload["prompt"], settings)
    if error:
        return {"error": error}
    mode = payload.get("mode", "chat")
    if mode not in MODES:
        return {"error": f"mode must be one of: {', '.join(MODES)}."}
    # The verified token's user wins; a payload user_id is only for unauthenticated
    # local runs. Callers should never send personal data such as an email as user_id.
    try:
        user_id = token_user(getattr(context, "request_headers", None))
    except ValueError as exc:
        log.warning("token_user_rejected reason=%s", exc)
        return {"error": "The access token could not be read. Sign in again."}
    if user_id is None:
        user_id = payload.get("user_id", "anonymous")
        if not valid_memory_id(user_id):
            return {"error": "user_id must start with a letter or digit and use only letters, digits, '-' or '_'."}

    request_id = str(uuid.uuid4())
    started = time.monotonic()

    def elapsed_ms() -> int:
        return round((time.monotonic() - started) * 1000)

    try:
        if mode == "chat":
            response = await run_chat(payload["prompt"], context, user_id, request_id)
        else:
            response = await run_one_shot(mode, payload["prompt"])
    except ConcurrencyException:
        # The same session sent a new chat message before the last one finished.
        log.warning("agent_busy request_id=%s", request_id)
        return {"error": "Still answering your previous message. Try again in a moment."}
    except Exception as exc:
        log.error(
            "agent_invocation_failed request_id=%s mode=%s duration_ms=%s error_type=%s",
            request_id, mode, elapsed_ms(), type(exc).__name__,
        )
        raise

    if blocked_by_guardrail(response["result"], settings):
        # The user got the Guardrail's block message, not an answer; Bedrock still said end_turn.
        response["stop_reason"] = "guardrail_intervened_output"
        log.warning("guardrail_blocked_output request_id=%s mode=%s", request_id, mode)

    log.info(
        "agent_invocation_completed request_id=%s mode=%s duration_ms=%s %s",
        request_id, mode, elapsed_ms(),
        " ".join(f"{key}={value}" for key, value in response.items() if key != "result"),
    )
    return {**response, "mode": mode}


async def run_chat(prompt: str, context, user_id: str, request_id: str) -> dict:
    """One turn with the session's agent; its history persists for the session."""
    session_id = getattr(context, "session_id", None) or "local"
    agent = agent_for_session(session_id, user_id)
    with rollback_on_failure(agent):
        result = await agent.invoke_async(prompt, invocation_state={"request_id": request_id})
    return {"result": reply_text(result), "stop_reason": result.stop_reason}


async def run_one_shot(mode: str, prompt: str) -> dict:
    """Run the Graph or Swarm once, with new agents and no history."""
    build, answer = ONE_SHOT_MODES[mode]
    result = await build(create_multi_agent_model(settings)).invoke_async(prompt)
    response = {"result": answer(result)}
    if mode == "swarm":
        response["path"] = handoff_path(result)  # which agents the swarm chose
    return response


if __name__ == "__main__":
    app.run()
