"""AgentCore Runtime entrypoint for the general-purpose assistant.

The agent itself comes from agent_factory.py (shared with the local CLI). This
file only adapts it to the AgentCore HTTP contract: POST /invocations and
GET /ping on port 8080, both provided by BedrockAgentCoreApp.

AgentCore gives every runtime session its own microVM, so the conversation
history can live in this process's memory, keyed by session ID. It is lost when
the session ends or goes idle; AgentCore Memory is the way to keep it longer.
"""

from functools import lru_cache
import time
import uuid

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.types.exceptions import ConcurrencyException

from agent_factory import AgentSettings, create_agent, prompt_error, rollback_on_failure
from observability import log_fields_in_message

app = BedrockAgentCoreApp()
# AgentCore's log format prints only the message text, so put the fields in it.
log_fields_in_message()
log = app.logger
settings = AgentSettings.from_env()


@lru_cache(maxsize=16)
def agent_for_session(session_id: str) -> Agent:
    return create_agent(settings, observe=True)


@app.entrypoint
async def invoke(payload, context):
    if not isinstance(payload, dict) or "prompt" not in payload:
        return {"error": "Send a JSON body like {\"prompt\": \"...\"}."}
    error = prompt_error(payload["prompt"], settings)
    if error:
        return {"error": error}

    session_id = getattr(context, "session_id", None) or "local"
    agent = agent_for_session(session_id)
    request_id = str(uuid.uuid4())
    started = time.monotonic()

    try:
        with rollback_on_failure(agent):
            result = await agent.invoke_async(
                payload["prompt"],
                invocation_state={"request_id": request_id},
            )
    except ConcurrencyException:
        # The same session sent a new message before the last one finished.
        log.warning("agent_busy request_id=%s", request_id)
        return {"error": "Still answering your previous message. Try again in a moment."}
    except Exception as exc:
        log.error(
            "agent_invocation_failed request_id=%s duration_ms=%s error_type=%s",
            request_id,
            round((time.monotonic() - started) * 1000),
            type(exc).__name__,
        )
        raise

    log.info(
        "agent_invocation_completed request_id=%s duration_ms=%s stop_reason=%s",
        request_id,
        round((time.monotonic() - started) * 1000),
        result.stop_reason,
    )
    return {"result": str(result).strip(), "stop_reason": result.stop_reason}


if __name__ == "__main__":
    app.run()
