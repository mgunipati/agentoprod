"""AgentCore Runtime entrypoint for the general-purpose assistant.

The agent itself comes from agent_factory.py (shared with the local CLI). This
file only adapts it to the AgentCore HTTP contract: POST /invocations and
GET /ping on port 8080, both provided by BedrockAgentCoreApp.

AgentCore gives every runtime session its own microVM, so the conversation
history can live in this process's memory, keyed by session ID. It is lost when
the session ends or goes idle; AgentCore Memory is the way to keep it longer.
"""

from functools import lru_cache

from bedrock_agentcore.runtime import BedrockAgentCoreApp
from strands import Agent
from strands.types.exceptions import ConcurrencyException

from agent_factory import AgentSettings, create_agent, prompt_error, rollback_on_failure

app = BedrockAgentCoreApp()
log = app.logger
settings = AgentSettings.from_env()


@lru_cache(maxsize=16)
def agent_for_session(session_id: str) -> Agent:
    return create_agent(settings)


@app.entrypoint
async def invoke(payload, context):
    if not isinstance(payload, dict) or "prompt" not in payload:
        return {"error": "Send a JSON body like {\"prompt\": \"...\"}."}
    error = prompt_error(payload["prompt"], settings)
    if error:
        return {"error": error}

    session_id = getattr(context, "session_id", None) or "local"
    agent = agent_for_session(session_id)

    try:
        with rollback_on_failure(agent):
            result = await agent.invoke_async(payload["prompt"])
    except ConcurrencyException:
        # The same session sent a new message before the last one finished.
        log.warning("agent_busy")
        return {"error": "Still answering your previous message. Try again in a moment."}
    except Exception as exc:
        log.error("agent_invocation_failed error_type=%s", type(exc).__name__)
        raise

    log.info("agent_invocation_completed stop_reason=%s", result.stop_reason)
    return {"result": str(result).strip(), "stop_reason": result.stop_reason}


if __name__ == "__main__":
    app.run()
