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

from agent_factory import AgentSettings, create_agent

app = BedrockAgentCoreApp()
log = app.logger
settings = AgentSettings.from_env()


@lru_cache(maxsize=16)
def agent_for_session(session_id: str) -> Agent:
    return create_agent(settings)


@app.entrypoint
async def invoke(payload, context):
    prompt = payload.get("prompt") if isinstance(payload, dict) else None
    if not isinstance(prompt, str) or not prompt.strip():
        return {"error": "Send a JSON body like {\"prompt\": \"...\"}."}
    if len(prompt) > settings.max_input_chars:
        return {"error": f"Prompt exceeds the {settings.max_input_chars}-character limit."}

    session_id = getattr(context, "session_id", None) or "local"
    agent = agent_for_session(session_id)

    # Same fix as the CLI: drop a failed turn so the next one doesn't send two
    # user messages in a row.
    history = list(agent.messages)
    try:
        result = await agent.invoke_async(prompt)
    except Exception as exc:
        agent.messages = history
        log.error("agent_invocation_failed error_type=%s", type(exc).__name__)
        raise

    log.info("agent_invocation_completed stop_reason=%s", result.stop_reason)
    return {"result": str(result).strip(), "stop_reason": result.stop_reason}


if __name__ == "__main__":
    app.run()
