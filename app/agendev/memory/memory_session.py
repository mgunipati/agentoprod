"""AgentCore Memory for the chat agent: history that outlives the microVM.

The AgentCore CLI's `chat` memory (agentcore/agentcore.json) gives the deployed
runtime MEMORY_CHAT_ID plus IAM access to it. With it set, the chat agent gets
an AgentCoreMemorySessionManager, which:

- short-term: saves every message as an event for (actor, session) and reloads
  them when a new microVM creates the agent for the same session
- long-term: AgentCore extracts facts and preferences per actor in the
  background; before each user message, the matching records are retrieved and
  added to it inside <user_context> tags

Without MEMORY_CHAT_ID (the local CLI, tests) the agent keeps history in
process memory only, as before. bedrock_agentcore is imported only when memory
is on, because only the runtime installs it.
"""

import os
import re

from strands.types.content import Message

MEMORY_ID_ENV = "MEMORY_CHAT_ID"

# Must match the strategies' namespaceTemplates in agentcore/agentcore.json.
# Every retrieved record is added to the user's message, and the model tends to
# act on it (an EMI question also fetched a remembered AAPL quote). Unrelated
# records score about 0.33-0.37 and related ones 0.64-0.75, so 0.5 keeps only
# memories about the current question.
RETRIEVAL = {
    "/users/{actorId}/facts": {"top_k": 5, "relevance_score": 0.5},
    "/users/{actorId}/preferences": {"top_k": 5, "relevance_score": 0.5},
}

# AgentCore Memory IDs for actors and sessions: start alphanumeric, then
# letters, digits, '-' or '_'.
_MEMORY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")


def valid_memory_id(value: object) -> bool:
    return isinstance(value, str) and bool(_MEMORY_ID.match(value))


def memory_session_manager(session_id: str, actor_id: str, region: str):
    """Return a session manager for this actor and session, or None if memory is off."""
    memory_id = os.getenv(MEMORY_ID_ENV)
    if not memory_id:
        return None

    from bedrock_agentcore.memory.integrations.strands.config import AgentCoreMemoryConfig, RetrievalConfig
    from bedrock_agentcore.memory.integrations.strands.session_manager import AgentCoreMemorySessionManager

    config = AgentCoreMemoryConfig(
        memory_id=memory_id,
        session_id=session_id,
        actor_id=actor_id,
        retrieval_config={namespace: RetrievalConfig(**options) for namespace, options in RETRIEVAL.items()},
    )
    return AgentCoreMemorySessionManager(config, region_name=region)


def _starts_turn(message: Message) -> bool:
    # A turn starts with the user's text; tool results are also "user" messages.
    return message["role"] == "user" and not any("toolResult" in block for block in message["content"])


def _finished(turn: list[Message]) -> bool:
    last = turn[-1]
    return last["role"] == "assistant" and not any("toolUse" in block for block in last["content"])


def drop_unfinished_turns(messages: list[Message]) -> list[Message]:
    """Remove restored turns that never got a final assistant reply.

    Memory saves each message as it is added, so a turn that failed still has
    its user message (and maybe tool calls) stored, although rollback_on_failure
    removed it from the agent in memory. Restored as-is, the next prompt would
    follow it with a second user message in a row, which Bedrock rejects.
    """
    turns: list[list[Message]] = []
    for message in messages:
        if _starts_turn(message) or not turns:
            turns.append([])
        turns[-1].append(message)
    return [message for turn in turns if _finished(turn) for message in turn]
