import os
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

from strands.hooks import AgentInitializedEvent

from agents import agent_factory
from agents.agent_factory import AgentSettings, create_agent
from memory.memory_session import (
    MEMORY_ID_ENV,
    RETRIEVAL,
    drop_unfinished_turns,
    memory_session_manager,
    valid_memory_id,
)


def user(text):
    return {"role": "user", "content": [{"text": text}]}


def assistant(text):
    return {"role": "assistant", "content": [{"text": text}]}


def tool_use():
    return {"role": "assistant", "content": [{"toolUse": {"toolUseId": "t1", "name": "loan_emi", "input": {}}}]}


def tool_result():
    return {"role": "user", "content": [{"toolResult": {"toolUseId": "t1", "content": [{"text": "{}"}]}}]}


class MemorySessionManagerTests(unittest.TestCase):
    def test_memory_is_off_without_a_memory_id(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(memory_session_manager("session-1", "user-1", "ap-south-1"))

    def test_builds_session_manager_for_actor_and_session(self) -> None:
        # bedrock_agentcore is only installed in the runtime, so stand in for it.
        config_module = types.ModuleType("config")
        config_module.AgentCoreMemoryConfig = MagicMock(name="AgentCoreMemoryConfig")
        config_module.RetrievalConfig = MagicMock(name="RetrievalConfig")
        manager_module = types.ModuleType("session_manager")
        manager_module.AgentCoreMemorySessionManager = MagicMock(name="AgentCoreMemorySessionManager")
        fake_modules = {
            "bedrock_agentcore.memory.integrations.strands.config": config_module,
            "bedrock_agentcore.memory.integrations.strands.session_manager": manager_module,
        }

        with patch.dict(os.environ, {MEMORY_ID_ENV: "chat-abc123"}), patch.dict(sys.modules, fake_modules):
            manager = memory_session_manager("session-1", "user-1", "ap-south-1")

        kwargs = config_module.AgentCoreMemoryConfig.call_args.kwargs
        self.assertEqual(kwargs["memory_id"], "chat-abc123")
        self.assertEqual(kwargs["session_id"], "session-1")
        self.assertEqual(kwargs["actor_id"], "user-1")
        self.assertEqual(set(kwargs["retrieval_config"]), set(RETRIEVAL))
        manager_module.AgentCoreMemorySessionManager.assert_called_once_with(
            config_module.AgentCoreMemoryConfig.return_value, region_name="ap-south-1"
        )
        self.assertIs(manager, manager_module.AgentCoreMemorySessionManager.return_value)

    def test_only_memories_about_the_question_are_retrieved(self) -> None:
        # Unrelated records score ~0.35 and get acted on as if requested; related ones ~0.65+.
        for options in RETRIEVAL.values():
            self.assertGreaterEqual(options["relevance_score"], 0.5)

    def test_valid_memory_id(self) -> None:
        self.assertTrue(valid_memory_id("user-42_a"))
        self.assertTrue(valid_memory_id("6f1c2a3b-0000-4000-8000-000000000000"))
        for bad in ("", "-starts-with-dash", "has space", "a/b", "x" * 101, None, 42):
            self.assertFalse(valid_memory_id(bad), bad)


class DropUnfinishedTurnsTests(unittest.TestCase):
    def test_keeps_finished_turns_including_tool_calls(self) -> None:
        history = [user("EMI?"), tool_use(), tool_result(), assistant("20,758"), user("Thanks"), assistant("Welcome")]
        self.assertEqual(drop_unfinished_turns(history), history)

    def test_drops_trailing_user_message_from_a_failed_turn(self) -> None:
        history = [user("hi"), assistant("hello"), user("failed turn")]
        self.assertEqual(drop_unfinished_turns(history), [user("hi"), assistant("hello")])

    def test_drops_failed_turn_in_the_middle(self) -> None:
        history = [user("failed"), user("retry"), assistant("answer")]
        self.assertEqual(drop_unfinished_turns(history), [user("retry"), assistant("answer")])

    def test_drops_turn_that_stopped_during_a_tool_call(self) -> None:
        history = [user("hi"), assistant("hello"), user("EMI?"), tool_use(), tool_result()]
        self.assertEqual(drop_unfinished_turns(history), [user("hi"), assistant("hello")])


class RestoredHistory:
    """Stand-in session manager that restores a saved history when the agent is created."""

    def __init__(self, messages):
        self.messages = messages

    def register_hooks(self, registry, **kwargs):
        registry.add_callback(AgentInitializedEvent, self.restore)

    def restore(self, event):
        event.agent.messages = list(self.messages)


class CreateAgentWithMemoryTests(unittest.TestCase):
    def test_restored_history_is_cleaned(self) -> None:
        saved = [user("My loan is 10 lakh"), assistant("Noted."), user("this turn failed")]
        with patch.object(agent_factory, "memory_session_manager", return_value=RestoredHistory(saved)) as factory:
            agent = create_agent(AgentSettings(), session_id="session-1", actor_id="user-1")

        factory.assert_called_once_with("session-1", "user-1", "ap-south-1")
        self.assertEqual(agent.messages, [user("My loan is 10 lakh"), assistant("Noted.")])

    def test_no_memory_without_session_and_actor(self) -> None:
        with patch.object(agent_factory, "memory_session_manager") as factory:
            create_agent(AgentSettings())
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
