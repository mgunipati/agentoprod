"""Tests for the AgentCore entrypoint, app/agendev/main.py.

The runtime installs bedrock-agentcore; CI does not, so a minimal stand-in for
BedrockAgentCoreApp is injected before main.py is imported from its own folder,
the way the runtime runs it.
"""

import asyncio
import base64
import importlib.util
import json
import logging
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from strands.types.exceptions import ConcurrencyException

RUNTIME_DIR = Path(__file__).resolve().parents[1] / "app" / "agendev"
SESSION = "session-0000000000000000000000000000001"
COGNITO_SUB = "4c1f2e8a-0b7d-4e6a-9f3c-2d5b8a1e7c90"


def bearer(claims: dict) -> dict:
    """An Authorization header with an (unsigned) JWT, as AgentCore forwards it after verifying."""
    encode = lambda part: base64.urlsafe_b64encode(json.dumps(part).encode()).rstrip(b"=").decode()
    return {"Authorization": f"Bearer {encode({'alg': 'RS256'})}.{encode(claims)}.signature"}


class FakeApp:
    def __init__(self):
        self.logger = logging.getLogger("test.runtime")

    def entrypoint(self, function):
        return function

    def run(self):
        raise AssertionError("tests must not start the server")


def load_main():
    runtime_module = types.ModuleType("bedrock_agentcore.runtime")
    runtime_module.BedrockAgentCoreApp = FakeApp
    fake_modules = {
        "bedrock_agentcore": types.ModuleType("bedrock_agentcore"),
        "bedrock_agentcore.runtime": runtime_module,
    }
    # Add only the fakes and remove only them afterwards: patch.dict(sys.modules) would also
    # drop everything main.py imported (numpy cannot be imported twice in one process).
    added = [name for name in fake_modules if name not in sys.modules]
    sys.modules.update({name: fake_modules[name] for name in added})
    sys.path.insert(0, str(RUNTIME_DIR))
    try:
        spec = importlib.util.spec_from_file_location("agendev_runtime_main", RUNTIME_DIR / "main.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(RUNTIME_DIR))
        for name in added:
            sys.modules.pop(name, None)


class FakeResult:
    def __init__(self, text, stop_reason="end_turn"):
        self.text, self.stop_reason = text, stop_reason

    def __str__(self):
        return self.text


class RuntimeEntrypointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.main = load_main()

    def setUp(self) -> None:
        self.main.agent_for_session.cache_clear()
        self.context = SimpleNamespace(session_id=SESSION)

    def invoke(self, payload):
        return asyncio.run(self.main.invoke(payload, self.context))

    def chat_agent(self, **invoke_async):
        agent = MagicMock()
        agent.messages = []
        agent.invoke_async = AsyncMock(**invoke_async)
        return agent

    def test_rejects_bad_requests_without_calling_an_agent(self) -> None:
        with patch.object(self.main, "create_agent") as create_agent:
            self.assertIn("prompt", self.invoke({"message": "hi"})["error"])
            self.assertIn("non-empty", self.invoke({"prompt": "  "})["error"])
            self.assertIn("mode must be one of: chat, report, swarm", self.invoke({"prompt": "hi", "mode": "x"})["error"])
            self.assertIn("user_id", self.invoke({"prompt": "hi", "user_id": "bad id!"})["error"])
        create_agent.assert_not_called()

    def test_chat_uses_one_agent_per_session_and_user(self) -> None:
        agent = self.chat_agent(return_value=FakeResult("<thinking>sum</thinking>2 + 2 is 4."))
        with patch.object(self.main, "create_agent", return_value=agent) as create_agent:
            first = self.invoke({"prompt": "What is 2+2?", "user_id": "user-1"})
            self.invoke({"prompt": "And 3+3?", "user_id": "user-1"})

        self.assertEqual(first, {"result": "2 + 2 is 4.", "stop_reason": "end_turn", "mode": "chat"})
        create_agent.assert_called_once()
        self.assertEqual(create_agent.call_args.kwargs["session_id"], SESSION)
        self.assertEqual(create_agent.call_args.kwargs["actor_id"], "user-1")

    def test_session_id_memory_cannot_use_skips_memory(self) -> None:
        self.context = SimpleNamespace(session_id="has spaces, not a memory id")
        with patch.object(self.main, "create_agent", return_value=self.chat_agent(return_value=FakeResult("ok"))) as create_agent:
            self.invoke({"prompt": "hi"})
        self.assertNotIn("session_id", create_agent.call_args.kwargs)

    def test_overlapping_chat_message_gets_busy_reply(self) -> None:
        agent = self.chat_agent(side_effect=ConcurrencyException("busy"))
        with patch.object(self.main, "create_agent", return_value=agent):
            self.assertIn("Still answering", self.invoke({"prompt": "hi"})["error"])

    def test_chat_failure_is_raised_and_turn_rolled_back(self) -> None:
        agent = self.chat_agent(side_effect=RuntimeError("model down"))
        agent.messages = [{"role": "user", "content": [{"text": "earlier"}]}]
        with patch.object(self.main, "create_agent", return_value=agent):
            with self.assertRaises(RuntimeError):
                self.invoke({"prompt": "hi"})
        self.assertEqual(len(agent.messages), 1)

    def test_token_user_is_the_memory_user_and_payload_cannot_override_it(self) -> None:
        self.context = SimpleNamespace(session_id=SESSION, request_headers=bearer({"sub": COGNITO_SUB, "client_id": "abc"}))
        with patch.object(self.main, "create_agent", return_value=self.chat_agent(return_value=FakeResult("ok"))) as create_agent:
            self.invoke({"prompt": "hi", "user_id": "someone-else"})
        self.assertEqual(create_agent.call_args.kwargs["actor_id"], COGNITO_SUB)

    def test_unreadable_token_is_rejected(self) -> None:
        for headers in ({"authorization": "Bearer not-a-jwt"}, bearer({"client_id": "abc"}), {"Authorization": "Basic abc"}):
            self.context = SimpleNamespace(session_id=SESSION, request_headers=headers)
            with patch.object(self.main, "create_agent") as create_agent:
                self.assertIn("Sign in again", self.invoke({"prompt": "hi"})["error"])
            create_agent.assert_not_called()

    def test_without_a_token_the_payload_user_is_used(self) -> None:
        self.assertIsNone(self.main.token_user(None))
        self.assertIsNone(self.main.token_user({"X-Other": "1"}))
        with patch.object(self.main, "create_agent", return_value=self.chat_agent(return_value=FakeResult("ok"))) as create_agent:
            self.invoke({"prompt": "hi", "user_id": "local-dev"})
        self.assertEqual(create_agent.call_args.kwargs["actor_id"], "local-dev")

    def test_report_and_swarm_modes_build_a_new_pattern_per_request(self) -> None:
        pattern = MagicMock()
        pattern.invoke_async = AsyncMock(return_value="result")
        build = MagicMock(return_value=pattern)
        modes = {"report": (build, lambda r: "REPORT"), "swarm": (build, lambda r: "ANSWER")}
        with patch.dict(self.main.ONE_SHOT_MODES, modes), \
             patch.object(self.main, "create_multi_agent_model"), \
             patch.object(self.main, "handoff_path", return_value="researcher → writer"):
            report = self.invoke({"prompt": "Compare A and B", "mode": "report"})
            swarm = self.invoke({"prompt": "Compare A and B", "mode": "swarm"})

        self.assertEqual(report, {"result": "REPORT", "mode": "report"})
        self.assertEqual(swarm, {"result": "ANSWER", "path": "researcher → writer", "mode": "swarm"})
        self.assertEqual(build.call_count, 2)


if __name__ == "__main__":
    unittest.main()
