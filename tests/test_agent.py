import os
import unittest
from datetime import date
from unittest.mock import patch

from agents.agent_factory import (
    AgentSettings,
    blocked_by_guardrail,
    create_agent,
    create_model,
    create_multi_agent_model,
    prompt_error,
    reply_text,
    rollback_on_failure,
)
from tools.finance_tools import FINANCE_TOOLS
from strands import Agent
from strands.types.exceptions import ConcurrencyException


class AgentSettingsTests(unittest.TestCase):
    def test_reads_bedrock_and_region_settings(self) -> None:
        with patch.dict(
            os.environ,
            {
                "BEDROCK_MODEL_ID": "us.anthropic.claude-sonnet-4-6",
                "AWS_REGION": "us-east-1",
                "BEDROCK_MAX_TOKENS": "2048",
                "AGENT_MAX_INPUT_CHARS": "12000",
            },
            clear=True,
        ):
            settings = AgentSettings.from_env()

        self.assertEqual(settings.model_id, "us.anthropic.claude-sonnet-4-6")
        self.assertEqual(settings.region, "us-east-1")
        self.assertEqual(settings.max_tokens, 2048)
        self.assertEqual(settings.max_input_chars, 12000)

    def test_rejects_non_positive_limits(self) -> None:
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            AgentSettings(max_tokens=0).validate()

        with self.assertRaisesRegex(ValueError, "greater than zero"):
            AgentSettings(max_input_chars=0).validate()

    def test_overrides_replace_invalid_environment_values_before_validation(self) -> None:
        with patch.dict(
            os.environ,
            {"BEDROCK_MODEL_ID": "", "AWS_REGION": " ", "BEDROCK_MAX_TOKENS": "invalid"},
            clear=True,
        ):
            settings = AgentSettings.from_env(
                model_id=" test-model ", region=" us-east-1 ", max_tokens=50,
            )

        self.assertEqual(settings.model_id, "test-model")
        self.assertEqual(settings.region, "us-east-1")
        self.assertEqual(settings.max_tokens, 50)

    def test_invalid_override_is_not_replaced_by_environment(self) -> None:
        with patch.dict(os.environ, {"BEDROCK_MODEL_ID": "valid-model"}, clear=True):
            with self.assertRaisesRegex(ValueError, "BEDROCK_MODEL_ID"):
                AgentSettings.from_env(model_id=" ")

    def test_invalid_non_overridden_environment_is_rejected(self) -> None:
        with patch.dict(os.environ, {"AGENT_MAX_INPUT_CHARS": "invalid"}, clear=True):
            with self.assertRaisesRegex(ValueError, "must be integers"):
                AgentSettings.from_env(max_tokens=50)

    def test_creates_agent_without_connecting_to_aws(self) -> None:
        agent = create_agent(AgentSettings())
        self.assertIsInstance(agent, Agent)
        self.assertEqual(agent.name, "agentoprod")
        self.assertIn(f"Today's date is {date.today().isoformat()}.", agent.system_prompt)
        self.assertEqual(
            sorted(agent.tool_names),
            sorted([t.tool_name for t in FINANCE_TOOLS] + ["math_tutor", "science_helper", "writing_coach"]),
        )

    def test_each_agent_gets_its_own_tutors(self) -> None:
        # Shared tutor agents would mix conversations between sessions.
        first, second = create_agent(AgentSettings()), create_agent(AgentSettings())
        self.assertIsNot(
            first.tool_registry.registry["math_tutor"],
            second.tool_registry.registry["math_tutor"],
        )

    def test_multi_agent_model_gets_at_least_2048_tokens(self) -> None:
        self.assertEqual(create_multi_agent_model(AgentSettings(max_tokens=1024)).config["max_tokens"], 2048)
        self.assertEqual(create_multi_agent_model(AgentSettings(max_tokens=4096)).config["max_tokens"], 4096)

    def test_guardrail_comes_from_the_environment_and_reaches_every_model(self) -> None:
        arn = "arn:aws:bedrock:ap-south-1:123456789012:guardrail/abc123"
        with patch.dict(os.environ, {"BEDROCK_GUARDRAIL_ID": arn, "BEDROCK_GUARDRAIL_VERSION": "3"}, clear=True):
            settings = AgentSettings.from_env()

        for model in (create_model(settings), create_multi_agent_model(settings)):
            self.assertEqual(model.config["guardrail_id"], arn)
            self.assertEqual(model.config["guardrail_version"], "3")
            # Trace would return the blocked text (possibly the PII caught) in the response.
            self.assertEqual(model.config["guardrail_trace"], "disabled")

    def test_without_a_guardrail_the_model_has_none(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = create_model(AgentSettings.from_env()).config
        self.assertNotIn("guardrail_id", config)

    def test_a_blocked_answer_is_recognised_by_the_block_message(self) -> None:
        message = "I can't share that answer. I can give factual market data."
        with patch.dict(os.environ, {"BEDROCK_GUARDRAIL_BLOCKED_OUTPUT": message}, clear=True):
            settings = AgentSettings.from_env()

        self.assertTrue(blocked_by_guardrail(f"  {message}\n", settings))
        self.assertFalse(blocked_by_guardrail("The EMI is ₹20,758.36.", settings))
        self.assertFalse(blocked_by_guardrail(f"{message} But here is the EMI anyway.", settings))
        # Without a configured message (local CLI) nothing counts as blocked.
        self.assertFalse(blocked_by_guardrail(message, AgentSettings()))

    def test_guardrail_needs_id_and_a_numbered_version(self) -> None:
        with self.assertRaisesRegex(ValueError, "both"):
            AgentSettings(guardrail_id="abc123").validate()
        with self.assertRaisesRegex(ValueError, "both"):
            AgentSettings(guardrail_version="1").validate()
        with self.assertRaisesRegex(ValueError, "not DRAFT"):
            AgentSettings(guardrail_id="abc123", guardrail_version="DRAFT").validate()

    def test_prompt_says_remembered_context_is_not_a_request(self) -> None:
        from agents.agent_factory import SYSTEM_PROMPT

        self.assertIn("Answer only what the user's latest message asks", SYSTEM_PROMPT)
        self.assertIn("Never reuse numbers from <user_context>", SYSTEM_PROMPT)

    def test_reply_text_removes_thinking(self) -> None:
        reply = "<thinking>The user wants\na square.</thinking>\n\n12 squared is 144. <thinking>done</thinking>"
        self.assertEqual(reply_text(reply), "12 squared is 144.")
        self.assertEqual(reply_text("  No reasoning here. "), "No reasoning here.")

    def test_prompt_error(self) -> None:
        settings = AgentSettings(max_input_chars=5)
        self.assertIsNone(prompt_error("hello", settings))
        self.assertIn("non-empty", prompt_error("  ", settings))
        self.assertIn("non-empty", prompt_error(None, settings))
        self.assertIn("limit", prompt_error("too long", settings))


if __name__ == "__main__":
    unittest.main()


class RollbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = Agent(callback_handler=None)
        self.agent.messages = [{"role": "user", "content": [{"text": "hi"}]}]

    def test_failed_turn_restores_history(self) -> None:
        with self.assertRaises(RuntimeError):
            with rollback_on_failure(self.agent):
                self.agent.messages.append({"role": "user", "content": [{"text": "lost"}]})
                raise RuntimeError
        self.assertEqual(len(self.agent.messages), 1)

    def test_concurrent_call_leaves_running_turn_history_alone(self) -> None:
        with self.assertRaises(ConcurrencyException):
            with rollback_on_failure(self.agent):
                # The other, still-running turn adds to the history meanwhile.
                self.agent.messages.append({"role": "assistant", "content": [{"text": "kept"}]})
                raise ConcurrencyException("busy")
        self.assertEqual(len(self.agent.messages), 2)
