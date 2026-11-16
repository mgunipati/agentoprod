import os
import unittest
from datetime import date
from unittest.mock import patch

from agent_factory import AgentSettings, create_agent, prompt_error
from finance_tools import FINANCE_TOOLS
from strands import Agent


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
        self.assertEqual(sorted(agent.tool_names), sorted(t.tool_name for t in FINANCE_TOOLS))

    def test_prompt_error(self) -> None:
        settings = AgentSettings(max_input_chars=5)
        self.assertIsNone(prompt_error("hello", settings))
        self.assertIn("non-empty", prompt_error("  ", settings))
        self.assertIn("non-empty", prompt_error(None, settings))
        self.assertIn("limit", prompt_error("too long", settings))


if __name__ == "__main__":
    unittest.main()
