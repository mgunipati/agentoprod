import logging
import contextlib
import io
import json
import os
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from strands import Agent, tool
from strands.models import Model

import cli
from observability.observability import ConsoleObservationHooks, FieldsInMessageFilter


class FakeModel(Model):
    def update_config(self, **model_config):
        pass

    def get_config(self):
        return {}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError
        yield

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": "private-response"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}
        yield {"metadata": {
            "usage": {"inputTokens": 12, "outputTokens": 4, "totalTokens": 16},
            "metrics": {"latencyMs": 5},
        }}


class ToolModel(FakeModel):
    def __init__(self):
        self.calls = 0

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        self.calls += 1
        if self.calls > 1:
            async for event in super().stream(messages, tool_specs, system_prompt, **kwargs):
                yield event
            return
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {"toolUse": {"name": "echo", "toolUseId": "tool-1"}}}}
        yield {"contentBlockDelta": {"delta": {"toolUse": {"input": '{"text":"private-tool-input"}'}}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "tool_use"}}
        yield {"metadata": {
            "usage": {"inputTokens": 2, "outputTokens": 1, "totalTokens": 3},
            "metrics": {"latencyMs": 1},
        }}


class FailingModel(FakeModel):
    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        raise RuntimeError("private-error-detail")
        yield


class RecordingModel(FakeModel):
    def __init__(self):
        self.roles_seen = []

    async def stream(self, messages, *args, **kwargs):
        self.roles_seen.append([message["role"] for message in messages])
        async for event in super().stream(messages, *args, **kwargs):
            yield event


class FlakyModel(RecordingModel):
    def __init__(self):
        super().__init__()
        self.failed = False

    async def stream(self, messages, *args, **kwargs):
        if not self.failed:
            self.failed = True
            raise RuntimeError("private-error-detail")
        async for event in super().stream(messages, *args, **kwargs):
            yield event


@tool
def echo(text: str) -> str:
    """Return the input text."""
    return text


class ObservabilityTests(unittest.TestCase):
    def run_cli(self, args=(), *, model=None, inputs=None, executor=None):
        """Run cli.main() with a fake model; returns (exit code, stdout, stderr)."""
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(sys, "argv", ["cli.py", *args]))
            stack.enter_context(
                patch("agents.agent_factory.BedrockModel", return_value=model or FakeModel())
            )
            if inputs is not None:
                stack.enter_context(patch("builtins.input", side_effect=inputs))
            if executor is not None:
                stack.enter_context(patch.object(cli, "ThreadPoolExecutor", executor))
            stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            exit_code = cli.main()
        return exit_code, stdout.getvalue(), stderr.getvalue()

    @staticmethod
    def json_records(stderr):
        return [json.loads(line) for line in stderr.splitlines() if line.startswith("{")]

    def test_observe_emits_correlated_metadata_without_content(self):
        code, stdout, stderr = self.run_cli(["--observe", "--prompt", "private-prompt"])

        self.assertEqual(code, 0)
        self.assertIn("private-response", stdout)
        self.assertNotIn("private-prompt", stderr)
        self.assertNotIn("private-response", stderr)
        records = self.json_records(stderr)
        by_message = {record["message"]: record for record in records}
        start = by_message["model_call_started"]
        end = by_message["model_call_completed"]
        completed = by_message["agent_invocation_completed"]
        self.assertEqual(start["request_id"], end["request_id"])
        self.assertEqual(start["request_id"], completed["request_id"])
        self.assertEqual(completed["token_usage"]["inputTokens"], 12)
        self.assertEqual(completed["token_usage"]["outputTokens"], 4)
        self.assertEqual(completed["cycles"], 1)

    def test_observation_is_opt_in(self):
        code, stdout, stderr = self.run_cli(["--prompt", "hello"])
        self.assertEqual(code, 0)
        self.assertIn("private-response", stdout)
        self.assertEqual(stderr, "")

    def test_failed_request_is_dropped_from_history(self):
        model = FlakyModel()
        code, stdout, stderr = self.run_cli(model=model, inputs=["first", "second", "/exit"])

        self.assertEqual(code, 0)
        self.assertIn("The request failed (RuntimeError)", stderr)
        self.assertIn("private-response", stdout)
        self.assertEqual(model.roles_seen, [["user"]])

    def test_ctrl_c_cancels_request_and_keeps_session(self):
        class InterruptFirstWait(ThreadPoolExecutor):
            """Simulate Ctrl+C arriving while the first request is running."""

            submitted = 0

            def submit(self, fn, /, *args, **kwargs):
                future = super().submit(fn, *args, **kwargs)
                type(self).submitted += 1
                if type(self).submitted == 1:
                    original = future.result
                    interrupted = []

                    def result(*a, **k):
                        if not interrupted:
                            interrupted.append(True)
                            original()  # let the request finish, then interrupt
                            raise KeyboardInterrupt
                        return original(*a, **k)

                    future.result = result
                return future

        model = RecordingModel()
        code, _, stderr = self.run_cli(
            model=model, inputs=["first", "second", "/exit"], executor=InterruptFirstWait,
        )

        self.assertEqual(code, 0)
        self.assertIn("Request cancelled.", stderr)
        self.assertEqual(model.roles_seen, [["user"], ["user"]])

    def test_cli_overrides_invalid_environment_settings(self):
        with patch.dict(
            os.environ,
            {"BEDROCK_MODEL_ID": "", "AWS_REGION": " ", "BEDROCK_MAX_TOKENS": "invalid"},
            clear=True,
        ):
            code, stdout, stderr = self.run_cli([
                "--model-id", "test-model", "--region", "us-east-1",
                "--max-tokens", "10", "--prompt", "hello",
            ])

        self.assertEqual(code, 0)
        self.assertIn("private-response", stdout)
        self.assertNotIn("Configuration error", stderr)

    def test_tool_hooks_emit_status_without_arguments_or_results(self):
        assistant = Agent(model=ToolModel(), tools=[echo], hooks=[ConsoleObservationHooks()], callback_handler=None)
        with self.assertLogs("agentoprod.observe", level="INFO") as captured:
            result = assistant("hello", invocation_state={"request_id": "request-1"})

        self.assertEqual(result.stop_reason, "end_turn")
        records = {record.getMessage(): record for record in captured.records}
        self.assertEqual(records["tool_call_started"].request_id, "request-1")
        self.assertEqual(records["tool_call_completed"].request_id, "request-1")
        self.assertEqual(records["tool_call_completed"].status, "completed")
        self.assertNotIn("private-tool-input", repr([record.__dict__ for record in captured.records]))

    def test_interactive_usage_is_per_request(self):
        code, _, stderr = self.run_cli(["--observe"], inputs=["first", "second", "/exit"])

        self.assertEqual(code, 0)
        completed = [r for r in self.json_records(stderr) if r["message"] == "agent_invocation_completed"]
        self.assertEqual(len(completed), 2)
        self.assertNotEqual(completed[0]["request_id"], completed[1]["request_id"])
        self.assertEqual([record["token_usage"]["inputTokens"] for record in completed], [12, 12])

    def test_failed_call_reports_error_without_private_detail(self):
        code, _, stderr = self.run_cli(["--observe", "--prompt", "hello"], model=FailingModel())

        self.assertEqual(code, 1)
        self.assertIn("RuntimeError", stderr)
        self.assertNotIn("private-error-detail", stderr)
        records = self.json_records(stderr)
        model_end = next(record for record in records if record["message"] == "model_call_completed")
        failure = next(record for record in records if record["message"] == "agent_invocation_failed")
        self.assertEqual(model_end["status"], "error")
        self.assertEqual(model_end["request_id"], failure["request_id"])


class FieldsInMessageFilterTests(unittest.TestCase):
    def test_appends_extra_fields_to_plain_text_message(self) -> None:
        record = logging.LogRecord("agentoprod.observe", logging.INFO, __file__, 1, "tool_call_started", None, None)
        record.request_id, record.tool_name, record.error_type = "r-1", "science_helper", None
        FieldsInMessageFilter().filter(record)
        self.assertEqual(record.getMessage(), "tool_call_started request_id=r-1 tool_name=science_helper")


if __name__ == "__main__":
    unittest.main()
