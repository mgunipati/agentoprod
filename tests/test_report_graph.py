import unittest

from strands.models import Model
from strands.multiagent.base import Status
from strands.multiagent.graph import GraphResult

from app.agendev.report_graph import create_report_graph, report_text

REPLIES = {
    "You collect market data": "FETCHED: price 100 INR",
    "You are a financial analyst": "ANALYSED: grew 10%",
    "You review a stock analysis": "FINAL REPORT. This is general information, not investment advice.",
}


class ScriptedModel(Model):
    """Answers by node, chosen from the system prompt, and records what each node saw."""

    def __init__(self, fail_on: str | None = None):
        self.fail_on = fail_on
        self.seen: dict[str, str] = {}

    def update_config(self, **model_config):
        pass

    def get_config(self):
        return {}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError
        yield

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        prefix = next(p for p in REPLIES if system_prompt and system_prompt.startswith(p))
        self.seen[prefix] = " ".join(
            block.get("text", "") for message in messages for block in message["content"]
        )
        if prefix == self.fail_on:
            raise RuntimeError("model failed")
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockDelta": {"delta": {"text": REPLIES[prefix]}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}


class ReportGraphTests(unittest.TestCase):
    def test_graph_is_fetch_then_analyse_then_review(self) -> None:
        graph = create_report_graph(ScriptedModel())

        self.assertEqual(set(graph.nodes), {"fetch", "analyse", "review"})
        self.assertEqual({node.node_id for node in graph.entry_points}, {"fetch"})
        self.assertEqual(
            {(edge.from_node.node_id, edge.to_node.node_id) for edge in graph.edges},
            {("fetch", "analyse"), ("analyse", "review")},
        )

    def test_only_fetch_has_data_tools(self) -> None:
        graph = create_report_graph(ScriptedModel())

        self.assertEqual(
            sorted(graph.nodes["fetch"].executor.tool_names),
            ["get_company_financials", "get_stock_history", "get_stock_price"],
        )
        self.assertEqual(graph.nodes["analyse"].executor.tool_names, [])
        self.assertEqual(graph.nodes["review"].executor.tool_names, [])

    def test_runs_in_order_and_passes_outputs_forward(self) -> None:
        model = ScriptedModel()

        result = create_report_graph(model)("Compare RELIANCE.NS and TCS.NS")

        self.assertEqual([node.node_id for node in result.execution_order], ["fetch", "analyse", "review"])
        self.assertIn("FETCHED", model.seen["You are a financial analyst"])
        self.assertIn("ANALYSED", model.seen["You review a stock analysis"])
        self.assertEqual(report_text(result), REPLIES["You review a stock analysis"])

    def test_failed_node_stops_the_pipeline(self) -> None:
        model = ScriptedModel(fail_on="You are a financial analyst")

        with self.assertRaisesRegex(RuntimeError, "model failed"):
            create_report_graph(model)("Compare RELIANCE.NS and TCS.NS")
        self.assertNotIn("You review a stock analysis", model.seen)  # review never ran

    def test_report_text_rejects_an_incomplete_result(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            report_text(GraphResult(status=Status.FAILED))


if __name__ == "__main__":
    unittest.main()
