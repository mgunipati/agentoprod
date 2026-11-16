import unittest

from strands.models import Model
from strands.multiagent.base import Status

from app.agendev.research_swarm import create_research_swarm, handoff_path, swarm_answer

FINAL = "Final answer. This is general information, not investment advice."


class HandoffModel(Model):
    """Plays each agent from a script: hand off to someone, or answer.

    `script` maps an agent's system-prompt prefix to a list of moves used in
    order, one per turn that agent takes: an agent name means "hand off to it",
    anything else is the final text reply.
    """

    PREFIXES = {
        "You are the researcher": "researcher",
        "You are the analyst": "analyst",
        "You are the writer": "writer",
    }

    def __init__(self, script: dict[str, list[str]]):
        self.script = {name: list(moves) for name, moves in script.items()}
        self.inputs: dict[str, list[str]] = {}

    def update_config(self, **model_config):
        pass

    def get_config(self):
        return {}

    async def structured_output(self, output_model, prompt, system_prompt=None, **kwargs):
        raise NotImplementedError
        yield

    async def stream(self, messages, tool_specs=None, system_prompt=None, **kwargs):
        agent = next(name for prefix, name in self.PREFIXES.items() if system_prompt.startswith(prefix))
        last = messages[-1]["content"]
        yield {"messageStart": {"role": "assistant"}}
        if any("toolResult" in block for block in last):
            # After its handoff tool ran, the agent just ends its turn.
            yield {"contentBlockDelta": {"delta": {"text": "Handed off."}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            return

        self.inputs.setdefault(agent, []).append(" ".join(b.get("text", "") for b in last))
        move = self.script[agent].pop(0)
        if move in self.PREFIXES.values():
            yield {"contentBlockStart": {"start": {"toolUse": {"toolUseId": f"t-{agent}-{move}", "name": "handoff_to_agent"}}}}
            yield {"contentBlockDelta": {"delta": {"toolUse": {
                "input": f'{{"agent_name": "{move}", "message": "from {agent}: data 42"}}'
            }}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "tool_use"}}
        else:
            yield {"contentBlockDelta": {"delta": {"text": move}}}
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}


class ResearchSwarmTests(unittest.TestCase):
    def test_team_and_tools(self) -> None:
        swarm = create_research_swarm(HandoffModel({}))

        self.assertEqual(set(swarm.nodes), {"researcher", "analyst", "writer"})
        self.assertEqual(swarm.entry_point.name, "researcher")
        researcher_tools = set(swarm.nodes["researcher"].executor.tool_names)
        self.assertTrue({"get_stock_price", "loan_emi", "handoff_to_agent"} <= researcher_tools)
        self.assertEqual(swarm.nodes["analyst"].executor.tool_names, ["handoff_to_agent"])

    def test_agents_choose_the_path_and_last_reply_is_the_answer(self) -> None:
        model = HandoffModel({"researcher": ["analyst"], "analyst": ["writer"], "writer": [FINAL]})

        result = create_research_swarm(model)("Compare RELIANCE.NS and TCS.NS")

        self.assertEqual(result.status, Status.COMPLETED)
        self.assertEqual(handoff_path(result), "researcher → analyst → writer")
        self.assertEqual(swarm_answer(result), FINAL)
        # The handoff message is how findings reach the next agent.
        self.assertIn("from researcher: data 42", model.inputs["analyst"][0])

    def test_analyst_can_send_researcher_back_once(self) -> None:
        model = HandoffModel({
            "researcher": ["analyst", "analyst"],
            "analyst": ["researcher", "writer"],
            "writer": [FINAL],
        })

        result = create_research_swarm(model)("Compare RELIANCE.NS and TCS.NS")

        self.assertEqual(handoff_path(result), "researcher → analyst → researcher → analyst → writer")
        self.assertEqual(swarm_answer(result), FINAL)

    def test_ping_pong_between_two_agents_fails(self) -> None:
        model = HandoffModel({"researcher": ["analyst"] * 5, "analyst": ["researcher"] * 5, "writer": [FINAL]})

        result = create_research_swarm(model)("Compare RELIANCE.NS and TCS.NS")

        self.assertEqual(result.status, Status.FAILED)
        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            swarm_answer(result)


if __name__ == "__main__":
    unittest.main()
