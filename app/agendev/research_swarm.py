"""Research team built as a Strands Swarm: the agents decide who works next.

Compare with report_graph.py, where the order is fixed in code. Here each agent
gets a handoff_to_agent tool and chooses whether to pass the task on, to whom,
and with what message. The swarm ends when an agent answers without handing
off; that agent's reply is the result.

- researcher: the only agent with data tools; starts every task
- analyst:    interprets the data; can send the researcher back for more
- writer:     turns the findings into the final answer for the user

The next agent only sees the user request, the handoff message and shared
context, not the previous agent's tool results, so every prompt tells the agent
to put its findings into the handoff message.
"""

from strands import Agent
from strands.models import Model
from strands.multiagent import Swarm, SwarmResult
from strands.multiagent.base import Status
from strands.multiagent.swarm import SwarmResult

if __package__:
    from .finance_tools import get_company_financials, get_stock_history, get_stock_price, loan_emi
    from .agent_factory import reply_text
else:
    from finance_tools import get_company_financials, get_stock_history, get_stock_price, loan_emi
    from agent_factory import reply_text

RESEARCHER_PROMPT = """You are the researcher in a team of financial agents. Use your
tools to collect the facts the request needs: prices, history, financials or loan
calculations. Do not interpret them. When you have the data, hand off to the
analyst with handoff_to_agent. Write the key numbers you found, with currency
and period, as a short list in the message text, and leave the context argument
empty. If the request needs no data, hand off to the writer."""

ANALYST_PROMPT = """You are the analyst in a team of financial agents. Interpret the
data you were handed: compare, explain trends and point out risks, quoting the
numbers you use. If important data is missing, hand back to the researcher and
say exactly what to fetch. Otherwise hand off to the writer, putting your
analysis and every key number, with its currency, in the message. Keep the
analysis under 200 words. Never give buy, sell or hold advice."""

WRITER_PROMPT = """You are the writer in a team of financial agents. Turn the analysis
you were handed into a clear, short answer for the user. Use only numbers that
appear in the handoff message or shared context; never invent, estimate or use
example figures. If a figure you need is missing, say it was not available. Do
not hand off; your reply is the final answer. End with: "This is general
information, not investment advice." """

# Small limits: three agents rarely need more than a few handoffs, and a stuck
# swarm (agents bouncing work back and forth) should fail fast.
MAX_HANDOFFS = 6
MAX_ITERATIONS = 8
EXECUTION_TIMEOUT_SECONDS = 180.0
NODE_TIMEOUT_SECONDS = 90.0


def create_research_swarm(model: Model) -> SwarmResult:
    """Build a new swarm per request: Swarm adds its handoff tool to these agents."""
    researcher = Agent(
        name="researcher",
        model=model,
        system_prompt=RESEARCHER_PROMPT,
        tools=[get_stock_price, get_stock_history, get_company_financials, loan_emi],
        callback_handler=None,
    )
    analyst = Agent(name="analyst", model=model, system_prompt=ANALYST_PROMPT, callback_handler=None)
    writer = Agent(name="writer", model=model, system_prompt=WRITER_PROMPT, callback_handler=None)
    return Swarm(
        [researcher, analyst, writer],
        entry_point=researcher,
        max_handoffs=MAX_HANDOFFS,
        max_iterations=MAX_ITERATIONS,
        execution_timeout=EXECUTION_TIMEOUT_SECONDS,
        node_timeout=NODE_TIMEOUT_SECONDS,
        # Fail if the last 6 turns only involved 2 agents bouncing work back and
        # forth. One "analyst sends researcher back for more data" loop is fine.
        repetitive_handoff_detection_window=6,
        repetitive_handoff_min_unique_agents=3,
    )


def swarm_answer(result: SwarmResult) -> str:
    """Return the last agent's reply, or raise if the swarm stopped early."""
    if result.status != Status.COMPLETED or not result.node_history:
        path = " → ".join(node.node_id for node in result.node_history) or "none"
        raise RuntimeError(f"Swarm did not complete (status: {result.status.value}, path: {path})")
    last = result.results[result.node_history[-1].node_id]
    return "\n".join(reply_text(agent_result) for agent_result in last.get_agent_results())


def handoff_path(result: SwarmResult) -> str:
    """The order the agents actually ran in, e.g. 'researcher → analyst → writer'."""
    return " → ".join(node.node_id for node in result.node_history)
