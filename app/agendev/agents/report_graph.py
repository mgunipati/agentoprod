"""Stock report pipeline built as a Strands Graph: fetch → analyse → review.

Unlike the chat agent, where the model decides which tool or tutor to call,
the flow here is fixed in code. Every report runs the same three steps, which
makes it predictable, easy to test and easy to audit:

- fetch:   the only node with data tools; collects numbers, gives no opinions
- analyse: no tools; compares the companies using only what fetch returned
- review:  no tools; checks the analysis against the data and writes the
           final report

Strands passes each node the original request plus the outputs of the nodes
before it, so analyse sees fetch's data and review sees both.
"""

from strands import Agent
from strands.models import Model
from strands.multiagent import GraphBuilder
from strands.multiagent.base import Status
from strands.multiagent.graph import Graph, GraphResult

from agents.agent_factory import reply_text
from tools.finance_tools import get_company_financials, get_stock_history, get_stock_price

FETCH_PROMPT = """You collect market data for a stock report. Use the tools to get
the current price, price history and company financials for every company in the
request. Report only the facts the tools returned, with each value's currency and
period. Do not analyse, compare or give opinions. If a tool returns an error, say
which data is missing."""

ANALYSE_PROMPT = """You are a financial analyst. Using only the data provided by the
fetch step, compare the companies on price performance, profitability, growth and
leverage. Quote the numbers you rely on with their currency. If a value is missing,
write "data not available"; never estimate or use outside knowledge."""

REVIEW_PROMPT = """You review a stock analysis before it is shown to a user. Check
that every number in the analysis appears in the fetched data, that currencies and
periods are not mixed up, and that there is no buy, sell or hold recommendation.
Remove or correct anything unsupported. Then write the final report: a short
summary, the key figures as a list, and this closing line: "This is general
information, not investment advice." Output only the final report."""

# Each node runs once in a straight line; the limit guards against loops if
# conditional edges are added later.
MAX_NODE_EXECUTIONS = 5


def create_report_graph(model: Model) -> Graph:
    """Build a new graph per report: its agents keep history while it runs."""
    fetcher = Agent(
        name="fetch",
        model=model,
        system_prompt=FETCH_PROMPT,
        tools=[get_stock_price, get_stock_history, get_company_financials],
        callback_handler=None,
    )
    analyst = Agent(name="analyse", model=model, system_prompt=ANALYSE_PROMPT, callback_handler=None)
    reviewer = Agent(name="review", model=model, system_prompt=REVIEW_PROMPT, callback_handler=None)

    builder = GraphBuilder()
    builder.add_node(fetcher, "fetch")
    builder.add_node(analyst, "analyse")
    builder.add_node(reviewer, "review")

    builder.add_edge("fetch", "analyse")
    builder.add_edge("analyse", "review")
    builder.set_entry_point("fetch")
    builder.set_max_node_executions(MAX_NODE_EXECUTIONS)
    return builder.build()


def report_text(result: GraphResult) -> str:
    """Return the review node's final report, or raise if the pipeline failed."""
    review = result.results.get("review")
    if result.status != Status.COMPLETED or review is None or review.status != Status.COMPLETED:
        failed = [node_id for node_id, node in result.results.items() if node.status == Status.FAILED]
        raise RuntimeError(f"Report pipeline did not complete (failed nodes: {failed or 'none'})")
    return "\n".join(reply_text(agent_result) for agent_result in review.get_agent_results())
