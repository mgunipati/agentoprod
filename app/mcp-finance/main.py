"""AgentCore Runtime `financetools` (folder app/mcp-finance): the finance tools as an MCP server.

The chat agent (runtime `agendev`) keeps calling these tools directly in
Python. This server offers the same tools over MCP so other agents and clients
can use them, through the AgentCore Gateway `tools`.

finance_tools.py is a symlink to app/agendev/tools/finance_tools.py, so both
runtimes ship the same code: names, docstrings, validation and error handling.

AgentCore's MCP contract: streamable HTTP on 0.0.0.0:8000 at POST /mcp,
stateless. mcp 2.x renamed FastMCP to MCPServer and moved host/port/stateless
settings to run().
"""

import os

from mcp.server.mcpserver import MCPServer

from finance_tools import plain_tool_functions

mcp = MCPServer(
    "financetools",
    instructions=(
        "Finance tools: loan EMI calculation, stock prices, price history and "
        "company financials. Results include the currency; errors come back as "
        '{"error": ...} without exception details.'
    ),
)

for function in plain_tool_functions():
    # The tool name, description and input schema come from the function itself.
    mcp.add_tool(function)


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=int(os.environ.get("FASTMCP_PORT", "8000")),
        stateless_http=True,
    )
