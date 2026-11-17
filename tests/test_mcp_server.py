import asyncio
import importlib.util
import json
import sys
import unittest
from pathlib import Path

from mcp import Client

from tools.finance_tools import FINANCE_TOOLS, plain_tool_functions

SERVER_DIR = Path(__file__).resolve().parents[1] / "app" / "mcp-finance"


def load_server():
    """Import app/mcp-finance/main.py the way the runtime does: from its own folder."""
    sys.path.insert(0, str(SERVER_DIR))
    try:
        spec = importlib.util.spec_from_file_location("mcp_finance_main", SERVER_DIR / "main.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.mcp
    finally:
        sys.path.remove(str(SERVER_DIR))


async def with_client(action):
    async with Client(load_server()) as client:  # in-process: real MCP, no network
        return await action(client)


class McpServerTests(unittest.TestCase):
    def test_serves_the_same_tools_as_the_agent(self) -> None:
        tools = asyncio.run(with_client(lambda c: c.list_tools())).tools

        self.assertEqual([t.name for t in tools], [t.tool_name for t in FINANCE_TOOLS])
        for tool in tools:
            self.assertTrue(tool.description, tool.name)
        loan = next(t for t in tools if t.name == "loan_emi")
        self.assertEqual(set(loan.input_schema["required"]), {"principal", "annual_rate_percent", "months"})

    def test_calls_a_tool_over_mcp(self) -> None:
        result = asyncio.run(with_client(lambda c: c.call_tool(
            "loan_emi", {"principal": 1_000_000, "annual_rate_percent": 9, "months": 60}
        )))

        self.assertFalse(result.is_error)
        self.assertEqual(json.loads(result.content[0].text)["emi"], 20758.36)

    def test_invalid_input_returns_the_tools_own_error(self) -> None:
        result = asyncio.run(with_client(lambda c: c.call_tool("get_stock_price", {"symbol": "not a ticker!"})))

        self.assertIn("not a valid ticker symbol", json.loads(result.content[0].text)["error"])

    def test_server_ships_the_agents_tools_file(self) -> None:
        # A symlink, so the two runtimes can't drift apart.
        shared = SERVER_DIR / "finance_tools.py"
        self.assertTrue(shared.is_symlink())
        self.assertEqual(shared.resolve(), (SERVER_DIR.parent / "agendev" / "tools" / "finance_tools.py").resolve())

    def test_plain_functions_match_the_strands_tools(self) -> None:
        self.assertEqual([f.__name__ for f in plain_tool_functions()], [t.tool_name for t in FINANCE_TOOLS])


if __name__ == "__main__":
    unittest.main()
