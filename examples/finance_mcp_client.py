"""Call the finance MCP server through the AgentCore Gateway with the official MCP client.

The Gateway accepts only IAM callers and forwards /finance/invocations to the
financetools MCP runtime. This client signs every HTTP request with your AWS
credentials (SigV4, service "bedrock-agentcore") and speaks MCP over streamable
HTTP, so it lists and calls the tools like any other MCP client would.

    uv run python examples/finance_mcp_client.py
    uv run python examples/finance_mcp_client.py get_stock_price '{"symbol": "TCS.NS"}'

Set GATEWAY_URL to skip looking the Gateway up by name.
"""

import asyncio
import json
import os
import sys
from collections.abc import Generator

import boto3
import httpx2
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from mcp import Client
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

REGION = os.getenv("AWS_REGION", "ap-south-1")
GATEWAY_NAME = "agendev-default-tools"
TARGET = "finance"  # the Gateway target in agentcore/agentcore.json


class SigV4(httpx2.Auth):
    """Sign each request the MCP client sends, the way the Gateway's AWS_IAM authorizer expects."""

    requires_request_body = True  # the signature covers the JSON-RPC body

    def __init__(self, region: str) -> None:
        self.credentials = boto3.Session().get_credentials()
        self.region = region

    def auth_flow(self, request: httpx2.Request) -> Generator[httpx2.Request, httpx2.Response, None]:
        signed = AWSRequest(method=request.method, url=str(request.url), data=request.content, headers=dict(request.headers))
        SigV4Auth(self.credentials.get_frozen_credentials(), "bedrock-agentcore", self.region).add_auth(signed)
        request.headers.update(dict(signed.headers))
        yield request


def gateway_url() -> str:
    """The Gateway's base URL, from GATEWAY_URL or looked up by name."""
    if url := os.getenv("GATEWAY_URL"):
        return url.rstrip("/")
    control = boto3.client("bedrock-agentcore-control", region_name=REGION)
    gateway = next((g for g in control.list_gateways()["items"] if g["name"] == GATEWAY_NAME), None)
    if gateway is None:
        sys.exit(f"No Gateway named {GATEWAY_NAME!r} in {REGION}.")
    return control.get_gateway(gatewayIdentifier=gateway["gatewayId"])["gatewayUrl"].rstrip("/")


async def main() -> None:
    # A Gateway without a protocol type forwards by path: /<target>/invocations.
    url = f"{gateway_url()}/{TARGET}/invocations"
    http_client = create_mcp_http_client(auth=SigV4(REGION), timeout=httpx2.Timeout(60.0))

    async with Client(streamable_http_client(url, http_client=http_client)) as client:
        tools = await client.list_tools()
        print("Tools:", ", ".join(tool.name for tool in tools.tools))

        name = sys.argv[1] if len(sys.argv) > 1 else "loan_emi"
        arguments = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {
            "principal": 1_000_000, "annual_rate_percent": 9, "months": 60,
        }
        result = await client.call_tool(name, arguments)
        # The tools return a dict; MCP carries it as JSON text.
        print(f"{name}:", json.dumps(json.loads(result.content[0].text), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
