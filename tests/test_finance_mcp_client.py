"""The example MCP client signs each request for the Gateway's AWS_IAM authorizer."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx2
from botocore.credentials import Credentials

PATH = Path(__file__).resolve().parents[1] / "examples" / "finance_mcp_client.py"
spec = importlib.util.spec_from_file_location("finance_mcp_client", PATH)
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)


class SigV4Tests(unittest.TestCase):
    def test_request_is_signed_for_bedrock_agentcore_including_the_body(self) -> None:
        with patch.object(client.boto3, "Session") as session:
            session.return_value.get_credentials.return_value = Credentials("AKIDEXAMPLE", "secret")
            auth = client.SigV4("ap-south-1")
        request = httpx2.Request(
            "POST", "https://gw.example.com/finance/invocations",
            headers={"Accept": "application/json, text/event-stream"}, content=b'{"jsonrpc": "2.0"}',
        )
        signed = next(auth.auth_flow(request))

        authorization = signed.headers["Authorization"]
        self.assertTrue(authorization.startswith("AWS4-HMAC-SHA256 Credential=AKIDEXAMPLE/"))
        self.assertIn("/ap-south-1/bedrock-agentcore/aws4_request", authorization)
        self.assertIn("X-Amz-Date", signed.headers)
        self.assertTrue(client.SigV4.requires_request_body)


if __name__ == "__main__":
    unittest.main()
