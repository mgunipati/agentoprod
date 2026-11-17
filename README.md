# Agendev

A general-purpose conversational assistant built with Strands Agents and Amazon Bedrock.

## Run locally

Install dependencies and run the interactive chat:

```bash
uv sync
uv run python cli.py
```

Send a single prompt:

```bash
uv run python cli.py --prompt "Explain how TLS works"
```

The AWS SDK uses its standard credential provider chain (for example, an IAM role,
AWS profile, or web identity). Do not put AWS credentials in source files or
environment variables. The configured identity needs permission to invoke the
selected Bedrock model.

## Configuration

| Variable / option | Default | Description |
|---|---|---|
| `BEDROCK_MODEL_ID` / `--model-id` | `apac.amazon.nova-lite-v1:0` | Bedrock model or inference profile ID |
| `AWS_REGION` / `AWS_DEFAULT_REGION` / `--region` | `ap-south-1` | AWS region used for the Bedrock request |
| `BEDROCK_MAX_TOKENS` / `--max-tokens` | `1024` | Maximum generated output tokens |
| `AGENT_MAX_INPUT_CHARS` | `20000` | Per-message input limit |
| `BEDROCK_GUARDRAIL_ID` / `BEDROCK_GUARDRAIL_VERSION` | unset | Bedrock Guardrail applied to every model call; set together, numbered version only. The deployed runtime gets both from CDK |

Command-line options take precedence over their corresponding environment variables.
The interactive session keeps a bounded conversation history in process memory;
it is not persisted between runs. A failed or cancelled request (Ctrl+C) is
dropped from that history, so the conversation can continue. Logs are JSON on
standard error, show only warnings and errors unless `--observe` is set, and
intentionally exclude user prompts and model responses.

## Tests

```bash
uv run pytest
```

## Local observability

The local CLI enables Strands lifecycle telemetry on standard error with
`--observe`:

```bash
uv run python cli.py --observe
uv run python cli.py --observe --prompt "Explain how TLS works"
```

Events include a request ID, model and tool lifecycle status, elapsed request
time, stop reason, per-invocation token usage, and event-loop cycle count.
Tool events are emitted only when tools are configured. They include tool names
and IDs but not tool arguments or results. The application does not emit
prompts, responses, or exception messages in its own logs. Other SDK loggers
may produce their own messages; review their configuration before handling
sensitive data. Token usage is reported by the model provider via Strands and
is not a cost or billing calculation. No external collector or dashboard is
configured by this option.

## Financial tools

`app/agendev/tools/finance_tools.py` gives the agent a loan EMI calculator, Yahoo Finance tools
for current price, price history and company financials, and an Alpha Vantage
quote tool. Each tool validates its arguments, returns a dict that includes the
listed currency, and reports fetch failures without exception details. To add
a tool, follow the template in the module docstring and append it to
`FINANCE_TOOLS`.

The Alpha Vantage tool reads `ALPHA_VANTAGE_API_KEY` from the environment
(for example a local, git-ignored `.env`) and reports itself as not configured
when the key is missing. Market data from both providers may be delayed.

In AgentCore Runtime the key comes from AgentCore Identity instead, and it
lives only in Secrets Manager:

1. Store it as a JSON secret, for example `agentoprod/alpha-vantage` with
   `{"api_key": "..."}`.
2. Reference it from the project (the reference, never the value, goes into
   `agentcore/agentcore.json`):

   ```bash
   agentcore add credentials api-key --name alpha-vantage \
     --api-key-secret-reference '{"secretId": "<secret ARN>", "jsonKey": "api_key"}'
   ```

3. Deploy. `agentcore deploy` creates the Identity credential provider with a
   direct API call, so the CI role needs the `agentcore-credentials` IAM policy
   (credential-provider actions on the token vault, read on that one secret).

The runtime receives only the provider name (`CREDENTIAL_ALPHA_VANTAGE_NAME`)
and fetches the key on each call with its workload identity. With the JWT
authorizer, AgentCore issues that workload token from the caller's verified
access token, so no `--user-id` is needed (see "Sign in with Cognito" below).

The construct only lets runtimes read secrets Identity manages itself, so
`agentcore/cdk/lib/cdk-stack.ts` grants the agent runtime read access to the
external secrets referenced in `agentcore.json`.

## Stock report graph

`app/agendev/agents/report_graph.py` builds a Strands Graph with a fixed flow:
`fetch` (the only node with the Yahoo Finance tools) → `analyse` (compares using
only the fetched data) → `review` (checks the numbers, removes recommendations
and writes the final report). Unlike the chat agent, where the model chooses
which tool or tutor to call, every report runs the same three steps. Run it once
from the CLI:

```bash
uv run python cli.py --report "Compare RELIANCE.NS and TCS.NS"
```

## Research swarm

`app/agendev/agents/research_swarm.py` builds a Strands Swarm of three agents —
`researcher` (the only one with data tools), `analyst` and `writer` — that pass
work to each other with a `handoff_to_agent` tool. Unlike the report graph, the
agents choose the path: a loan question may go researcher → writer, and the
analyst can send the researcher back for missing data. Limits on handoffs, time
and back-and-forth between two agents stop a swarm that gets stuck. Run it once
from the CLI; `--observe` logs the path the agents took:

```bash
uv run python cli.py --observe --swarm "Compare RELIANCE.NS and TCS.NS"
```

## AgentCore Runtime

`app/agendev/main.py` wraps the same agent in the AgentCore HTTP contract.
The shared modules live in packages under `app/agendev/`: `agents/` (chat agent,
report graph, research swarm), `tools/` (finance and tutor tools), `memory/` and
`observability/`. `app/agendev` is the import root (`from tools.finance_tools
import ...`) for the runtime, the local CLI and the tests alike. The runtime always enables
the lifecycle hooks and logs model/tool events with a per-invocation request ID;
events omit prompts, responses, tool arguments, and tool results. AgentCore's
application logger also records invocation completion and failure metadata.
Add any new dependency to both `pyproject.toml` files. Each runtime session
runs in its own microVM, so conversation history is kept in memory per
session ID and lost when the session ends. `agentcore/` holds the AgentCore CLI
project and its CDK stack.

```bash
(cd agentcore/cdk && npm install)
agentcore dev --agent agendev               # local server on :8080
agentcore invoke --local --runtime agendev \
  --session-id session-0000000000000000000000000000001 \
  --payload '{"prompt": "Hello"}'
agentcore deploy --region ap-south-1     # creates AWS resources
agentcore invoke --runtime agendev --bearer-token "$TOKEN" --session-id ... --payload '{"prompt": "Hello"}'
```

An optional `mode` in the payload picks the multi-agent pattern. `chat` (the
default) is the session's agent with its tools and tutors. `report` runs the
stock report Graph and `swarm` runs the research Swarm; both build new agents
for each request and keep no history. Swarm responses include the `path` the
agents took.

```bash
agentcore invoke --runtime agendev --bearer-token "$TOKEN" --payload '{"prompt": "Compare RELIANCE.NS and TCS.NS", "mode": "report"}'
agentcore invoke --runtime agendev --bearer-token "$TOKEN" --payload '{"prompt": "Compare RELIANCE.NS and TCS.NS", "mode": "swarm"}'
```

### Sign in with Cognito

The `agendev` runtime uses a `CUSTOM_JWT` authorizer: AgentCore accepts only
requests with a valid access token from the `agentoprod-users` Cognito user
pool, issued to the `agentoprod-cli` app client, and rejects everything else
before it reaches the container. It forwards the `Authorization` header (the
only header on `requestHeaderAllowlist`), and `main.py` takes the token's
`sub` as the user for both Memory and Identity. A `user_id` in the payload is
ignored when a token is present; it only applies to unauthenticated local runs
such as `agentcore dev`.

Get an access token (valid for one hour) without the password appearing in
history or the process list:

```bash
read -s "PW?Cognito password: " && echo && TOKEN=$(aws cognito-idp initiate-auth --region ap-south-1 \
  --client-id 73o0cfns4dge1l2dovtqmt009h --auth-flow USER_PASSWORD_AUTH \
  --auth-parameters "$(PW="$PW" python3 -c 'import json, os; print(json.dumps({"USERNAME": "mahi", "PASSWORD": os.environ["PW"]}))')" \
  --query AuthenticationResult.AccessToken --output text); unset PW
```

The `financetools` MCP runtime and the `tools` Gateway still use IAM (SigV4).

## AgentCore Memory

The `chat` memory in `agentcore/agentcore.json` gives the deployed runtime
`MEMORY_CHAT_ID`. With it set, `app/agendev/memory/memory_session.py` attaches an
AgentCore Memory session manager to the chat agent:

- **Short-term:** every message is saved as an event for the user and session,
  and reloaded when a new microVM picks up the same session. Turns that failed
  before getting a reply are dropped on reload.
- **Long-term:** AgentCore extracts facts (`/users/{actorId}/facts`) and
  preferences (`/users/{actorId}/preferences`) in the background. Matching
  records are added to each user message inside `<user_context>` tags, so the
  agent remembers a user across sessions. Only records scoring 0.5 or more for
  the question are added (unrelated ones score about 0.35), and the system
  prompt says the context is background: it never triggers a tool call, and
  remembered numbers are fetched again rather than reused.

The user is the `sub` of the caller's Cognito access token (see "Sign in with
Cognito"); only unauthenticated local runs fall back to the payload's
`user_id`. Raw events expire after 30 days. Without
`MEMORY_CHAT_ID`, as in the local CLI, history stays in process memory.

```bash
agentcore invoke --runtime agendev --bearer-token "$TOKEN" --session-id ... --payload '{"prompt": "My home loan is 10 lakh at 9%"}'
```

## Finance tools as an MCP server

The `financetools` runtime (`app/mcp-finance/main.py`) serves the five finance
tools over MCP (streamable HTTP, `POST /mcp` on port 8000), so other agents and
clients can use them. The chat agent still calls them directly in Python.
`app/mcp-finance/finance_tools.py` is a symlink to
`app/agendev/tools/finance_tools.py`, so both runtimes ship the same code; the CodeZip
bundle contains the real file.

The `tools` Gateway (inbound `AWS_IAM`, no protocol type) forwards to the
runtime with its own IAM role. It routes by target name, so the MCP endpoint is
`<gateway URL>/finance/invocations`; MCP requests need the
`Accept: application/json, text/event-stream` header. An MCP-protocol Gateway
would merge tool catalogues across targets, but its MCP targets only support
OAuth or no outbound auth, not IAM.

```bash
# Through the Gateway (the CLI signs the request with your IAM credentials)
agentcore invoke --gateway tools --path /finance/invocations \
  --accept "application/json, text/event-stream" \
  --payload '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'

# The runtime directly: for an MCP runtime the payload is the whole JSON-RPC message
agentcore invoke --runtime financetools \
  --payload '{"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "loan_emi", "arguments": {"principal": 1000000, "annual_rate_percent": 9, "months": 60}}}'

# With the MCP Python SDK: SigV4-signed streamable HTTP through the Gateway
uv run python examples/finance_mcp_client.py
uv run python examples/finance_mcp_client.py get_stock_price '{"symbol": "TCS.NS"}'

# Locally: serves http://localhost:8000/mcp
cd app/mcp-finance && uv run python main.py
```

## Guardrails

`app/guardrails/finance-guardrail.json` defines a Bedrock Guardrail that CDK
creates on deploy: content filters, prompt-attack detection, a denied topic for
personal buy/sell/hold advice, and PII rules (card and bank account numbers
blocked; email, phone and PAN masked). A numbered version is pinned, and the
agent runtime gets `BEDROCK_GUARDRAIL_ID` and `BEDROCK_GUARDRAIL_VERSION`, which
`agents/agent_factory.py` passes on every Bedrock call (mode `guardrailConfig`).
The runtime's IAM role denies model calls that don't carry this Guardrail. When
it steps in, the chat response has `"stop_reason": "guardrail_intervened"`.
See [app/guardrails/README.md](app/guardrails/README.md) for the policy table,
tuning, and the check script:

```bash
uv run python app/guardrails/check_guardrail.py
```

## Monitoring

`agentcore/cdk/lib/monitoring.ts` adds a CloudWatch dashboard (`agentoprod`) and
alarms that notify the SNS topic `agentoprod-alarms`, both on alarm and on recovery:

| Alarm | Fires when |
|---|---|
| `agentoprod-AgentErrors` | an invocation fails: `agent_invocation_failed` logged, or an AgentCore system error |
| `agentoprod-AgentSlow` | p90 latency above 120 s over 15 minutes |
| `agentoprod-Throttled` | the runtime or the Bedrock model throttles a request |
| `agentoprod-GuardrailBlockRate` | over 30% of Guardrail checks blocked or masked in an hour (10+ checks) |
| `agentoprod-FinancetoolsErrors` | the MCP runtime returns a system error |

Runtime, model and Guardrail figures are AWS's own metrics. Failures and busy
replies come from metric filters on `main.py`'s log lines (namespace
`agentoprod`); they match only the JSON copy of each line, since AgentCore
writes every line three times. No data counts as OK, so quiet periods never alarm.

Subscribe your email once (confirm the link AWS sends); it stays out of the repo:

```bash
aws sns subscribe --region ap-south-1 --protocol email --notification-endpoint you@example.com \
  --topic-arn "arn:aws:sns:ap-south-1:$(aws sts get-caller-identity --query Account --output text):agentoprod-alarms"
```

## Production considerations

The agent is deployed on AgentCore through CI/CD, with Memory, Identity, an
MCP tool server and a Gateway. `agentcore/cdk/lib/cdk-stack.ts` adds what the
generated stack leaves out: the agent runtime alone may read the external
Alpha Vantage secret, the tool-only `financetools` runtime is denied Memory,
model and API key access (the construct grants every runtime access to every
project resource), every model call from the agent runtime must carry the
finance Guardrail, runtime logs are kept for 30 days instead of forever, and
alarms and a dashboard watch errors, latency, throttling and Guardrail blocks.

Still open before real users:

- **Sign-in for real users:** the agent runtime accepts Cognito tokens, but the
  user pool has one admin-created test user and a password-flow client. A web
  app needs hosted sign-in (authorization code + PKCE), self sign-up rules and
  MFA. The Gateway and `financetools` still accept IAM callers only.
- **One way in to the tools:** `financetools` can still be called directly by
  anyone with IAM permission to invoke it; a resource policy can limit it to
  the Gateway's role.
- **Evaluation:** a fixed prompt set that checks tool routing, Guardrail
  decisions and invented figures after each change.
