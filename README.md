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

`app/agendev/finance_tools.py` gives the agent a loan EMI calculator, Yahoo Finance tools
for current price, price history and company financials, and an Alpha Vantage
quote tool. Each tool validates its arguments, returns a dict that includes the
listed currency, and reports fetch failures without exception details. To add
a tool, follow the template in the module docstring and append it to
`FINANCE_TOOLS`.

The Alpha Vantage tool reads `ALPHA_VANTAGE_API_KEY` from the environment
(for example a local, git-ignored `.env`) and reports itself as not configured
when the key is missing. Market data from both providers may be delayed.

In AgentCore Runtime the key comes from AgentCore Identity instead: add it once
with `agentcore add credentials api-key --name alpha-vantage --api-key -`
(reading the key from standard input) and deploy. The runtime receives only the
provider name (`CREDENTIAL_ALPHA_VANTAGE_NAME`) and fetches the key on each
call, which requires invoking the runtime with a user ID (`--user-id`).

## Stock report graph

`app/agendev/report_graph.py` builds a Strands Graph with a fixed flow:
`fetch` (the only node with the Yahoo Finance tools) → `analyse` (compares using
only the fetched data) → `review` (checks the numbers, removes recommendations
and writes the final report). Unlike the chat agent, where the model chooses
which tool or tutor to call, every report runs the same three steps. Run it once
from the CLI:

```bash
uv run python cli.py --report "Compare RELIANCE.NS and TCS.NS"
```

## Research swarm

`app/agendev/research_swarm.py` builds a Strands Swarm of three agents —
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
The shared agent, finance, and observability modules live under `app/agendev/`
and are imported by both the local CLI and runtime. The runtime always enables
the lifecycle hooks and logs model/tool events with a per-invocation request ID;
events omit prompts, responses, tool arguments, and tool results. AgentCore's
application logger also records invocation completion and failure metadata.
Add any new dependency to both `pyproject.toml` files. Each runtime session
runs in its own microVM, so conversation history is kept in memory per
session ID and lost when the session ends. `agentcore/` holds the AgentCore CLI
project and its CDK stack.

```bash
(cd agentcore/cdk && npm install)
agentcore dev --agent agentoprod            # local server on :8080
agentcore invoke --local --runtime agentoprod \
  --session-id session-0000000000000000000000000000001 \
  --payload '{"prompt": "Hello"}'
agentcore deploy --region ap-south-1     # creates AWS resources
agentcore invoke --runtime agentoprod --session-id ... --payload '{"prompt": "Hello"}'
```

The deployed runtime uses IAM (SigV4) for inbound auth and public networking,
which suits development only.

## Production considerations

This repository provides a CLI baseline, not a complete hosted enterprise
deployment. Before exposing it to users, integrate your organization's identity
and authorization layer, provision and configure Bedrock Guardrails, set up
centralized trace/metric collection and alerting, and define retention and
network policies. Keep tool integrations opt-in and grant each tool only the
permissions it needs.
