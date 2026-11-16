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
it is not persisted between runs. Logs are JSON and intentionally exclude user
prompts and model responses.

## Local observability

Enable Strands lifecycle telemetry on standard error:

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

## Production considerations

This repository provides a CLI baseline, not a complete hosted enterprise
deployment. Before exposing it to users, integrate your organization's identity
and authorization layer, provision and configure Bedrock Guardrails, set up
centralized trace/metric collection and alerting, and define retention and
network policies. Keep tool integrations opt-in and grant each tool only the
permissions it needs.
