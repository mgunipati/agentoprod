# Bedrock Guardrail for the finance agent

`finance-guardrail.json` is the Guardrail definition. The CDK stack
(`agentcore/cdk/lib/cdk-stack.ts`, `addGuardrail`) reads it on every deploy, so
editing this file and pushing is how you change the policy.

## What it does

| Policy | Setting |
|--------|---------|
| Content filters | Hate and sexual HIGH; violence, insults and misconduct MEDIUM (input and output) |
| Prompt attack | HIGH on input (jailbreaks, "ignore previous instructions") |
| Denied topic `PersonalInvestmentAdvice` | Blocks "should I buy/sell/hold X", "where should I put my money". Prices, history, financials, EMI and factual comparisons still work |
| PII, blocked | Card number, CVV, PIN, US and international bank account numbers |
| PII, masked | Email, phone, Indian PAN (regex) |
| Messages | `blockedInputMessaging` and `blockedOutputsMessaging` explain what the agent can do instead |

## How it is applied

- **Mode: `guardrailConfig`.** `agents/agent_factory.py` passes the Guardrail on
  every Bedrock call (chat agent, tutors, report graph, swarm). Bedrock checks the
  user's message, tool results and the model's answer; the system prompt is not checked.
- **Pinned version.** CDK creates a numbered version whose logical ID contains a hash
  of this file, so a change creates a new version and the runtime switches to it in
  the same deploy. The runtime gets `BEDROCK_GUARDRAIL_ID` and `BEDROCK_GUARDRAIL_VERSION`.
- **Enforced in IAM.** The agent runtime's role denies `InvokeModel` and
  `InvokeModelWithResponseStream` unless the request carries this Guardrail
  (`bedrock:GuardrailIdentifier`), so a model call that forgets it fails instead of
  running unguarded.
- **Trace disabled**, so blocked text (which can be the PII caught) is not returned.
- When the Guardrail steps in, the chat response has `"stop_reason": "guardrail_intervened"`
  and the runtime log line `agent_invocation_completed` records it.

Masking only changes what Bedrock returns. The runtime logs do not contain prompts
or answers, but anything that does (Bedrock model invocation logging, if you turn
it on) would still hold the original text.

## Checking it

After a deploy, run the sample prompts through the deployed Guardrail (no model is
called; it prints the action and the policy that fired, never the matched text):

```bash
uv run python app/guardrails/check_guardrail.py
```

To use the Guardrail with the local CLI, set the same variables:

```bash
BEDROCK_GUARDRAIL_ID=<guardrail ARN> BEDROCK_GUARDRAIL_VERSION=<n> uv run python cli.py
```

## Tuning

- Something legitimate is blocked → lower that filter's strength, or tighten the
  topic definition and examples. Rerun `check_guardrail.py` after the deploy.
- Something harmful gets through → raise the strength or add an example.
- Limits Bedrock enforces (checked by `tests/test_guardrail_definition.py`): messages
  up to 500 characters, topic definitions up to 200, up to 5 examples of 100 each,
  and `PROMPT_ATTACK` must have `outputStrength` `NONE`.
