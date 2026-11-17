"""Score one recorded agent answer with AgentCore Evaluations.

Give it the trace ID of one invocation (from the runtime log line
`agent_invocation_completed`, field trace_id) and what the answer should be.
It reads that trace's spans from CloudWatch, the same telemetry AgentCore
records for every call, and runs AgentCore's built-in evaluators on it:

- Builtin.Correctness      the answer against your expected answer (LLM judge)
- Builtin.Faithfulness     the answer is supported by the tool results it got
- Builtin.Helpfulness      useful from the user's point of view
- Builtin.ToolSelectionAccuracy   each tool call was the right tool
- Builtin.TrajectoryExactOrderMatch  exactly the expected tools, in order
- Builtin.GoalSuccessRate  the session met the goal, checked against your assertions

    uv run python evals/evaluate_trace.py emi <trace-id>

Cases live in CASES below; add one per question you want to keep checking.
"""

import json
import sys
import time

import boto3

REGION = "ap-south-1"
RUNTIME_NAME = "agendev_default_agendev"
SERVICE_NAME = f"{RUNTIME_NAME}.DEFAULT"
LOOKBACK_SECONDS = 7 * 24 * 3600

CASES = {
    "emi": {
        "question": "What is the EMI on 10 lakh at 9% for 5 years?",
        "expected": "The monthly EMI is ₹20,758.36. Total interest is ₹2,45,501.31 and the total paid is ₹12,45,501.31.",
        "tools": ["loan_emi"],
        "assertions": [
            "The answer states a monthly EMI of about ₹20,758.",
            "The EMI comes from the loan_emi tool, not from the model's own arithmetic.",
            "The answer does not include stock prices or quotes, which the user did not ask for.",
        ],
    },
}

EVALUATORS = [
    "Builtin.Correctness",
    "Builtin.Faithfulness",
    "Builtin.Helpfulness",
    "Builtin.ToolSelectionAccuracy",
    "Builtin.TrajectoryExactOrderMatch",
    "Builtin.GoalSuccessRate",
]


def runtime_log_group(control) -> str:
    runtime = next(r for r in control.list_agent_runtimes()["agentRuntimes"] if r["agentRuntimeName"] == RUNTIME_NAME)
    return f"/aws/bedrock-agentcore/runtimes/{runtime['agentRuntimeId']}-DEFAULT"


def trace_spans(logs, log_group: str, trace_id: str) -> tuple[str, list[dict]]:
    """The session ID and every telemetry record of one trace, as AgentCore Evaluations expects them."""
    query = f"""fields @message, attributes.session.id as sessionId
        | filter resource.attributes.service.name in ['{SERVICE_NAME}']
        | filter traceId = '{trace_id}'
        | sort @timestamp asc
        | limit 10000"""
    now = int(time.time())
    query_id = logs.start_query(
        logGroupNames=[log_group, "aws/spans"], startTime=now - LOOKBACK_SECONDS, endTime=now, queryString=query,
    )["queryId"]
    while (response := logs.get_query_results(queryId=query_id))["status"] in ("Scheduled", "Running"):
        time.sleep(1)
    if response["status"] != "Complete":
        sys.exit(f"CloudWatch query {response['status']}")

    session_id, docs = None, []
    for row in response["results"]:
        fields = {f["field"]: f["value"] for f in row}
        if "@message" in fields and fields.get("sessionId"):
            session_id = fields["sessionId"]
            docs.append(json.loads(fields["@message"]))
    if not any("kind" in doc for doc in docs):
        sys.exit(f"No spans found for trace {trace_id} in the last 7 days.")
    return session_id, docs


def tool_call_span_ids(docs: list[dict]) -> list[str]:
    return [
        doc["spanId"] for doc in docs
        if "kind" in doc and doc.get("attributes", {}).get("gen_ai.operation.name") == "execute_tool"
    ]


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in CASES:
        sys.exit(f"usage: evaluate_trace.py <{'|'.join(CASES)}> <trace-id>")
    case, trace_id = CASES[sys.argv[1]], sys.argv[2]
    control = boto3.client("bedrock-agentcore-control", region_name=REGION)
    data = boto3.client("bedrock-agentcore", region_name=REGION)

    session_id, docs = trace_spans(boto3.client("logs", region_name=REGION), runtime_log_group(control), trace_id)
    tool_spans = tool_call_span_ids(docs)
    print(f"Trace {trace_id}: session {session_id}, {len(docs)} records, {len(tool_spans)} tool calls\n")

    reference = {
        "context": {"spanContext": {"sessionId": session_id, "traceId": trace_id}},
        "expectedResponse": {"text": case["expected"]},
    }
    session_reference = {
        "context": {"spanContext": {"sessionId": session_id}},
        "assertions": [{"text": text} for text in case["assertions"]],
        "expectedTrajectory": {"toolNames": case["tools"]},
    }
    levels = {e["evaluatorId"]: e.get("level") for e in control.list_evaluators()["evaluators"]}

    for evaluator in EVALUATORS:
        level = levels.get(evaluator, "SESSION")
        request = {"evaluatorId": evaluator, "evaluationInput": {"sessionSpans": docs}}
        if level == "TRACE":
            request |= {"evaluationTarget": {"traceIds": [trace_id]}, "evaluationReferenceInputs": [reference]}
        elif level == "TOOL_CALL":
            if not tool_spans:
                print(f"{evaluator:36} skipped: no tool calls")
                continue
            request["evaluationTarget"] = {"spanIds": tool_spans}
        else:
            request["evaluationReferenceInputs"] = [session_reference]

        for result in data.evaluate(**request)["evaluationResults"]:
            if result.get("errorMessage"):
                print(f"{evaluator:36} error: {result['errorMessage'][:200]}")
                continue
            score = f"{result.get('value', 0):.2f}" if result.get("value") is not None else "-"
            print(f"{evaluator:36} {score:>5}  {result.get('label', '')}")
            print(f"    {(result.get('explanation') or '').strip()[:400]}")


if __name__ == "__main__":
    main()
