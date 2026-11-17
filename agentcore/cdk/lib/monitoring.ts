import { Duration } from 'aws-cdk-lib';
import * as cloudwatch from 'aws-cdk-lib/aws-cloudwatch';
import * as cwActions from 'aws-cdk-lib/aws-cloudwatch-actions';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as sns from 'aws-cdk-lib/aws-sns';
import { Construct, type IDependable } from 'constructs';

/** A deployed AgentCore runtime, as the monitoring needs it. */
export interface MonitoredRuntime {
  /** Short name used in alarm names and graph labels, e.g. "agendev". */
  readonly label: string;
  readonly runtimeArn: string;
  readonly runtimeName: string;
  readonly logGroupName: string;
  /** Something that guarantees the log group exists (AgentCore only creates it on first start). */
  readonly logGroupReady: IDependable;
}

export interface AgentMonitoringProps {
  /** The runtime that runs the agent (main.py logs agent_invocation_* lines). */
  readonly agent: MonitoredRuntime;
  /** Runtimes that only serve tools; watched for errors only. */
  readonly tools: MonitoredRuntime[];
  readonly guardrail: { readonly arn: string; readonly version: string };
  /** Bedrock model or inference profile the agent calls. */
  readonly modelId: string;
}

/** Namespace for the metrics made from the agent's own log lines. */
const NAMESPACE = 'agentoprod';
const FIVE_MINUTES = Duration.minutes(5);

/**
 * Alarms and a dashboard for the agent. Alarms notify the SNS topic `agentoprod-alarms`
 * (subscribe your email to it once; see the README). Missing data counts as OK, so a quiet
 * period never alarms.
 *
 * Built-in metrics cover the runtime (AWS/Bedrock-AgentCore), the model (AWS/Bedrock) and the
 * Guardrail (AWS/Bedrock/Guardrails). Two metrics come from main.py's log lines: failed
 * invocations and busy replies. AgentCore writes each line three times (OpenTelemetry, JSON
 * and plain text), so the filters match only the JSON form to count each event once.
 */
export class AgentMonitoring extends Construct {
  readonly topic: sns.Topic;
  readonly dashboard: cloudwatch.Dashboard;

  constructor(scope: Construct, id: string, props: AgentMonitoringProps) {
    super(scope, id);
    const { agent, guardrail, modelId } = props;

    this.topic = new sns.Topic(this, 'AlarmTopic', {
      topicName: 'agentoprod-alarms',
      displayName: 'agentoprod alarms',
      enforceSSL: true,
    });

    const failures = this.logMetric(agent, 'AgentFailures', 'agent_invocation_failed');
    const busy = this.logMetric(agent, 'AgentBusyReplies', 'agent_busy');
    const invocations = runtimeMetric(agent, 'Invocations', 'Sum');
    const systemErrors = runtimeMetric(agent, 'SystemErrors', 'Sum');
    const userErrors = runtimeMetric(agent, 'UserErrors', 'Sum');
    const latencyP50 = runtimeMetric(agent, 'Latency', 'p50');
    const latencyP90 = runtimeMetric(agent, 'Latency', 'p90');
    const runtimeThrottles = runtimeMetric(agent, 'Throttles', 'Sum');
    const model = (metricName: string, statistic: string) =>
      new cloudwatch.Metric({ namespace: 'AWS/Bedrock', metricName, dimensionsMap: { ModelId: modelId }, statistic, period: FIVE_MINUTES });
    const modelThrottles = model('InvocationThrottles', 'Sum');
    const guardrailMetric = (metricName: string, period: Duration) =>
      new cloudwatch.Metric({
        namespace: 'AWS/Bedrock/Guardrails',
        metricName,
        dimensionsMap: { GuardrailArn: guardrail.arn, GuardrailVersion: guardrail.version },
        statistic: 'Sum',
        period,
      });

    // Our own failure line and AgentCore's SystemErrors can both count the same failure; take
    // the larger so a crash before main.py logs anything (import error, timeout) still alarms.
    const errors = new cloudwatch.MathExpression({
      expression: 'MAX([FILL(failures, 0), FILL(systemErrors, 0)])',
      usingMetrics: { failures, systemErrors },
      label: 'failed invocations',
      period: FIVE_MINUTES,
    });
    const throttles = new cloudwatch.MathExpression({
      expression: 'FILL(runtimeThrottles, 0) + FILL(modelThrottles, 0)',
      usingMetrics: { runtimeThrottles, modelThrottles },
      label: 'throttled requests',
      period: FIVE_MINUTES,
    });
    // Share of Guardrail checks that stepped in, over an hour with at least 10 checks. A jump
    // means an attack, a misbehaving client, or a policy blocking legitimate questions.
    const guardrailBlockRate = new cloudwatch.MathExpression({
      expression: 'IF(checks >= 10, 100 * FILL(intervened, 0) / checks, 0)',
      usingMetrics: {
        checks: guardrailMetric('Invocations', Duration.hours(1)),
        intervened: guardrailMetric('InvocationsIntervened', Duration.hours(1)),
      },
      label: 'Guardrail block rate (%)',
      period: Duration.hours(1),
    });

    const alarms = [
      this.alarm('AgentErrors', errors, 1, `An ${agent.label} invocation failed (main.py logged agent_invocation_failed, or AgentCore counted a system error).`),
      this.alarm('AgentSlow', latencyP90.with({ period: Duration.minutes(15) }), 120_000,
        `${agent.label} p90 latency above 120 s over 15 minutes (chat is usually 10-25 s; report and swarm up to about 2 minutes).`,
        cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD),
      this.alarm('Throttled', throttles, 1, `${agent.label} or the Bedrock model ${modelId} throttled requests.`),
      this.alarm('GuardrailBlockRate', guardrailBlockRate, 30,
        'More than 30% of Guardrail checks were blocked or masked in the last hour (at least 10 checks).',
        cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD),
      ...props.tools.map((tool) =>
        this.alarm(`${pascal(tool.label)}Errors`, runtimeMetric(tool, 'SystemErrors', 'Sum'), 1, `The ${tool.label} runtime returned a system error.`)),
    ];

    const width = 8;
    const graph = (title: string, left: cloudwatch.IMetric[], extra: Partial<cloudwatch.GraphWidgetProps> = {}) =>
      new cloudwatch.GraphWidget({ title, left, width, height: 6, ...extra });
    this.dashboard = new cloudwatch.Dashboard(this, 'Dashboard', {
      dashboardName: 'agentoprod',
      defaultInterval: Duration.days(1),
      widgets: [
        [new cloudwatch.TextWidget({
          markdown: `## agentoprod\nRuntime **${agent.label}** (chat, report, swarm) with Guardrail and model **${modelId}**. Alarms notify the SNS topic \`agentoprod-alarms\`.`,
          width: 24, height: 2,
        })],
        [
          graph('Invocations', [invocations.with({ label: agent.label }), ...props.tools.map((t) => runtimeMetric(t, 'Invocations', 'Sum').with({ label: t.label }))]),
          graph('Errors and busy replies', [
            failures.with({ label: 'agent_invocation_failed' }), systemErrors.with({ label: 'system errors' }),
            userErrors.with({ label: 'user errors' }), busy.with({ label: 'busy replies' }),
          ]),
          graph('Latency (ms)', [latencyP50.with({ label: 'p50' }), latencyP90.with({ label: 'p90' })], {
            leftAnnotations: [{ value: 120_000, label: 'alarm' }],
          }),
        ],
        [
          graph('Model tokens', [model('InputTokenCount', 'Sum').with({ label: 'input' }), model('OutputTokenCount', 'Sum').with({ label: 'output' })]),
          graph('Model latency and throttles', [model('InvocationLatency', 'Average').with({ label: 'latency avg (ms)' })], {
            right: [modelThrottles.with({ label: 'throttles' })],
          }),
          graph('Guardrail checks', [
            guardrailMetric('Invocations', FIVE_MINUTES).with({ label: 'checks' }),
            guardrailMetric('InvocationsIntervened', FIVE_MINUTES).with({ label: 'blocked or masked' }),
          ]),
        ],
        [
          graph('Rejected sign-ins', [new cloudwatch.MathExpression({
            expression: `SEARCH('{AWS/Bedrock-AgentCore,ExceptionType,ResourceId} MetricName="InboundAuthorizationFailure"', 'Sum', 300)`,
            label: '',
            period: FIVE_MINUTES,
          })]),
          new cloudwatch.AlarmStatusWidget({ title: 'Alarms', alarms, width: 16, height: 6 }),
        ],
      ],
    });
  }

  /** A count of main.py log lines whose message starts with `prefix`, one per event. */
  private logMetric(runtime: MonitoredRuntime, metricName: string, prefix: string): cloudwatch.Metric {
    const filter = new logs.MetricFilter(this, `${metricName}Filter`, {
      logGroup: logs.LogGroup.fromLogGroupName(this, `${metricName}LogGroup`, runtime.logGroupName),
      filterPattern: logs.FilterPattern.all(
        logs.FilterPattern.stringValue('$.logger', '=', 'bedrock_agentcore.app'),
        logs.FilterPattern.stringValue('$.message', '=', `${prefix}*`),
      ),
      metricNamespace: NAMESPACE,
      metricName,
      metricValue: '1',
    });
    filter.node.addDependency(runtime.logGroupReady);
    return filter.metric({ statistic: 'Sum', period: FIVE_MINUTES });
  }

  private alarm(
    name: string,
    metric: cloudwatch.IMetric,
    threshold: number,
    description: string,
    comparisonOperator = cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
  ): cloudwatch.Alarm {
    const alarm = new cloudwatch.Alarm(this, name, {
      alarmName: `agentoprod-${name}`,
      alarmDescription: description,
      metric,
      threshold,
      comparisonOperator,
      evaluationPeriods: 1,
      treatMissingData: cloudwatch.TreatMissingData.NOT_BREACHING,
    });
    const action = new cwActions.SnsAction(this.topic);
    alarm.addAlarmAction(action);
    alarm.addOkAction(action);
    return alarm;
  }
}

/** AgentCore runtime metrics carry the runtime's name, operation and ARN as dimensions. */
function runtimeMetric(runtime: MonitoredRuntime, metricName: string, statistic: string): cloudwatch.Metric {
  return new cloudwatch.Metric({
    namespace: 'AWS/Bedrock-AgentCore',
    metricName,
    dimensionsMap: {
      Name: `${runtime.runtimeName}::DEFAULT`,
      Operation: 'InvokeAgentRuntime',
      Resource: runtime.runtimeArn,
    },
    statistic,
    period: FIVE_MINUTES,
  });
}

const pascal = (value: string) => value.charAt(0).toUpperCase() + value.slice(1);
