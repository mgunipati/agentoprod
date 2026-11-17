import { AgentCoreApplication, type AgentCoreApplicationProps } from '@aws/agentcore-cdk';
import { Stack, type StackProps } from 'aws-cdk-lib';
import * as bedrock from 'aws-cdk-lib/aws-bedrock';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';
import { createHash } from 'crypto';
import * as fs from 'fs';
import * as path from 'path';
import { AgentMonitoring } from './monitoring';

/** Runtimes that run an agent: they call models, use Memory and fetch API keys. */
const AGENT_RUNTIMES = ['agendev'];

/** Runtimes that only serve tools (the finance MCP server). */
const TOOL_ONLY_RUNTIMES = ['financetools'];

/**
 * What a tool-only runtime must never do. The AgentCore construct wires every project resource to
 * every runtime, so without this the MCP server could read and delete chat memory, call models
 * and fetch API keys. An explicit Deny overrides those generated Allows.
 */
const TOOL_ONLY_DENIED_ACTIONS = [
  'bedrock-agentcore:CreateEvent',
  'bedrock-agentcore:GetEvent',
  'bedrock-agentcore:ListEvents',
  'bedrock-agentcore:DeleteEvent',
  'bedrock-agentcore:GetMemory',
  'bedrock-agentcore:GetMemoryRecord',
  'bedrock-agentcore:ListMemoryRecords',
  'bedrock-agentcore:RetrieveMemoryRecords',
  'bedrock-agentcore:DeleteMemoryRecord',
  'bedrock-agentcore:BatchCreateMemoryRecords',
  'bedrock-agentcore:BatchUpdateMemoryRecords',
  'bedrock-agentcore:BatchDeleteMemoryRecords',
  'bedrock-agentcore:GetResourceApiKey',
  'bedrock-agentcore:GetApiKeyCredential',
  'bedrock:InvokeModel',
  'bedrock:InvokeModelWithResponseStream',
];

/**
 * The Bedrock model (or inference profile) the agent runtimes call. Set on the runtime as
 * BEDROCK_MODEL_ID and used by the dashboard, so the two cannot drift apart.
 */
const MODEL_ID = 'apac.amazon.nova-lite-v1:0';

/** The Guardrail every model call from an agent runtime must carry (see app/guardrails/README.md). */
const GUARDRAIL_DEFINITION = path.join('app', 'guardrails', 'finance-guardrail.json');

/** Runtime logs contain prompts and answers; keep them a month, not forever. */
const LOG_RETENTION = logs.RetentionDays.ONE_MONTH;

export interface AgentCoreStackProps extends StackProps {
  /** Props for the AgentCore application, produced by transformAgentCoreJson from agentcore/agentcore.json. */
  application: AgentCoreApplicationProps;
}

/**
 * The stack the AgentCore CLI deploys. Everything declared in agentcore/agentcore.json is
 * created by the AgentCoreApplication construct. Add your own resources below it and wire
 * them to your runtimes and harnesses through the application's accessors.
 */
export class AgentCoreStack extends Stack {
  public readonly application: AgentCoreApplication;

  constructor(scope: Construct, id: string, props: AgentCoreStackProps) {
    super(scope, id, props);
    this.application = new AgentCoreApplication(this, 'Application', props.application);

    const runtimeNames = (props.application.spec.runtimes ?? []).map((runtime) => runtime.name);
    const agentRuntimes = runtimeNames.filter((name) => AGENT_RUNTIMES.includes(name));
    this.grantExternalCredentialSecrets(props.application, agentRuntimes);
    const guardrail = this.addGuardrail(props.application.projectRoot, agentRuntimes);
    const toolRuntimes = runtimeNames.filter((name) => TOOL_ONLY_RUNTIMES.includes(name));
    this.restrictToolOnlyRuntimes(toolRuntimes);
    const logGroups = this.limitLogRetention(runtimeNames);
    for (const name of agentRuntimes) {
      this.application.runtime(name).addEnvironmentVariable('BEDROCK_MODEL_ID', MODEL_ID);
    }

    const monitored = (name: string) => {
      const runtime = this.application.runtime(name);
      return {
        label: name,
        runtimeArn: runtime.runtimeArn,
        runtimeName: runtime.runtimeName,
        logGroupName: logGroups[name].logGroupName,
        logGroupReady: logGroups[name].retention,
      };
    };
    if (agentRuntimes.length === 1) {
      new AgentMonitoring(this, 'Monitoring', {
        agent: monitored(agentRuntimes[0]),
        tools: toolRuntimes.map(monitored),
        guardrail,
        modelId: MODEL_ID,
      });
    }
  }

  /**
   * Let agent runtimes read the Secrets Manager secrets behind external API key credentials.
   *
   * A credential added with `--api-key-secret-reference` points AgentCore Identity at your own
   * secret (apiKeySecretSource EXTERNAL). The construct only lets runtimes read the secrets
   * Identity manages (`bedrock-agentcore-identity!*`), so fetching such a key with
   * GetResourceApiKey fails with AccessDeniedException until the runtime can read the secret.
   */
  private grantExternalCredentialSecrets(application: AgentCoreApplicationProps, runtimes: string[]): void {
    // transformAgentCoreJson drops secretRef from the spec it passes in, so read the
    // references from agentcore/agentcore.json itself.
    type Credential = { secretRef?: { secretId?: unknown } };
    const projectFile = path.join(application.projectRoot, 'agentcore', 'agentcore.json');
    const credentials: Credential[] = JSON.parse(fs.readFileSync(projectFile, 'utf8')).credentials ?? [];
    const secretArns = credentials
      .map((credential) => credential.secretRef?.secretId)
      .filter((arn): arn is string => typeof arn === 'string' && arn.startsWith('arn:'));

    secretArns.forEach((arn, index) => {
      const secret = secretsmanager.Secret.fromSecretCompleteArn(this, `ExternalCredentialSecret${index}`, arn);
      for (const name of runtimes) {
        secret.grantRead(this.application.runtime(name));
      }
    });
  }

  /**
   * Create the finance Guardrail from app/guardrails/finance-guardrail.json and apply it to agent runtimes.
   *
   * A numbered version is pinned (DRAFT can change underneath a running agent). Its logical ID
   * carries a hash of the definition, so editing the JSON creates a new version and points the
   * runtime at it in the same deploy. Each runtime gets BEDROCK_GUARDRAIL_ID and _VERSION, which
   * agent_factory passes on every model call, and an IAM Deny on any model call made without this
   * Guardrail, so code that forgets it fails instead of running unguarded.
   */
  private addGuardrail(projectRoot: string, runtimes: string[]): { arn: string; version: string } {
    const text = fs.readFileSync(path.join(projectRoot, GUARDRAIL_DEFINITION), 'utf8');
    const guardrail = new bedrock.CfnGuardrail(this, 'FinanceGuardrail', JSON.parse(text) as bedrock.CfnGuardrailProps);
    const hash = createHash('sha256').update(text).digest('hex').slice(0, 8);
    const version = new bedrock.CfnGuardrailVersion(this, `FinanceGuardrailVersion${hash}`, {
      guardrailIdentifier: guardrail.attrGuardrailId,
      description: `${GUARDRAIL_DEFINITION} sha256 ${hash}`,
    });

    for (const name of runtimes) {
      const runtime = this.application.runtime(name);
      runtime.addEnvironmentVariable('BEDROCK_GUARDRAIL_ID', guardrail.attrGuardrailArn);
      runtime.addEnvironmentVariable('BEDROCK_GUARDRAIL_VERSION', version.attrVersion);
      runtime.addToRolePolicy(
        new iam.PolicyStatement({
          sid: 'ApplyFinanceGuardrail',
          actions: ['bedrock:ApplyGuardrail'],
          resources: [guardrail.attrGuardrailArn],
        }),
      );
      runtime.addToRolePolicy(
        new iam.PolicyStatement({
          sid: 'DenyModelCallsWithoutFinanceGuardrail',
          effect: iam.Effect.DENY,
          actions: ['bedrock:InvokeModel', 'bedrock:InvokeModelWithResponseStream'],
          resources: ['*'],
          conditions: { ArnNotLike: { 'bedrock:GuardrailIdentifier': `${guardrail.attrGuardrailArn}:*` } },
        }),
      );
    }
    return { arn: guardrail.attrGuardrailArn, version: version.attrVersion };
  }

  /** Deny tool-only runtimes the Memory, model and API key access the construct gives every runtime. */
  private restrictToolOnlyRuntimes(runtimes: string[]): void {
    for (const name of runtimes) {
      this.application.runtime(name).addToRolePolicy(
        new iam.PolicyStatement({
          sid: 'ToolOnlyRuntimeDenyAgentAccess',
          effect: iam.Effect.DENY,
          actions: TOOL_ONLY_DENIED_ACTIONS,
          resources: ['*'],
        }),
      );
    }
  }

  /**
   * AgentCore creates each runtime's log group with no expiry; set a retention period instead.
   * LogRetention also creates the group if the runtime has not started yet, so metric filters
   * can depend on it.
   */
  private limitLogRetention(runtimes: string[]): Record<string, { logGroupName: string; retention: logs.LogRetention }> {
    const groups: Record<string, { logGroupName: string; retention: logs.LogRetention }> = {};
    for (const name of runtimes) {
      const runtime = this.application.runtime(name);
      const logGroupName = `/aws/bedrock-agentcore/runtimes/${runtime.runtimeId}-DEFAULT`;
      const retention = new logs.LogRetention(this, `LogRetention-${name}`, { logGroupName, retention: LOG_RETENTION });
      groups[name] = { logGroupName, retention };
    }
    return groups;
  }
}
