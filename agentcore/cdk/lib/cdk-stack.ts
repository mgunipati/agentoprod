import { AgentCoreApplication, type AgentCoreApplicationProps } from '@aws/agentcore-cdk';
import { Stack, type StackProps } from 'aws-cdk-lib';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import { Construct } from 'constructs';
import * as fs from 'fs';
import * as path from 'path';

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
    this.grantExternalCredentialSecrets(props.application);

    // Example: give a runtime a DynamoDB table.
    //
    // import * as dynamodb from 'aws-cdk-lib/aws-dynamodb';
    // const orders = new dynamodb.Table(this, 'Orders', {
    //   partitionKey: { name: 'pk', type: dynamodb.AttributeType.STRING },
    // });
    // const checkout = this.application.runtime('checkout'); // or .harness('support')
    // checkout.grantReadWrite(orders);
    // checkout.addEnvironmentVariable('ORDERS_TABLE', orders.tableName);
  }

  /**
   * Let every runtime read the Secrets Manager secrets behind external API key credentials.
   *
   * A credential added with `--api-key-secret-reference` points AgentCore Identity at your own
   * secret (apiKeySecretSource EXTERNAL). The construct only lets runtimes read the secrets
   * Identity manages (`bedrock-agentcore-identity!*`), so fetching such a key with
   * GetResourceApiKey fails with AccessDeniedException until the runtime can read the secret.
   */
  private grantExternalCredentialSecrets(application: AgentCoreApplicationProps): void {
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
      for (const runtime of application.spec.runtimes ?? []) {
        secret.grantRead(this.application.runtime(runtime.name));
      }
    });
  }
}
