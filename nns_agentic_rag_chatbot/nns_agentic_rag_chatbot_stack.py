from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_iam as iam,
    aws_bedrock as cfn_bedrock,
    RemovalPolicy,
    CfnOutput,
    Duration,
)
from constructs import Construct
from cdklabs.generative_ai_cdk_constructs import bedrock


class NnsAgenticRagChatbotStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # ---------- RAG: S3 + Knowledge Base (unchanged from before) ----------
        docs_bucket = s3.Bucket(
            self, "CompanyDocsBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
        )

        knowledge_base = bedrock.VectorKnowledgeBase(
            self, "CompanyKnowledgeBase",
            embeddings_model=bedrock.BedrockFoundationModel.TITAN_EMBED_TEXT_V2_1024,
            instruction=(
                "Use this knowledge base to answer employee questions about "
                "HR policy, safety procedures, and shipyard operations SOPs."
            ),
        )

        data_source = bedrock.S3DataSource(
            self, "CompanyDocsDataSource",
            bucket=docs_bucket,
            knowledge_base=knowledge_base,
            data_source_name="company-docs",
            chunking_strategy=bedrock.ChunkingStrategy.FIXED_SIZE,
        )

        CfnOutput(self, "DocsBucketName", value=docs_bucket.bucket_name)
        CfnOutput(self, "KnowledgeBaseId", value=knowledge_base.knowledge_base_id)
        CfnOutput(self, "DataSourceId", value=data_source.data_source_id)

        # ---------- NEW: mock SMAX/calendar/Jabber backend for MCP tools ----------
        # This Lambda will be registered as an AgentCore Gateway Target in the
        # next step (via a boto3 script, not CDK — Gateway itself is too new
        # for solid CDK support yet). CDK just creates the reliable pieces:
        # the function itself, and the role Gateway will assume to invoke it.
        mcp_tools_lambda = lambda_.Function(
            self, "McpToolsFunction",
            runtime=lambda_.Runtime.PYTHON_3_12,
            handler="lambda_mcp_tools_handler.lambda_handler",
            code=lambda_.Code.from_asset("lambda"),
            timeout=Duration.seconds(10),
        )

        # The role AgentCore Gateway assumes to call this Lambda on an
        # agent's behalf. Trust policy allows the Gateway service itself;
        # the grant_invoke line gives it permission to actually call our
        # specific function (nothing else).
        gateway_execution_role = iam.Role(
            self, "GatewayExecutionRole",
            assumed_by=iam.ServicePrincipal("bedrock-agentcore.amazonaws.com"),
        )
        mcp_tools_lambda.grant_invoke(gateway_execution_role)

        CfnOutput(self, "McpToolsLambdaArn", value=mcp_tools_lambda.function_arn)
        CfnOutput(self, "GatewayExecutionRoleArn", value=gateway_execution_role.role_arn)

        # ---------- NEW: Bedrock Guardrail (content filtering + PII redaction) ----------
        # Bedrock-native safety layer. Only applies when an agent calls Bedrock
        # (MODEL_PROVIDER=bedrock) — Ollama never touches this. Free to create;
        # tiny per-request cost only when it actually evaluates a real Bedrock call.
        guardrail = cfn_bedrock.CfnGuardrail(
            self, "CompanyGuardrail",
            name="nns-chatbot-guardrail",
            description="Safety guardrail for the NNS assistant: blocks harmful content, redacts PII, and blocks export-controlled/technical-data requests.",
            blocked_input_messaging="Sorry, I can't help with that request.",
            blocked_outputs_messaging="Sorry, I can't provide that response.",
            content_policy_config=cfn_bedrock.CfnGuardrail.ContentPolicyConfigProperty(
                filters_config=[
                    cfn_bedrock.CfnGuardrail.ContentFilterConfigProperty(
                        type=filter_type,
                        input_strength="HIGH",
                        # PROMPT_ATTACK is input-only — Bedrock requires its
                        # output strength to be NONE.
                        output_strength="NONE" if filter_type == "PROMPT_ATTACK" else "HIGH",
                    )
                    for filter_type in ("HATE", "INSULTS", "SEXUAL", "VIOLENCE", "MISCONDUCT", "PROMPT_ATTACK")
                ]
            ),
            sensitive_information_policy_config=cfn_bedrock.CfnGuardrail.SensitiveInformationPolicyConfigProperty(
                # NAME is deliberately absent: anonymizing it turned every
                # answer mentioning any person ("Your name is {NAME}", "the
                # president is {NAME}") into nonsense for a chatbot demo.
                pii_entities_config=[
                    cfn_bedrock.CfnGuardrail.PiiEntityConfigProperty(type=pii_type, action="ANONYMIZE")
                    for pii_type in ("EMAIL", "PHONE", "US_SOCIAL_SECURITY_NUMBER")
                ]
            ),
            topic_policy_config=cfn_bedrock.CfnGuardrail.TopicPolicyConfigProperty(
                topics_config=[
                    cfn_bedrock.CfnGuardrail.TopicConfigProperty(
                        name="ExportControlledTechnicalData",
                        definition=(
                            "Requests for technical specifications, drawings, or data about "
                            "ship designs, weapons systems, or defense articles that would be "
                            "considered export-controlled (ITAR/CUI) information."
                        ),
                        examples=[
                            "What are the hull design specifications for the submarine?",
                            "Can you give me the technical drawings for the weapons system?",
                        ],
                        type="DENY",
                    )
                ]
            ),
        )

        guardrail_version = cfn_bedrock.CfnGuardrailVersion(
            self, "CompanyGuardrailVersion",
            guardrail_identifier=guardrail.attr_guardrail_id,
            description="Initial published version",
        )

        CfnOutput(self, "GuardrailId", value=guardrail.attr_guardrail_id)
        CfnOutput(self, "GuardrailVersion", value=guardrail_version.attr_version)