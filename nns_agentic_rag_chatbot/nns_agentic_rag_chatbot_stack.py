from aws_cdk import (
    Stack,
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_iam as iam,
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