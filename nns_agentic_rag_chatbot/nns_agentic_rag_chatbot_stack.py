from aws_cdk import (
    Stack,
    aws_s3 as s3,
    RemovalPolicy,
    CfnOutput,
)
from constructs import Construct
from cdklabs.generative_ai_cdk_constructs import bedrock


class NnsAgenticRagChatbotStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # S3 bucket that holds the company documents the agents can search.
        # RemovalPolicy.DESTROY + auto_delete_objects are fine for a learning
        # project (so `cdk destroy` cleans up completely). Remove both before
        # this ever touches real company documents.
        docs_bucket = s3.Bucket(
            self, "CompanyDocsBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
        )

        # One shared Knowledge Base for now (HR + Safety + Ops docs together).
        # We split this into per-department knowledge bases in a later step,
        # once the single shared one is working end to end.
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

        # Printed out after `cdk deploy` — you'll need all three for the
        # next step (uploading docs + running the ingestion job).
        CfnOutput(self, "DocsBucketName", value=docs_bucket.bucket_name)
        CfnOutput(self, "KnowledgeBaseId", value=knowledge_base.knowledge_base_id)
        CfnOutput(self, "DataSourceId", value=data_source.data_source_id)