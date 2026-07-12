"""
Generates docs/architecture.png — the AWS architecture diagram for this
project — using the `diagrams` package (diagram-as-code, so it can be
regenerated when the architecture changes).

Needs GraphViz + the diagrams package:
    brew install graphviz          (or: apt/choco install graphviz)
    pip install diagrams

Run from the repo root: python docs/architecture_diagram.py

Note: Bedrock AgentCore (Gateway/Memory) and Guardrails don't have
official icons in the diagrams package yet, so they use the Bedrock icon
with explicit labels.
"""
from diagrams import Cluster, Diagram, Edge
from diagrams.aws.analytics import AmazonOpensearchService
from diagrams.aws.compute import Lambda
from diagrams.aws.ml import Bedrock
from diagrams.aws.security import WAF, Cognito
from diagrams.aws.storage import S3
from diagrams.onprem.client import User
from diagrams.onprem.compute import Server
from diagrams.programming.language import Python

with Diagram(
    "NNS Assistant — Agentic RAG Chatbot",
    filename="docs/architecture",
    outformat="png",
    show=False,
    direction="LR",
    graph_attr={
        "fontsize": "22",
        "pad": "0.4",
        "splines": "spline",
        "ranksep": "1.1",
        "nodesep": "0.5",
    },
):
    user = User("Employee")

    with Cluster("Local machine (free testing)"):
        ui = Python("Streamlit chat UI\n(chat_ui.py)")
        supervisor = Python("Supervisor agent\n(Strands, agents-as-tools)")
        with Cluster("Specialist agents"):
            hr = Python("HR agent")
            safety = Python("Safety agent")
            ops = Python("Operations agent")
        ollama = Server("Ollama llama3.1:8b\n(local LLM, default)")

    with Cluster("AWS us-east-1"):
        with Cluster("Bedrock services"):
            guardrail = Bedrock("Guardrail\n(ApplyGuardrail on\nevery input & output)")
            memory = Bedrock("AgentCore Memory\n(short-term history)")
            bedrock_llm = Bedrock("Claude\n(MODEL_PROVIDER=bedrock)")

        with Cluster("RAG — CDK stack"):
            docs_bucket = S3("Company docs bucket\n(sample_docs/*.txt)")
            kb = Bedrock("Knowledge Base\n(Titan embeddings)")
            vector_store = AmazonOpensearchService("OpenSearch Serverless\n(vector index)")

        with Cluster("Tool calling — boto3 scripts"):
            waf = WAF("WAF Web ACL\n(managed rules + rate limit)")
            cognito = Cognito("Cognito user pool\n(M2M client credentials)")
            gateway = Bedrock("AgentCore Gateway\n(MCP, JWT authorizer)")

        mock_lambda = Lambda("Mock tools Lambda — CDK\n(SMAX / calendar / Jabber)")

    # Chat flow
    user >> ui >> supervisor
    supervisor >> Edge(label="route") >> [hr, safety, ops]
    supervisor - Edge(style="dashed", label="LLM calls") - ollama
    supervisor - Edge(style="dashed", label="or") - bedrock_llm

    # Safety + memory (supervisor's handle_request wraps every turn)
    supervisor >> Edge(label="check input/output") >> guardrail
    supervisor - Edge(label="store / recall turns") - memory

    # RAG flow
    docs_bucket >> Edge(label="sync + embed") >> kb
    kb - vector_store
    [hr, safety, ops] >> Edge(label="Retrieve API") >> kb

    # Tool-calling flow
    [safety, ops] >> Edge(label="MCP over HTTPS\n(Bearer JWT)") >> gateway
    cognito >> Edge(label="issues JWT", style="dashed") >> gateway
    waf >> Edge(label="inspects requests", style="dashed") >> gateway
    gateway >> Edge(label="invoke") >> mock_lambda
