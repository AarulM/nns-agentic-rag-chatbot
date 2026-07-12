# NNS Assistant — Agentic RAG Chatbot

A multi-agent internal chatbot for a (fictional) shipbuilding company, built to
learn AWS Bedrock AgentCore + the Strands Agents SDK. A supervisor agent routes
employee questions to three specialists — HR, Safety, and Operations — which
answer via RAG over a Bedrock Knowledge Base and take real actions (tickets,
calendar, messages) through an AgentCore Gateway.

## Architecture

```
Streamlit UI (local)                        AWS
      │
      ▼
supervisor ──agents-as-tools──► hr_agent ─────► Bedrock Knowledge Base
   │ │                          safety_agent ──►   (S3 docs → Titan embeddings)
   │ │                          operations_agent
   │ └── AgentCore Memory                │
   │     (short-term history)            ▼
   │                            AgentCore Gateway (MCP, Cognito M2M auth, WAF)
   └── Bedrock Guardrail                 │
       (ApplyGuardrail on every          ▼
        input & output)         mock Lambda (SMAX tickets / calendar / Jabber)
```

- **CDK stack** (`nns_agentic_rag_chatbot/`): S3 docs bucket, Knowledge Base +
  data source, mock-tools Lambda, Gateway execution role, Bedrock Guardrail.
- **boto3 scripts** (repo root): Gateway, Cognito, Memory, and WAF — CDK
  support for these is still immature, so they're provisioned imperatively.
- **Agents** (`agents/`): Strands agents. `runtime_app.py` (future AgentCore
  Runtime entrypoint) is planned but not written yet.

## Local testing (free) vs real Bedrock

Every agent picks its model from `MODEL_PROVIDER` (see `agents/model_config.py`):

```bash
# default — free, local; needs `ollama serve` with llama3.1:8b pulled
export MODEL_PROVIDER=ollama

# real Claude on Bedrock (costs money)
export MODEL_PROVIDER=bedrock
export GUARDRAIL_ID=... GUARDRAIL_VERSION=...   # from CDK outputs; defaults in agents/guardrail.py
```

The Guardrail is enforced in **both** modes: `handle_request` runs every user
message and final reply through the standalone `ApplyGuardrail` API
(`agents/guardrail.py`) — blocking harmful/ITAR content and anonymizing PII —
so switching to Ollama doesn't switch off safety. In bedrock mode the model
invocation additionally applies it natively.

Note: the Knowledge Base retrieval and Gateway tool calls always hit real AWS
(small cost — the KB's OpenSearch backing is the expensive part), only the
LLM itself is swapped.

## Running the chat UI

```bash
cd agents
pip install -r requirements.txt
streamlit run chat_ui.py        # or: python supervisor.py for a terminal REPL
```

Requires `agents/gateway_secrets.py` (gitignored) or `COGNITO_CLIENT_SECRET`
exported — the value is printed at the end of `setup_gateway.py`.

## Provisioning AWS resources

In order:

1. `cdk deploy` — S3, Knowledge Base, Lambda, Gateway role, Guardrail.
   Upload `sample_docs/*.txt` to the docs bucket and sync the KB data source.
2. `python setup_gateway.py` — Cognito M2M user pool + AgentCore Gateway +
   Lambda target. Paste the Lambda/role ARNs from the CDK outputs in first;
   paste its printed IDs into `agents/mcp_gateway_client.py` after.
   (`finish_gateway_setup.py` resumes this if it dies partway.)
3. `python create_memory.py` — AgentCore short-term Memory; paste the printed
   MEMORY_ID into `agents/memory_hook.py`.
4. `python setup_waf.py` — WAF Web ACL (managed rules + rate limit) on the
   Gateway.

After any `cdk destroy` + `cdk deploy` cycle, `reconnect_gateway.py` (and
`finish_reconnect_gateway.py` if target deletion is slow to propagate)
re-points the surviving Gateway at the new Lambda/role — update the ARNs at
the top first.

## Tearing everything down

```bash
python teardown_everything.py
```

Deletes WAF → Gateway → Memory → Cognito → then runs `cdk destroy`. Safe to
re-run; each step skips what's already gone. Do this when not actively
testing — the Knowledge Base's OpenSearch collection bills hourly.

## Repo map

| Path | What it is |
|---|---|
| `agents/` | Supervisor + specialists, model picker, Gateway MCP client, Memory hook, trace log, Streamlit UI |
| `nns_agentic_rag_chatbot/` | CDK stack |
| `lambda/` | Mock SMAX/calendar/Jabber backend behind the Gateway |
| `sample_docs/` | Seed documents for the Knowledge Base |
| `setup_gateway.py`, `finish_gateway_setup.py` | Gateway + Cognito provisioning |
| `create_memory.py`, `setup_waf.py` | Memory and WAF provisioning |
| `reconnect_gateway.py`, `finish_reconnect_gateway.py` | Re-wire Gateway after a CDK redeploy |
| `teardown_everything.py` | Full teardown (boto3 resources + CDK stack) |

Hardcoded resource IDs (Gateway ID, Memory ID, KB ID, ARNs) throughout the
scripts and agents are point-in-time values for the current deployment — they
change on every teardown/rebuild and need re-pasting.
