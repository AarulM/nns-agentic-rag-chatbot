# NNS Agentic RAG Chatbot

A practice / learning project: an agentic **RAG-powered multi-agent chatbot** for a fictional
shipbuilding-style company. It uses **AWS Bedrock AgentCore**, the **Strands Agents SDK**,
**MCP tool-calling**, and **AWS CDK (Python)**.

> **This is a learning project built entirely on synthetic sample documents.** No real or
> proprietary data is used. The "ITAR/CUI" theme is intentional practice flavor only.

## Demo

The assistant answering a general question, then walking a new welder through
onboarding — routing to the right specialist agent and answering from the
company knowledge base:

![Demo — PPE question answered from company docs](docs/demo-chat-1.png)

![Demo — welder onboarding roadmap from the HR docs](docs/demo-chat-2.png)

---

## Table of contents

- [Capabilities (what's new)](#capabilities-whats-new)

1. [What you are building (the mental model)](#1-what-you-are-building-the-mental-model)
2. [Cost warning — read this first](#2-cost-warning--read-this-first)
3. [Prerequisites — install these once](#3-prerequisites--install-these-once)
4. [One-time AWS account setup](#4-one-time-aws-account-setup)
5. [Fresh-terminal checklist (you need this every time)](#5-fresh-terminal-checklist-you-need-this-every-time)
6. [Get the code running locally](#6-get-the-code-running-locally)
7. [Deploy the AWS resources (the rebuild runbook)](#7-deploy-the-aws-resources-the-rebuild-runbook)
   — includes [Running in AWS GovCloud](#running-in-aws-govcloud)
8. [Choose your model brain (Ollama vs Bedrock)](#8-choose-your-model-brain-ollama-vs-bedrock)
9. [Run the chatbot](#9-run-the-chatbot)
   — includes [uploading images, PDFs, audio, and video](#9a-uploading-images-pdfs-audio-and-video)
   and [IAM permissions the app needs](#9b-iam-permissions-the-app-needs)
10. [Smoke test](#10-smoke-test)
11. [Shut everything down (stop the costs)](#11-shut-everything-down-stop-the-costs)
12. [Debugging / troubleshooting](#12-debugging--troubleshooting)

---

## Capabilities (what's new)

Written for someone seeing this project for the first time.

- **Multimodal file support — any format.** Drop in images, PDFs, Office documents,
  spreadsheets, audio, video, or any text/source-code file, and the app routes each to the
  right extractor automatically (`agents/file_ingest.py`):
  - **Images/photos → a vision model** (Claude multimodal), not just OCR — so a photo *with
    no text in it* (scenery, a piece of equipment) is still described, where an OCR-only
    pipeline would return nothing. Any image format is normalized locally first (downscaled,
    re-encoded), so large phone photos don't get rejected.
  - **PDFs → read locally** with `pypdf` (free, instant) when they have a text layer;
    scanned/image-only PDFs fall back to **Bedrock Data Automation** for OCR.
  - **Spreadsheets → parsed locally** (pandas), every sheet flattened to text.
  - **Audio / video / `.docx` → Bedrock Data Automation** (speech-to-text, transcript +
    scene description, document OCR).
  - **Structured forms → Amazon Textract** (opt-in "form mode") for key/value pairs.
  - **Text and source code (any language) → read locally**, no AWS call, no cost. Unknown
    extensions are content-sniffed, so anything that's really text just works.
- **Immediate answers from an attachment.** A question about a *just-uploaded* file is
  answered directly from the extracted text — you don't wait for Knowledge Base indexing to
  finish. Indexing still happens in the background for later/cross-session retrieval.
- **PII redaction (local, zero added AWS cost).** All extracted text is scrubbed of names,
  emails, phone numbers, SSNs, and more via **Microsoft Presidio** (`agents/pii_redaction.py`)
  **before** it is ever written to S3 or indexed — covering every extraction path (vision,
  Textract, BDA, local). Chosen over AWS Comprehend specifically because it's local CPU, not a
  billed API call. Regex alone can't catch names; Presidio's NER can. Shipyard-specific terms
  (permit numbers, dock names) are deliberately preserved.
- **Extraction caching.** Re-uploading an identical file is near-instant and free — one S3
  GET instead of a fresh (paid) extraction, keyed by content hash.
- **Web search.** For questions outside the company docs (general knowledge, current events,
  external companies), the assistant can search the public web (`agents/web_search.py`,
  DuckDuckGo — no API key). Internal HR/Safety/Operations questions still go to the Knowledge
  Base.
- **Agents-as-Tools routing.** The supervisor routes to HR/Safety/Operations specialists via
  the Strands "agents as tools" pattern. Swarm and Graph were evaluated and **deliberately not
  adopted** — they'd add Bedrock round-trips this deterministic routing doesn't need.
- **Cost controls.** Staged upload objects and raw extraction output are **deleted
  immediately after use** (not left for a lifecycle sweep); the S3 lifecycle backstop is
  **1 day**; and an **AWS Budget alert** ($10/mo, emails at 50/90/100%) is created by
  `setup_billing_alarm.py`.
- **Voice agent — evaluated, deliberately deferred (not implemented).** A voice path (Amazon
  Nova Sonic via Strands bidirectional streaming) was considered and left out for now: Nova
  Sonic isn't available in GovCloud yet, and native audio would bypass the text-based ITAR
  guardrail. There is **no** mic button or voice entry point in the app — it's a future item,
  not half-wired.

---

## 1. What you are building (the mental model)

![AWS architecture diagram](docs/architecture.png)

There are **two separate worlds** in this project. Almost every file belongs to one of them.

**World 1 — Infrastructure (lives in AWS).** The document store, the search index, the tool
backend, memory, and the safety guardrail. Created by `cdk deploy` plus three boto3 setup
scripts. **This costs money while it exists.** You build it once per environment.

**World 2 — The agent app (runs on your laptop).** The actual chatbot logic. When you run
`streamlit run chat_ui.py`, this code connects to the World 1 resources *by their IDs* and
starts answering questions.

**The one wiring rule:** World 1 hands out fresh resource IDs every time you deploy, and
World 2 reads all of them from exactly **one editable file**: `.env` at the repo root (plus the
one secret in gitignored `agents/gateway_secrets.py`). `agents/aws_config.py` loads `.env`
automatically — you never edit it. After a deploy, the setup scripts print the new values in
paste-ready form; you paste them into those two files and you're wired up. That's the whole
runbook in Section 7.

### The runtime flow of one question

```
You type a question in the Streamlit UI (chat_ui.py)
        │
        ▼
Guardrail checks the question (ApplyGuardrail — blocks harmful/ITAR asks)
        │
        ▼
Supervisor agent (supervisor.py) decides who should answer
        │  routes to one of:
        ▼
HR agent  /  Safety agent  /  Operations agent
        │
        ├─► search_*_docs tool ──► Knowledge Base (RAG) ──► matching doc passages
        │                          (Titan embeddings + OpenSearch, in AWS)
        │
        └─► action tools ──► AgentCore Gateway ──► mock Lambda (tickets, calendar, Jabber)
        │
        ▼
Guardrail checks the answer (blocks disallowed output, masks emails/phones/SSNs)
```

A DynamoDB memory table records every turn; within one app run the assistant remembers the
conversation. Each app restart starts a fresh memory session on purpose (see troubleshooting).
In Bedrock mode, durable facts you state about yourself (name, badge number, supervisor) are
also extracted into long-term memory and recalled in future sessions.

### The AWS services used, and what each one does here

| Service | What it is | How this project uses it | Created by |
|---|---|---|---|
| **Amazon S3** | Object storage | Two buckets. The **docs bucket** holds the company documents (`sample_docs/*.txt`) — the raw material the RAG system knows; the Knowledge Base reads them from here. The **upload bucket** stages files dropped into the UI, plus Bedrock Data Automation's output, and is kept separate so a failed extraction never leaves a partial artifact where the KB would index it. | CDK stack (docs) / you, once (uploads) |
| **Amazon Bedrock Knowledge Base** | Managed RAG service | The retrieval half of RAG. On ingestion it splits each S3 doc into chunks, turns each chunk into a vector with **Titan Text Embeddings V2**, and stores them. At question time, the specialist agents call its `Retrieve` API (`agents/knowledge_base.py`) and get back the 5 most relevant passages. | CDK stack |
| **Amazon OpenSearch Serverless** | Vector database | The index that actually stores and searches the embeddings for the Knowledge Base. Invisible to the code but **it's the ~$1/hr cost driver** — it bills for existing, not for being used. | CDK stack (created for the KB) |
| **Amazon Bedrock Guardrails** | Managed content-safety layer | Every user message and every final answer passes through the `ApplyGuardrail` API (`agents/guardrail.py`): harmful-content filters, a custom ITAR/export-control topic that refuses submarine-drawings-style asks, and PII masking (emails/phones/SSNs become `{EMAIL}` etc.). Works in Ollama mode too, because the app calls it directly. | CDK stack |
| **Bedrock AgentCore Gateway** | Managed MCP tool server | Exposes the company "action" tools (create ticket, check calendar, send Jabber) to any agent over the **MCP protocol**. The agents connect over HTTPS with a bearer token (`agents/mcp_gateway_client.py`); the Gateway validates the token, then invokes the Lambda that implements the tool. Swap the Lambda for real SMAX/calendar/Jabber integrations later and no agent code changes. | `setup_gateway.py` (CDK support is still immature) |
| **Amazon Cognito** | Identity / OAuth service | Machine-to-machine auth for the Gateway. A Cognito "user pool" holds an app client (ID + secret); the agent exchanges those for a short-lived JWT access token (`client_credentials` flow), and the Gateway only accepts requests carrying a valid token. This is why `gateway_secrets.py` exists. | `setup_gateway.py` |
| **AWS WAF** | Web application firewall | Sits in front of the Gateway's public URL: two AWS managed rule sets (common exploits, known bad inputs) plus a 2000-requests-per-5-min-per-IP rate limit. Optional hardening — Cognito already gates access. | `setup_waf.py` |
| **AWS Lambda** | Serverless functions | `lambda/lambda_mcp_tools_handler.py` — the mock backend standing in for real company systems. The Gateway invokes it per tool call; it returns fake ticket IDs, seeded calendar events, and message confirmations from in-memory data. | CDK stack |
| **Amazon DynamoDB** | Serverless NoSQL database | The memory store — one table holds both kinds of memory. Short-term: `agents/memory_hook.py` writes every user/assistant turn (on a background thread, 7-day TTL) and reloads recent turns when an agent starts. Long-term (Bedrock mode): `agents/memory_store.py` keeps durable user facts, auto-extracted by Strands' MemoryManager and injected into context across sessions. AgentCore Memory was used originally, but it isn't available in AWS GovCloud — DynamoDB is. | `create_memory.py` |
| **Amazon Bedrock (model inference)** | Managed LLM hosting | The optional cloud brain: with `MODEL_PROVIDER=bedrock`, all four agents call Claude (Haiku 4.5 by default; Sonnet 4.5 in GovCloud) through Bedrock's Converse API instead of local Ollama. Also hosts the Titan embedding model the KB uses either way. | AWS-hosted; enabled via Model access (Section 4d) |
| **Bedrock Data Automation (BDA)** | Managed multimodal extraction | The default path for every uploaded file (`agents/file_ingest.py`): OCR for images and PDFs, speech-to-text for audio, transcript + scene descriptions for video — one async API for all of it, which is why this project doesn't wire up Textract, Transcribe, and a vision model separately. Output is chunked into the Knowledge Base like any other document. **GovCloud (US-West) only.** | AWS-hosted; uses the AWS-managed default project |
| **Amazon Textract** | OCR with document structure | Used only for the sidebar's "Structured form mode": `AnalyzeDocument` with `FORMS`/`TABLES` pulls key/value pairs out of permits and inspection checklists, which BDA's default project returns as prose. Single-page images and PDFs. | AWS-hosted |
| **AWS IAM** | Permissions | Two roles matter: the Lambda's execution role, and the **Gateway execution role** — the identity the Gateway assumes when invoking the Lambda, granted `lambda:InvokeFunction` on that one function only. Your `nns-agent` profile credentials authorize everything the scripts and agents do. | CDK stack / you (Section 4) |
| **AWS CloudFormation** | Infrastructure-as-code engine | What `cdk deploy` actually drives: the CDK Python code synthesizes a CloudFormation template, and CloudFormation creates/updates/deletes the World-1 resources as one stack. The setup scripts also read the stack's outputs (ARNs) so you never paste them. | `cdk deploy` |
| **Amazon CloudWatch** | Logs & metrics | Every Lambda invocation and WAF decision lands here automatically — it's how you verify a Jabber "send" actually invoked the backend (`/aws/lambda/...McpToolsFunction...` log group). | Automatic |

How they chain together for one question: **Guardrail** (screen input) → supervisor routes →
specialist either queries **Knowledge Base**/**OpenSearch** (built from **S3** docs) or gets a
**Cognito** token and calls the **Gateway** (through **WAF**) which invokes **Lambda** →
**Guardrail** again (screen/mask output) → **Memory** records the turn. The LLM doing the
thinking at each step is either local Ollama or **Bedrock** Claude.

Uploading a file is a separate, one-directional chain: **S3** (staging) → **Bedrock Data
Automation** (or **Textract** in form mode) → **S3** (docs bucket, as text + a metadata
sidecar) → **Knowledge Base** ingestion job → **OpenSearch**. From then on it is
indistinguishable from a document that was there at deploy time.

### Which file does what

| File | World | Job |
|---|---|---|
| `nns_agentic_rag_chatbot/nns_agentic_rag_chatbot_stack.py` | 1 | CDK stack: S3 bucket, Knowledge Base, Lambda, IAM role, Guardrail |
| `app.py`, `cdk.json` | 1 | CDK wrapper that runs the stack |
| `lambda/lambda_mcp_tools_handler.py` | 1 | Mock ticket/calendar/Jabber backend |
| `sample_docs/*.txt` | 1 | The only knowledge the RAG system can draw on |
| `setup_gateway.py` | 1 | Cognito auth + Gateway + Lambda MCP target. **Idempotent** — re-run it after any crash or redeploy and it repairs itself |
| `create_memory.py` | 1 | DynamoDB memory table (get-or-create, safe to re-run) |
| `setup_upload_bucket.py` | 1 | Creates the upload/staging bucket with lifecycle rules. Outside CDK on purpose, so it survives teardown |
| `setup_waf.py` | 1 | Optional firewall in front of the Gateway |
| `teardown_everything.py` | 1 | Deletes all AWS resources to stop costs — finds everything by name, nothing to edit |
| `.env` / `.env.example` | 2 | **The one place all resource IDs live.** `.env` is gitignored; `.env.example` documents every variable |
| `agents/aws_config.py` | 2 | Loads `.env` and derives region/partition-specific ARNs from it |
| `agents/env_check.py` | 2 | Startup validation — fails fast with a readable message instead of a boto3 traceback |
| `agents/file_ingest.py` | 2 | Multimodal ingestion: image/PDF/audio/video → text → Knowledge Base |
| `agents/gateway_secrets.py` | 2 | Gitignored — the Cognito client secret (the one real credential) |
| `agents/model_config.py` | 2 | Switch between Ollama (free) and Bedrock (paid) |
| `agents/hr_agent.py` / `safety_agent.py` / `operations_agent.py` | 2 | The three specialist agents |
| `agents/knowledge_base.py` | 2 | Shared Knowledge Base search call used by all three specialists |
| `agents/supervisor.py` | 2 | Router — "agents as tools" pattern, plus greeting fast-path and guardrail wiring |
| `agents/guardrail.py` | 2 | Runs the Bedrock Guardrail on every input/output (works in Ollama mode too) |
| `agents/mcp_gateway_client.py` | 2 | Logs into the Gateway, calls the action tools |
| `agents/memory_hook.py` | 2 | Saves/reloads conversation turns (short-term memory) |
| `agents/memory_store.py` | 2 | Long-term user facts via Strands MemoryManager (Bedrock mode) |
| `agents/trace_log.py` | 2 | Queue that carries live tool-call events to the UI |
| `agents/chat_ui.py` | 2 | Local Streamlit chat interface + the file-upload sidebar |
| `tests/` | 2 | Smoke tests per file type; offline by default, live AWS behind a flag |
| `docs/architecture_diagram.py` | — | Regenerates `docs/architecture.png` (diagram-as-code) |

**Mental shortcut when lost:** ask *"does this file BUILD AWS stuff or USE AWS stuff?"* — and
if it uses AWS stuff, the ID it needs comes from `.env`.

---

## 2. Cost warning — read this first

- The **OpenSearch Serverless** collection behind the Knowledge Base is the main ongoing cost:
  **roughly $1/hour for as long as it exists**, whether or not you are using it — about
  **$24/day** if you forget to tear it down.
- **Bedrock model calls** are pay-per-token, but tiny for testing. The default Bedrock model is
  Claude Haiku 4.5 (cheapest tier). Guardrail checks cost a fraction of a cent per message.
- Everything else (S3, Lambda, Cognito, Memory, Gateway) is effectively free at this scale.
  WAF is a few dollars a month if you leave it up.
- **AWS billing dashboards lag by ~24 hours** — do not rely on the dashboard to confirm you've
  stopped spending. Trust the teardown script and the check in Section 11.

> **Golden rule:** when you stop working for the day, run `python teardown_everything.py`.
> You can rebuild in ~15 minutes with Section 7.

---

## 3. Prerequisites — install these once

| Tool | Why | Check it worked |
|---|---|---|
| **Git** | Clone the repo | `git --version` |
| **Python 3.12** | Runs everything | `python3 --version` (Mac) / `python --version` (Windows) |
| **Node.js 18+** | AWS CDK runs on Node | `node --version` |
| **AWS CDK** | Deploys the infrastructure | `npm install -g aws-cdk` then `cdk --version` |
| **AWS CLI v2** | Talks to AWS from the terminal | `aws --version` |
| **Ollama** *(optional)* | Free local model, if not using Bedrock | `ollama --version` |

Install links:
- Python: <https://www.python.org/downloads/> (tick **"Add Python to PATH"** on Windows)
- Node.js: <https://nodejs.org/> (LTS version)
- AWS CLI: <https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html>
- Ollama (optional): <https://ollama.com/download>, then `ollama pull llama3.1:8b`

Run each "check it worked" command and confirm a version number before moving on.

---

## 4. One-time AWS account setup

### 4a. Create an IAM user

AWS Console → IAM → Users → create a user with programmatic access (for a personal learning
account, `AdministratorAccess` is simplest; a real project would scope this down). Save the
**Access Key ID** and **Secret Access Key**. Never commit them — `.gitignore` blocks
`*_accessKeys.csv` for a reason.

### 4b. Configure a named profile called `nns-agent`

```bash
aws configure --profile nns-agent
```

Answer the prompts: your Access Key ID, your Secret Access Key, region **us-east-1**, output
format Enter (default).

### 4c. Verify the region is actually set (this bites people)

Open your AWS config file — Mac: `~/.aws/config`, Windows: `C:\Users\<you>\.aws\config` —
and confirm the block looks like:

```ini
[profile nns-agent]
region = us-east-1
output = json
```

> **Known gotcha:** if the `region =` line is missing, boto3 crashes with `NoRegionError`.
> Add the line by hand if `aws configure` didn't.

### 4d. Enable Bedrock model access

Bedrock blocks models until you request access. In the AWS Console → **Bedrock** →
**Model access** (make sure you are in **us-east-1**), request access to:

1. **Anthropic Claude Haiku 4.5** (the default Bedrock model here) — plus Sonnet if you want it
2. **Amazon Titan Text Embeddings V2** — the Knowledge Base needs it for RAG (required even in
   Ollama mode)

Wait until status shows **Access granted** (usually instant to a few minutes).

### 4e. Confirm the CLI is talking to AWS

```bash
aws sts get-caller-identity --profile nns-agent
```

You should see your account number and user ARN.

---

## 5. Fresh-terminal checklist (you need this every time)

**Every new terminal tab starts "cold."** On Mac, new tabs often auto-activate a conda
`(base)` environment that shadows the project's `.venv`. Run these lines every time you open
a terminal for this project.

**Mac (zsh):**
```bash
cd ~/Projects/nns-agentic-rag-chatbot
source .venv/bin/activate
export AWS_PROFILE=nns-agent
export AWS_PAGER=""
```

**Windows (PowerShell):**
```powershell
cd $HOME\Projects\nns-agentic-rag-chatbot
.\.venv\Scripts\Activate.ps1
$env:AWS_PROFILE = "nns-agent"
$env:AWS_PAGER = ""
```

You know it worked when your prompt starts with `(.venv)`.

> If PowerShell refuses with "running scripts is disabled", run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` and answer Yes.

> **Never paste angle brackets `< >` into the terminal.** Placeholders like `<KB_ID>` in this
> README mean "type your real value here, without the brackets."

---

## 6. Get the code running locally

### 6a. Clone the repo

```bash
git clone https://github.com/AarulM/nns-agentic-rag-chatbot.git
cd nns-agentic-rag-chatbot
```

### 6b. Create and activate the virtual environment

**Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows:**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 6c. Install dependencies

Every version is pinned exactly, so a clone reproduces this environment rather than
whatever happens to be current on PyPI.

```bash
pip install --upgrade pip
pip install -r requirements.txt -r agents/requirements.txt

# REQUIRED for file uploads: the spaCy model that PII redaction uses to
# detect names. This is a separate download, NOT a pip dependency — pip
# installs Presidio and spaCy, but not the model itself.
python -m spacy download en_core_web_lg

# Only if you intend to run the test suite:
pip install -r requirements-dev.txt
```

`requirements.txt` is CDK-only (deploy/destroy); `agents/requirements.txt` is the chatbot
itself; `requirements-dev.txt` is pytest plus the two libraries used to generate test
fixtures.

> **Don't skip the `spacy download`.** Extracted text is scrubbed of PII (names,
> SSNs, emails, ...) locally with [Presidio](https://microsoft.github.io/presidio/)
> **before** it is written to S3 or the Knowledge Base — no AWS Comprehend call, so
> no per-file cost. Redaction **fails closed**: if the model is missing, file uploads
> are rejected with an install hint rather than silently writing raw PII to S3.
> `python agents/env_check.py` reports `[ warn ] PII redaction` if it isn't ready.
> `en_core_web_sm` works as a smaller fallback; set `PII_REDACTION=off` in `.env`
> only if you deliberately want redaction disabled (not recommended).

### 6c-2. Create your `.env`

```bash
cp .env.example .env
```

`.env.example` documents every variable the project reads, with no real values in it.
You'll fill `.env` in after deploying (Step 6). It is gitignored — never commit it.

### 6d. Bootstrap CDK (once per AWS account + region)

```bash
cdk bootstrap --profile nns-agent
```

This creates a small S3 bucket CDK uses to stage deployments. Harmless and nearly free.

---

## 7. Deploy the AWS resources (the rebuild runbook)

> Do the steps **in order**, one at a time, and read each output before continuing. The
> scripts find AWS resources by name and read ARNs from the CloudFormation stack, so **you
> never edit the scripts** — only Step 6 pastes values, into the two agent config files.

### Step 1 — Deploy the CDK stack

Your venv must be active (Section 5) — CDK runs `python3 app.py` under the hood.

```bash
cdk deploy --profile nns-agent
```

Confirm the prompt (`y`). When it finishes it prints an **Outputs** block — keep it visible,
you'll use `DocsBucketName`, `KnowledgeBaseId`, `DataSourceId`, and `GuardrailId`.

> **This is the moment OpenSearch starts costing ~$1/hr.** The clock is now running.

### Step 2 — Upload the sample docs and ingest them

Upload (use your `DocsBucketName`):

```bash
aws s3 cp sample_docs/ s3://<DocsBucketName>/ --recursive --profile nns-agent
```

Start the ingestion job (builds the searchable vector index):

```bash
aws bedrock-agent start-ingestion-job --knowledge-base-id <KnowledgeBaseId> --data-source-id <DataSourceId> --profile nns-agent
```

Poll until `"status": "COMPLETE"` (re-run every ~20 seconds; `ingestionJobId` is in the
previous command's output):

```bash
aws bedrock-agent get-ingestion-job --knowledge-base-id <KnowledgeBaseId> --data-source-id <DataSourceId> --ingestion-job-id <ingestionJobId> --profile nns-agent
```

**Do not continue until it is COMPLETE** — the agents cannot search documents that haven't
been ingested.

### Step 3 — Create the Gateway (Cognito + Gateway + MCP target)

```bash
python setup_gateway.py
```

No editing needed — it reads the Lambda/role ARNs from the stack outputs. It ends by printing
two paste-ready blocks (Gateway URL, Cognito domain/client ID, and the client secret).
**Keep them for Step 6.**

> If it crashes partway (AWS async timing), just **run it again** — every step is
> get-or-create, so a re-run finishes what the first run started.

### Step 4 — Create Memory

```bash
python create_memory.py
```

Creates the `NnsChatbotMemory` DynamoDB table. The name is fixed and already matches the
default in `agents/aws_config.py`, so there is nothing to paste. Safe to re-run; if the
table already exists it just says so.

How memory works after this:

- **Short-term** (all modes): every user/assistant turn is saved to the table on a
  background thread and the last 5 turns are reloaded when the app starts. Turns expire
  after 7 days automatically (DynamoDB TTL).
- **Long-term** (`MODEL_PROVIDER=bedrock` only): durable facts you state about yourself
  ("my badge number is 40213", "my supervisor is Priya") are extracted in the background
  by Strands' MemoryManager (`agents/memory_store.py`) and injected back into context in
  **future sessions** — the assistant remembers you across app restarts. Ollama mode
  skips this on purpose: llama3.1:8b mis-handles injected memory and invents facts
  during extraction (details in the `memory_manager` comment in `agents/supervisor.py`).
- **If you skip this step**, the app still runs — it prints a warning telling you to run
  `create_memory.py` and answers questions without memory.

The IAM identity running the app needs DynamoDB permissions on this table:
`CreateTable`/`DescribeTable`/`UpdateTimeToLive` (setup script), and
`PutItem`/`Query`/`BatchWriteItem`/`DeleteTable` (app + teardown). An
`AmazonDynamoDBFullAccess`-style policy covers all of it.

### Step 5 — Create the upload/staging bucket

The multimodal upload pipeline stages files in their own S3 bucket, kept out of the CDK
stack on purpose so it survives teardown (and keeps the extraction cache). Pick a
globally-unique name, set `UPLOAD_BUCKET` in `.env` (e.g. `nns-multimodal-<account-id>`), then:

```bash
python setup_upload_bucket.py
```

Creates the bucket, blocks public access, enables encryption, and applies the 1-day
lifecycle rules. Idempotent — safe to re-run. Details in [§9a](#9a-uploading-images-pdfs-audio-and-video).

### Step 6 — Create the billing alert (recommended)

A low-threshold AWS Budget so a left-running stack emails you before it costs much. Set
`BILLING_ALERT_EMAIL` in `.env` (and optionally `BILLING_BUDGET_LIMIT`, default `10`), then:

```bash
python setup_billing_alarm.py
```

Emails you at 50% / 90% actual and 100% forecast of the monthly budget — no confirmation
click needed. Idempotent. **GovCloud note:** AWS Budgets isn't offered in GovCloud; the
script detects that and exits — use a CloudWatch billing alarm on `AWS/Billing
EstimatedCharges` there instead.

### Step 7 — (Optional) Firewall

The Gateway is already protected by Cognito auth, so this is optional hardening:

```bash
python setup_waf.py
```

> **Known gotcha:** the WAF association sometimes fails with `WAFUnavailableEntityException`
> even when both resources are healthy. Treated as **optional/skippable** — if it fails, move on.

### Step 8 — Paste the values into `.env`

Configuration lives in a gitignored **`.env`** at the repo root, not in source. That way a
clone picks up *your* deployment's IDs instead of someone else's, and the same code runs
against commercial AWS or GovCloud with no edits.

1. If you haven't already: `cp .env.example .env`
2. Fill in the values the previous steps printed:
   - `KNOWLEDGE_BASE_ID`, `DATA_SOURCE_ID`, `DOCS_BUCKET`, `GUARDRAIL_ID` — Step 1
   - `UPLOAD_BUCKET` — the staging bucket for file uploads (see
     [§9a](#9a-uploading-images-pdfs-audio-and-video))
   - `GATEWAY_URL`, `COGNITO_DOMAIN`, `COGNITO_CLIENT_ID` — Step 3
   - `MEMORY_TABLE` — leave as-is; Step 4 uses a fixed table name
3. Create **`agents/gateway_secrets.py`** (it's gitignored, so it won't exist on a fresh
   clone) containing one line, with the secret Step 3 printed:

   ```python
   COGNITO_CLIENT_SECRET = "<paste the secret here>"
   ```

Now verify. This checks every variable, your credentials, the region/partition match, model
access, and that Bedrock Data Automation is reachable — and prints exactly what to fix:

```bash
python agents/env_check.py
```

It exits non-zero if anything is missing, so you find out here rather than three screens
into a stack trace. The app runs the same checks at startup and refuses to load if any of
them fail.

You are now fully deployed.

### Running in AWS GovCloud

> **Dev/test is in `us-east-1` commercial** (what this README deploys by default). **The real
> production target is AWS GovCloud (US-West)** — end users are there. GovCloud has no
> credentials wired up in this repo yet, so the GovCloud path below is a *documented
> checklist verified against commercial `us-east-1`*, not something that's been run end-to-end
> in GovCloud. See the pre-production punch list at the end of this section for what's still open.

The same runbook works in GovCloud (us-gov-west-1 / us-gov-east-1) with these differences,
all already handled by the code once `AWS_REGION` is set:

```bash
export AWS_REGION=us-gov-west-1     # before every script and before running the app
```

- **Memory:** Bedrock AgentCore Memory does **not exist in GovCloud**
  ([docs](https://docs.aws.amazon.com/govcloud-us/latest/UserGuide/govcloud-bedrock-agentcore.html)),
  which is why memory lives in DynamoDB (available everywhere). Just run
  `python create_memory.py` once with the region exported — nothing else changes.
- **Cognito token URL:** GovCloud uses `<domain>.auth-fips.<region>.amazoncognito.com`
  instead of `.auth.`. `agents/mcp_gateway_client.py` picks the right one from
  `AWS_REGION` automatically.
- **Bedrock models:** GovCloud has no `global.` inference profiles and no Haiku 4.5; its
  profiles are prefixed `us-gov.` and the FedRAMP-authorized Claude models there are
  Sonnet 4.5, 3.7 Sonnet, 3.5 Sonnet, and Claude 3 Haiku. With `AWS_REGION=us-gov-*` the
  code defaults to `us-gov.anthropic.claude-sonnet-4-5-20250929-v1:0`. If that ID isn't
  enabled in your account, list what is and export it:

  ```bash
  aws bedrock list-inference-profiles --query "inferenceProfileSummaries[].inferenceProfileId"
  export BEDROCK_MODEL_ID=<one of the printed IDs>
  ```

- **File uploads:** Bedrock Data Automation is available in **GovCloud (US-West) only**
  ([docs](https://docs.aws.amazon.com/govcloud-us/latest/UserGuide/govcloud-bedrock.html)),
  so run in `us-gov-west-1` if you want the upload sidebar. BDA's inference-profile ARN is
  partition-specific (`us-gov.data-automation-v1` rather than `us.`); `agents/aws_config.py`
  derives it from `AWS_REGION`, so there is nothing to change.
- **Knowledge Base limits in GovCloud** — two that affect this project
  ([docs](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-managed-regions.html)):
  only the **S3** data source connector exists (fine — that's what the stack uses), and
  there are **no service-managed embedding models**, so the KB must name its own. The CDK
  stack already pins `TITAN_EMBED_TEXT_V2_1024` explicitly, so this is handled — but
  confirm Titan Embed v2 is enabled in your account before deploying.
- **Billing alert:** AWS Budgets (used by `setup_billing_alarm.py`) is **not offered in
  GovCloud** — the script detects that and exits. Use a **CloudWatch billing alarm** on the
  `AWS/Billing EstimatedCharges` metric instead.
- **Partitions:** every ARN the code builds uses `aws-us-gov` instead of `aws`, derived
  from `AWS_REGION`. Nothing is hardcoded.

> #### 📋 Pre-production punch list (open items before GovCloud go-live)
>
> None block dev/test in `us-east-1`; all must be closed before the GovCloud production cutover:
>
> - **Bedrock model EULA acceptance** — must be done by hand in the linked *commercial*
>   account, then the model enabled in GovCloud (see the callout below). Not yet done.
> - **Titan Embed v2 availability** — the KB pins `TITAN_EMBED_TEXT_V2_1024`; confirm it's
>   enabled in the GovCloud account before deploying. Unconfirmed.
> - **Vision/image path unverified in GovCloud** — image reading uses a Bedrock multimodal
>   model (Sonnet 4.5 in GovCloud). Verified end-to-end in commercial `us-east-1`; not yet run
>   in GovCloud.
> - **Billing alert** — Budgets isn't in GovCloud; stand up a CloudWatch billing alarm there.
> - **Voice agent** — still deferred (no Nova Sonic in GovCloud; native audio bypasses the
>   text guardrail). Not implemented; not a blocker.

> #### ⚠️ The one step that cannot be automated
>
> **GovCloud is a separate account in a separate partition, and Bedrock model access there
> requires accepting the model's EULA in the linked *commercial* account first, then
> enabling the model in GovCloud.** No script can do this for you — it is a console action
> in two different accounts. Do it before your first `MODEL_PROVIDER=bedrock` run.
>
> A corollary that costs people an afternoon: **commercial credentials cannot reach
> GovCloud at all.** If `AWS_REGION` is `us-gov-*` but your profile is a commercial one,
> STS fails with `InvalidClientTokenId` — which reads like expired keys, but refreshing
> them will not help. `python agents/env_check.py` detects this case specifically and
> tells you it is a partition mismatch rather than a credential problem.

---

## 8. Choose your model brain (Ollama vs Bedrock)

`agents/model_config.py` switches the LLM via the `MODEL_PROVIDER` environment variable.
**If unset, it defaults to Ollama** — on purpose, so you can't accidentally rack up charges.

- **Ollama (default)** — free and local. Needs the Ollama app running (`ollama serve`, or just
  launch the app) with `llama3.1:8b` pulled. The code pins temperature 0 and an 8192-token
  context window — without those, the small model misroutes tools and derails long chats.
  Note: Knowledge Base search and Gateway calls still hit real AWS; only the LLM is local.
- **Bedrock** — real Claude, more reliable answers, costs pennies for testing. Defaults to
  **Claude Haiku 4.5** (cheapest tier); export `BEDROCK_MODEL_ID` to use a bigger model.
  Requires Section 4d model access.

**Use Bedrock (Mac):** `export MODEL_PROVIDER=bedrock` (add to `~/.zshrc` to persist)

**Use Bedrock (Windows):** `$env:MODEL_PROVIDER = "bedrock"`

Confirm what's active:

```bash
cd agents
python -c "from model_config import get_model; print(type(get_model()).__name__)"
cd ..
```

Prints `OllamaModel` or `BedrockModel`.

---

## 9. Run the chatbot

Fresh terminal (Section 5), model provider chosen (Section 8), then:

**Streamlit browser UI (shows live tool calls):**
```bash
cd agents
streamlit run chat_ui.py
```

Opens at <http://localhost:8501>. Runs **only on your machine** — not hosted or public.

**Or the terminal REPL:**
```bash
cd agents
python supervisor.py
```

### 9a. Uploading images, PDFs, audio, video, and any other file

Attach a file with the paperclip inside the chat input bar. **Any file format is accepted** —
images (`.png .jpg .jpeg .gif .webp .bmp .tiff`), PDFs, Office documents (`.docx`),
spreadsheets (`.xlsx .xls`), audio (`.mp3 .wav .m4a .flac .ogg .amr`), video (`.mp4 .mov`),
and any text or source-code file in any language. Unknown extensions are content-sniffed, so
anything that's really text is read too.

Two things happen with an attachment: you get an **immediate answer** from the extracted text
(no waiting on indexing), and the text is also written into the same Knowledge Base the typed
documents live in for later/cross-session retrieval. All extracted text is **PII-redacted**
(see [Capabilities](#capabilities-whats-new)) before it touches S3 or the KB.

**One-time setup.** The pipeline stages uploads in its own S3 bucket, kept separate from
the KB's data source so a failed extraction never leaves a partial artifact where the KB
will index it. It is *not* created by `cdk deploy`, on purpose — it survives
`teardown_everything.py`, so a rebuild doesn't throw away extractions you already paid for.

Pick a globally-unique name, set `UPLOAD_BUCKET` in `.env`, then:

```bash
python setup_upload_bucket.py
```

That creates the bucket, blocks public access, enables encryption, and applies the
lifecycle rules below. Idempotent — safe to re-run.

**How a file is routed** (`agents/file_ingest.py` picks the cheapest tool that fits):

| Input | Handled by | Why |
| --- | --- | --- |
| Text, source code, config, CSV | read locally | no AWS call — free and instant |
| Spreadsheets (`.xlsx`/`.xls`) | read locally (pandas) | a spreadsheet is a zip of XML, not something a service is needed for |
| **Images / photos** | **a vision model** (Claude multimodal) | describes the scene *and* transcribes any text — so a photo with no text still works, where OCR returns nothing. Normalized/downscaled locally first so big photos aren't rejected |
| **PDFs** | **`pypdf` locally**, BDA fallback | digital PDFs have a text layer pypdf reads for free; scanned/image-only PDFs fall back to BDA OCR |
| Audio, video, `.docx`, scanned PDFs | **Bedrock Data Automation** | one async API for speech-to-text, video transcript + scene description, and document OCR |
| Structured forms (**"form mode"** toggle) | **Amazon Textract** `FORMS`/`TABLES` | key/value pairs from permits and checklists, which BDA renders as prose |

BDA covers the modalities where it genuinely wins (audio, video, scanned docs) rather than
wiring Textract + Transcribe + a multimodal model separately: one set of IAM permissions, one
polling loop, one output format. Images go to a vision model instead of BDA because a
photograph needs describing, not OCR; PDFs are read locally first because most have real text.

Extracted text is chunked and embedded by the existing KB path, and each chunk is tagged
with `source_filename`, `source_modality`, and `extractor` via a `.metadata.json` sidecar,
so answers can cite the original file rather than an opaque chunk ID.

> A KB sync takes a minute or two. If it fails, the extracted text is still saved in S3 and
> the next successful sync picks it up — don't re-upload, you'd pay for the extraction twice.

**What each path costs** (us-east-1 list prices, checked July 2026 — confirm current rates
for your region before relying on these):

| Path | Rate | Notes |
|---|---|---|
| Text / code / spreadsheets / digital PDFs | **$0** | read locally, no API call |
| Cache hit (identical file) | **~$0** | one S3 GET; measured 0.06s vs 6.1s for a fresh extraction |
| Vision model — image | ~$0.001/image | roughly a tenth of a BDA image job at Haiku rates |
| BDA — scanned PDF / `.docx` | ~$0.010/page | fallback for PDFs with no text layer, and Office docs |
| BDA — audio | ~$0.006/min | |
| BDA — video | ~$0.050/min | most expensive per unit; a 10-min video ≈ 50 pages of PDF |
| Textract — form mode | ~$0.050/page | **~5x BDA.** Deliberate exception, not the default |
| Textract — if TABLES added | +~$0.015/page | Textract bills **per feature**, additively |

Two consequences worth internalizing:

- **BDA is ~5x cheaper per page than Textract form mode**, which is why it's the default
  and why the form toggle is opt-in. Leave the toggle off unless you actually need form
  fields — on a 100-page batch that's ~$1 versus ~$5.
- **Extraction results are cached in S3 by content hash.** Re-uploading a file that anyone
  has already processed costs a fraction of a cent instead of a full re-extraction. The key
  covers the extraction settings too, so switching modes correctly re-extracts.

**Cleanup and lifecycle.** Staged originals under `uploads/` and raw BDA JSON under
`bda-output/` hold unredacted content, so `file_ingest.py` **deletes them the instant
extraction finishes** (success or failure) rather than leaving them to age out. The S3
lifecycle (applied by `setup_upload_bucket.py`) is only a backstop for a crashed job:
`uploads/` and `bda-output/` expire after **1 day**, incomplete multipart uploads abort after
**3 days**, and the `extract-cache/` prefix is **kept** because that's the prefix that saves
money. No Standard-IA transition — IA bills a 128 KB minimum per object and requires 30 days
in Standard first, so it would *raise* the bill on the small cache objects and never fire on
the 1-day ones.

### 9b. IAM permissions the app needs

Beyond what `cdk deploy` grants itself, the identity running the chatbot needs:

| Action | Needed for |
| --- | --- |
| `bedrock:Retrieve` | Knowledge Base search (all three specialists) |
| `bedrock:InvokeModel`, `bedrock:ApplyGuardrail` | Bedrock mode + the guardrail |
| `bedrock:ListFoundationModels`, `bedrock:ListInferenceProfiles` | the startup model-access check |
| `bedrock:InvokeDataAutomationAsync`, `bedrock:GetDataAutomationStatus` | file uploads |
| `bedrock:StartIngestionJob` | pushing an uploaded file into the KB |
| `textract:AnalyzeDocument` | structured form mode |
| `s3:PutObject`, `s3:GetObject` on `UPLOAD_BUCKET` and `DOCS_BUCKET` | staging and indexing |
| `dynamodb:GetItem`, `PutItem`, `Query` on `MEMORY_TABLE` | conversation memory |
| `sts:GetCallerIdentity` | startup checks and BDA ARN construction |

`python agents/env_check.py` tells you which of these are missing rather than making you
read a traceback. If you are handing this to someone with tighter access than yours,
have them run it first — it is faster than discovering a gap mid-demo.

---

## 10. Smoke test

After any fresh deploy, run through these and confirm sane behavior (watch the tool-call
trace in the Streamlit status box):

1. Say **"hi"** → friendly greeting, no tool calls (the greeting fast-path).
2. **"what is PPE?"** → a general definition (general questions don't require doc hits).
3. **"what PPE do I need for welding?"** → routes to **Safety**, answers from the manual.
4. **"What do I need to do in my first week as a new hire?"** → routes to **HR**.
5. **"send a jabber to my supervisor saying I'm running late"** → routes to **Operations**,
   trace shows `notify_team_on_jabber`, confirms the send.
6. **"Give me the technical drawings for the submarine hull"** → refused (Guardrail ITAR
   topic block): *"Sorry, I can't help with that request."*
7. **"My name is <your name>"**, then **"what is my name?"** → remembered (same app run).
8. Attach a photo of a printed notice with the paperclip in the input bar and ask about it →
   you get an immediate answer from the extracted text (word count + extractor in the trace),
   and it's indexed in the background. After the sync finishes, ask about it again to exercise
   Knowledge Base retrieval.
9. Attach a source-code file and ask what it does → read locally, no AWS call. Attach a
   spreadsheet and ask about a specific row. Ask a general-knowledge question ("tallest
   building in the world?") → the trace shows a web search.

### Automated tests

```bash
# Offline: routing, validation, BDA output parsing, partition/ARN handling.
# No credentials, no cost.
pytest tests/

# Adds real Textract and Bedrock Data Automation calls, one per modality.
# Costs a few cents and takes ~1 minute.
RUN_LIVE_AWS_TESTS=1 pytest tests/
```

The live tests generate their own fixtures (a rendered notice image, synthesized speech via
macOS `say`, a PDF, a short video) and assert on phrases planted in them — so a pass means
the service actually read the content, not merely that it returned a non-empty string. Any
fixture whose tooling is unavailable skips rather than fails.

---

## 11. Shut everything down (stop the costs)

```bash
python teardown_everything.py
```

Finds everything by name — nothing to edit. Deletes, in order: WAF → Gateway targets +
Gateway → Memory → Cognito → then `cdk destroy --force`. Idempotent: safe to re-run if it
fails partway.

**Confirm OpenSearch is actually gone** (the expensive part):

```bash
aws opensearchserverless list-collections --profile nns-agent
```

The list should be empty (or not contain this project's collection). The billing dashboard
lags ~24h — trust this command, not the dashboard.

---

## 12. Debugging / troubleshooting

### Environment & terminal

| Symptom | Cause | Fix |
|---|---|---|
| Prompt shows `(base)` not `(.venv)` | conda shadowed your venv | Re-run Section 5; `conda deactivate` if needed |
| `NoRegionError` | `region` missing from the profile | Add `region = us-east-1` to `~/.aws/config` (Section 4c) |
| Wrong AWS account / auth errors | `AWS_PROFILE` not exported in this tab | Re-run Section 5 |
| Windows: "running scripts is disabled" | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| "no such file or directory" after pasting | You pasted `< >` brackets literally | Retype without the brackets |
| `ModuleNotFoundError: aws_cdk` during `cdk deploy` | venv not active (cdk.json runs `python3 app.py`) | Activate the venv (Section 5) |

### Deploy & infrastructure

| Symptom | Cause | Fix |
|---|---|---|
| `setup_gateway.py` crashes ("CREATING status", propagation timeout) | AWS async timing | **Run it again** — it's idempotent and resumes/repairs |
| WAF: `WAFUnavailableEntityException` | Known flaky association | Optional — skip it; Cognito already protects the Gateway |
| CDK deploy fails on the Guardrail | `PROMPT_ATTACK` filter needs `output_strength="NONE"` | Already handled in the stack; keep it NONE if editing (it's input-only) |
| `AccessDenied` / model not available on Bedrock | Model access not granted | Console → Bedrock → Model access (Section 4d) |
| Model ID "end of life" / not found | Bedrock retired a model | `aws bedrock list-inference-profiles --profile nns-agent`, export `BEDROCK_MODEL_ID` or update `model_config.py` |
| `RuntimeError: No Cognito client secret found` | `agents/gateway_secrets.py` missing (it's gitignored) | Create it per Step 6, or export `COGNITO_CLIENT_SECRET` |

### File uploads & configuration

**Start here for any of these: `python agents/env_check.py`.** It names the specific
variable, permission, or mismatch rather than making you read a stack trace.

| Symptom | Cause | Fix |
|---|---|---|
| App refuses to start, lists config problems | Startup validation caught a missing/blank value | Fill the named variables in `.env`; the report says which |
| `InvalidClientTokenId`, and refreshing credentials doesn't help | `AWS_REGION` is `us-gov-*` but the profile is commercial (or vice versa) | Partition mismatch — GovCloud needs GovCloud credentials. `env_check.py` says so explicitly |
| A `.env` value seems ignored, code uses its default | You set the variable but left it blank — a blank line means "no value", so it isn't exported | Give it a real value, or delete the line |
| `Starting knowledge base sync failed (ResourceNotFoundException)` | `KNOWLEDGE_BASE_ID`/`DATA_SOURCE_ID` are stale — they change on every teardown/rebuild | Re-paste from the latest `cdk deploy` outputs. The extracted text is already in S3; don't re-upload |
| Upload rejected: "UPLOAD_BUCKET is not set" | BDA reads its input from S3, so a staging bucket is required | Create one and set `UPLOAD_BUCKET` ([§9a](#9a-uploading-images-pdfs-audio-and-video)) |
| "Extraction succeeded but returned no text" | The file has nothing readable — e.g. a photo of scenery rather than a document | Expected. Use a file with text or speech in it |
| Bedrock Data Automation unreachable / access denied | Missing BDA permissions, or the region has no BDA | BDA is **GovCloud US-West only**; in commercial use a supported region ([§9b](#9b-iam-permissions-the-app-needs)) |
| Uploaded file indexed but answers don't mention it | KB sync hadn't finished | Wait a minute or two; the sidebar prints the sync job ID |
| Edited `file_ingest.py` but the app behaves the same | Streamlit hot-reloads the page script, not imported modules | Restart `streamlit run chat_ui.py` |

### Agent behavior

| Symptom | Cause | Fix |
|---|---|---|
| Empty answers / "couldn't find documentation" for everything | Ingestion not COMPLETE, or stale `KNOWLEDGE_BASE_ID` | Re-check Step 2; compare `.env` against the CDK outputs |
| Gateway tool calls fail (401/auth) | Stale Gateway/Cognito values or secret after a rebuild | Re-run `setup_gateway.py`, re-paste Step 6 values |
| Answers derail onto random topics mid-chat (Ollama) | Small-model drift; long chats | Restart the app (each run starts a clean memory session); keep `temperature=0`; or switch to Bedrock |
| Assistant forgets your name after many turns | Memory replays only the last 5 turns | Raise `k=5` in `agents/memory_hook.py` |
| Doesn't remember you across app restarts (Ollama) | Long-term memory is Bedrock-only — llama3.1:8b derails on injected memory and invents facts during extraction | `export MODEL_PROVIDER=bedrock`; see the `memory_manager` comment in `agents/supervisor.py` |
| Raw JSON like `{"name": "ask_operations"...}` as the answer | Small model emits the tool call as text | Already auto-recovered in `supervisor.py`; if you see it, check that code path survived your edits |
| Names/emails show as `{EMAIL}`/`{PHONE}` | Guardrail PII anonymization (by design) | Edit the PII list in the CDK stack + `cdk deploy` (takes effect immediately — guardrail runs as DRAFT) |
| Everything refused ("Sorry, I can't help with that") | Guardrail misconfig or stale `GUARDRAIL_ID` | Compare `GUARDRAIL_ID` with the CDK outputs; test with `python -c` ApplyGuardrail |
| Ollama: connection refused / hangs | Ollama not running or model not pulled | `ollama serve` (or open the app), `ollama pull llama3.1:8b` |
| Streamlit tool trace not updating live | Widgets only update from the main thread | The thread + `queue.Queue` + `st.status()` pattern in `chat_ui.py` handles this — don't update widgets from the worker thread |

### General debugging habits for this repo

- **One command at a time** — read each script's output before running the next.
- **`grep` after editing config:** `grep -rn "aws_config\|KNOWLEDGE_BASE_ID\|MEMORY_TABLE" agents/`
  — resource IDs should appear only in `agents/aws_config.py`.
- **When agent answers look wrong, check the tool trace first** (the Streamlit status box):
  wrong routing, no tool call, or a failing tool each point to a different fix above.

---

*Fixed names the scripts search by (do not change casually): Gateway `NnsCompanyToolsGateway`,
user pool `nns-agentcore-gateway-pool`, client `nns-agentcore-gateway-client`, memory table
`NnsChatbotMemory`, Web ACL `nns-gateway-web-acl`, stack
`NnsAgenticRagChatbotStack`. Region: `us-east-1`.*
