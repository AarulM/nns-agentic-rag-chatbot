#!/usr/bin/env python3
import aws_cdk as cdk

from nns_agentic_rag_chatbot.nns_agentic_rag_chatbot_stack import NnsAgenticRagChatbotStack

app = cdk.App()
NnsAgenticRagChatbotStack(app, "NnsAgenticRagChatbotStack")
app.synth()
