import aws_cdk as core
import aws_cdk.assertions as assertions

from nns_agentic_rag_chatbot.nns_agentic_rag_chatbot_stack import NnsAgenticRagChatbotStack

# example tests. To run these tests, uncomment this file along with the example
# resource in nns_agentic_rag_chatbot/nns_agentic_rag_chatbot_stack.py
def test_sqs_queue_created():
    app = core.App()
    stack = NnsAgenticRagChatbotStack(app, "nns-agentic-rag-chatbot")
    template = assertions.Template.from_stack(stack)

#     template.has_resource_properties("AWS::SQS::Queue", {
#         "VisibilityTimeout": 300
#     })
