"""
Sequential Workflow with MAF and Microsoft Foundry

This script demonstrates a simple sequential workflow:
1. Researcher Agent gathers information on a topic.
2. Writer Agent writes an essay based on the research.

To run:
    python sequential_workflow.py
"""

import os
import asyncio
import logging
import random
from dotenv import load_dotenv
from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
    WorkflowViz,
)
from agent_framework import ChatAgent
from agent_framework.azure import AzureAIClient
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import AzureCliCredential
from agent_framework.devui import serve

# Load deployment configuration from .env instead of hard-coding environment-
# specific values in the script.
load_dotenv()
project_endpoint = os.getenv("AI_FOUNDRY_PROJECT_ENDPOINT") or os.getenv("FOUNDRY_PROJECT_ENDPOINT")
model = os.getenv("AI_FOUNDRY_DEPLOYMENT_NAME") or os.getenv("MODEL_DEPLOYMENT_NAME")
azure_tenant_id = os.getenv("AZURE_TENANT_ID")

print("Project Endpoint:", project_endpoint)
print("Model:", model)

if not project_endpoint or not model:
    missing = []
    if not project_endpoint:
        missing.append("AI_FOUNDRY_PROJECT_ENDPOINT or FOUNDRY_PROJECT_ENDPOINT")
    if not model:
        missing.append("AI_FOUNDRY_DEPLOYMENT_NAME or MODEL_DEPLOYMENT_NAME")
    raise ValueError("Missing required .env value(s): " + ", ".join(missing))


async def run_agent_with_retry(agent: ChatAgent, message, *, max_tokens: int = 800):
    """Run an agent and wait/retry when Azure returns a transient rate limit."""
    max_attempts = int(os.getenv("AGENT_RETRY_ATTEMPTS", "5"))
    for attempt in range(max_attempts):
        try:
            return await agent.run(message, max_tokens=max_tokens)
        except Exception as exc:
            error_text = str(exc).lower()
            is_rate_limit = (
                "429" in error_text
                or "too many requests" in error_text
                or "rate_limit" in error_text
                or "rate limit" in error_text
            )
            if not is_rate_limit or attempt == max_attempts - 1:
                raise

            delay = min(30, (2 ** attempt) + random.uniform(0.25, 1.25))
            print(f"Rate limit hit. Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)


async def create_agent(agent_name: str, agent_instructions: str) -> ChatAgent:
    """Create one Foundry agent with its own conversation context."""
    # AzureCliCredential authenticates with the identity from `az login`.
    # Set AZURE_TENANT_ID when the active Azure CLI account can access more
    # than one tenant. This prevents a token for the wrong tenant from being
    # sent to the Foundry project.
    credential = AzureCliCredential(tenant_id=azure_tenant_id) if azure_tenant_id else AzureCliCredential()
    # The project client provides access to Foundry project resources.
    project_client = AIProjectClient(
        endpoint=project_endpoint,
        credential=credential
    )
    # A conversation preserves the messages belonging to this agent.
    openai_client = project_client.get_openai_client()
    conversation = await openai_client.conversations.create()
    conversation_id = conversation.id
    print("Conversation ID:", conversation_id)

    # Bind the selected model deployment to the new conversation.
    chat_client = AzureAIClient(
        project_client=project_client,
        conversation_id=conversation_id,
        model_deployment_name=model
    )

    agent = chat_client.create_agent(
        name=agent_name,
        instructions=agent_instructions,
    )
    print(f"{agent_name} Agent created successfully!")
    return agent

# Executors are workflow nodes. Their handlers transform an incoming message
# into the message consumed by the next stage.
class ResearcherExecutor(Executor):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, query: str, ctx: WorkflowContext[str]) -> None:
        response = await run_agent_with_retry(self.agent, query, max_tokens=700)
        # send_message passes an intermediate result to the next executor.
        await ctx.send_message(str(response))

class WriterExecutor(Executor):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, research_data: str, ctx: WorkflowContext[str]) -> None:
        response = await run_agent_with_retry(self.agent, research_data, max_tokens=900)
        # yield_output publishes the final workflow result to DevUI.
        await ctx.yield_output(str(response))

async def build_workflow():
    """Create the agents and connect them in a researcher-to-writer pipeline."""
    # Each instruction prompt gives an agent one clear responsibility.
    researcher_agent = await create_agent(
        agent_name="Researcher-Agent",
        agent_instructions=(
            "You are a knowledgeable researcher. Gather useful facts and insights on the topic. "
            "Keep the research summary concise, practical, and under 300 words."
        )
    )
    writer_agent = await create_agent(
        agent_name="Writer-Agent",
        agent_instructions=(
            "You are a clear writer. Turn the research into a coherent short essay. "
            "Keep the final essay focused and under 500 words."
        )
    )

    # Executor IDs make workflow traces and diagrams easier to read.
    researcher_executor = ResearcherExecutor(researcher_agent, id="ResearcherExecutor")
    writer_executor = WriterExecutor(writer_agent, id="WriterExecutor")

    # A single directed edge creates strict sequential execution:
    # the writer starts only after the researcher sends its result.
    workflow = (
        WorkflowBuilder(
            name="Sequential Research & Writing Workflow",
            description="A two-step workflow: research a topic, then write an essay."
        )
        .set_start_executor(researcher_executor)
        .add_edge(researcher_executor, writer_executor)
        .build()
    )

    # Mermaid text is a portable representation of the workflow graph.
    viz = WorkflowViz(workflow)
    mermaid_content = viz.to_mermaid()
    print("Mermaid Diagram:\n", mermaid_content)

    return workflow

def main():
    """Launch the sequential workflow in DevUI."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    devui_port = int(os.getenv("DEVUI_SEQUENTIAL_PORT") or os.getenv("DEVUI_PORT", "8090"))
    logger.info("Starting Sequential Research & Writing Workflow")
    logger.info("Available at: http://localhost:%s", devui_port)
    logger.info("Entity ID: workflow_sequential_research_writer")

    # asyncio.run performs async setup before starting the synchronous server.
    workflow = asyncio.run(build_workflow())
    # DevUI exposes the workflow in a browser; tracing makes each stage
    # observable for debugging and learning.
    serve(entities=[workflow], port=devui_port, auto_open=True, tracing_enabled=True)

if __name__ == "__main__":
    # Prevent server startup when this module is imported elsewhere.
    main()


#######User Promts - Sample Examples:
# (1) Write a short essay on how artificial intelligence is changing education.
# (2) Write an essay about how cloud computing helps modern companies scale faster.
# (3) Research the impact of electric vehicles on urban transportation and write a clear essay.
# (4) Explain the importance of cybersecurity for small businesses in a short essay.
