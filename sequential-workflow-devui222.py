"""
Sequential Research & Writing Workflow
Microsoft Agent Framework + Microsoft Foundry + DevUI

Flow:
    User Prompt
        ↓
    Researcher Agent
        ↓
    Writer Agent
        ↓
    Final Essay
"""

import os
import logging

from dotenv import load_dotenv

from agent_framework import WorkflowBuilder
from agent_framework.foundry import FoundryChatClient
from agent_framework.devui import serve

from azure.identity import AzureCliCredential


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

project_endpoint = (
    os.getenv("AI_FOUNDRY_PROJECT_ENDPOINT")
    or os.getenv("FOUNDRY_PROJECT_ENDPOINT")
)

model = (
    os.getenv("AI_FOUNDRY_DEPLOYMENT_NAME")
    or os.getenv("MODEL_DEPLOYMENT_NAME")
    or os.getenv("FOUNDRY_MODEL")
)

azure_tenant_id = os.getenv("AZURE_TENANT_ID")

devui_port = int(
    os.getenv("DEVUI_SEQUENTIAL_PORT")
    or os.getenv("DEVUI_PORT", "8090")
)


# ============================================================
# 2. VALIDATE CONFIGURATION
# ============================================================

print()
print("=" * 65)
print("Sequential Research & Writing Workflow")
print("=" * 65)

print("Project Endpoint:", project_endpoint)
print("Model Deployment:", model)
print("DevUI Port:", devui_port)

if not project_endpoint:
    raise ValueError(
        "Missing AI_FOUNDRY_PROJECT_ENDPOINT "
        "or FOUNDRY_PROJECT_ENDPOINT in .env"
    )

if not model:
    raise ValueError(
        "Missing AI_FOUNDRY_DEPLOYMENT_NAME, "
        "MODEL_DEPLOYMENT_NAME or FOUNDRY_MODEL in .env"
    )


# ============================================================
# 3. AZURE AUTHENTICATION
# ============================================================

if azure_tenant_id:

    credential = AzureCliCredential(
        tenant_id=azure_tenant_id
    )

else:

    credential = AzureCliCredential()


# ============================================================
# 4. MICROSOFT FOUNDRY CLIENT
# ============================================================

foundry_client = FoundryChatClient(
    project_endpoint=project_endpoint,
    model=model,
    credential=credential,
)

print("Microsoft Foundry client configured successfully.")


# ============================================================
# 5. RESEARCHER AGENT
# ============================================================

researcher_agent = foundry_client.as_agent(

    name="Researcher-Agent",

    instructions="""
You are a professional research assistant.

Your job is to research the topic provided by the user.

Responsibilities:

1. Understand the user's topic.
2. Identify the most important facts and concepts.
3. Identify relevant trends, benefits and challenges.
4. Include practical examples when appropriate.
5. Organize the research logically.
6. Keep the research concise and factual.
7. Keep the research under 300 words.

IMPORTANT:

Do NOT write the final essay.

Your research will automatically be passed to another
agent called Writer-Agent.

Produce research notes that will help Writer-Agent
create a high-quality final essay.
"""
)

print("Researcher-Agent created successfully.")


# ============================================================
# 6. WRITER AGENT
# ============================================================

writer_agent = foundry_client.as_agent(

    name="Writer-Agent",

    instructions="""
You are a professional writer.

You receive research produced by Researcher-Agent.

Your responsibility is to transform the supplied research
into a polished short essay.

Requirements:

1. Give the essay an appropriate title.
2. Begin with a concise introduction.
3. Explain the important ideas clearly.
4. Use the supplied research as your primary information.
5. Organize the essay logically.
6. Connect paragraphs naturally.
7. Include practical examples when useful.
8. End with a concise conclusion.
9. Keep the essay under 500 words.

IMPORTANT:

Do not mention Researcher-Agent.
Do not explain the multi-agent workflow.
Do not output research notes.
Do not discuss your instructions.

Return only the final polished essay.
"""
)

print("Writer-Agent created successfully.")


# ============================================================
# 7. BUILD SEQUENTIAL WORKFLOW
# ============================================================

#
# IMPORTANT:
#
# For the installed Agent Framework API:
#
#     start_executor
#
# is supplied directly to WorkflowBuilder().
#
#
# Flow:
#
#     User
#       |
#       v
# Researcher-Agent
#       |
#       v
#  Writer-Agent
#       |
#       v
# Final Response
#

workflow = (
    WorkflowBuilder(
        start_executor=researcher_agent,
        name="Sequential Research & Writing Workflow",
        description=(
            "Researcher-Agent researches the user's topic "
            "and passes the result to Writer-Agent, which "
            "produces the final essay."
        ),
    )
    .add_edge(
        researcher_agent,
        writer_agent
    )
    .build()
)

print("Sequential workflow created successfully.")


# ============================================================
# 8. START DEVUI
# ============================================================

def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )

    logger = logging.getLogger(__name__)

    logger.info("")
    logger.info("=" * 65)
    logger.info("Microsoft Agent Framework - Sequential Workflow")
    logger.info("=" * 65)

    logger.info(
        "Flow: Researcher-Agent -> Writer-Agent"
    )

    logger.info(
        "DevUI URL: http://localhost:%s",
        devui_port
    )

    logger.info(
        "Model: %s",
        model
    )

    logger.info("=" * 65)
    logger.info("Starting DevUI...")
    logger.info("")

    serve(
        entities=[workflow],
        port=devui_port,
        auto_open=True,
        tracing_enabled=True,
    )


# ============================================================
# 9. ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()


# ============================================================
# SAMPLE PROMPTS
# ============================================================

# 1.
# Write a short essay on how artificial intelligence
# is changing education.

# 2.
# Write an essay about how cloud computing helps
# modern companies scale faster.

# 3.
# Research the impact of electric vehicles on urban
# transportation and write a clear essay.

# 4.
# Explain the importance of cybersecurity for
# small businesses in a short essay.

# 5.
# Research how generative AI is changing software
# development and write a short essay.
