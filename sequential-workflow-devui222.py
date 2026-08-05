"""
Sequential Workflow with Microsoft Agent Framework + Microsoft Foundry + DevUI

Flow:
    User Prompt
        ↓
    Researcher Agent
        ↓
    Writer Agent
        ↓
    Final Essay

Requirements:
    - Python 3.10+
    - Microsoft Agent Framework
    - Microsoft Foundry project
    - Azure CLI authentication
    - .env configuration

Run:
    python sequential-workflow-devui.py
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

# Support both your existing variable names and the newer naming style.
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

print("=" * 70)
print("Sequential Research & Writing Workflow")
print("=" * 70)

print("Project Endpoint:", project_endpoint)
print("Model Deployment:", model)
print("DevUI Port:", devui_port)

if not project_endpoint or not model:

    missing = []

    if not project_endpoint:
        missing.append(
            "AI_FOUNDRY_PROJECT_ENDPOINT or FOUNDRY_PROJECT_ENDPOINT"
        )

    if not model:
        missing.append(
            "AI_FOUNDRY_DEPLOYMENT_NAME, MODEL_DEPLOYMENT_NAME "
            "or FOUNDRY_MODEL"
        )

    raise ValueError(
        "Missing required .env value(s): " + ", ".join(missing)
    )


# ============================================================
# 3. AZURE AUTHENTICATION
# ============================================================

# AzureCliCredential uses the identity authenticated through:
#
#     az login
#
# If AZURE_TENANT_ID is present, explicitly use that tenant.

if azure_tenant_id:

    credential = AzureCliCredential(
        tenant_id=azure_tenant_id
    )

else:

    credential = AzureCliCredential()


# ============================================================
# 4. CREATE MICROSOFT FOUNDRY CHAT CLIENT
# ============================================================

# FoundryChatClient is the current Microsoft Agent Framework
# client for models deployed inside a Microsoft Foundry project.

foundry_client = FoundryChatClient(
    project_endpoint=project_endpoint,
    model=model,
    credential=credential,
)

print("Microsoft Foundry client configured successfully.")


# ============================================================
# 5. CREATE RESEARCHER AGENT
# ============================================================

researcher_agent = foundry_client.as_agent(

    name="Researcher-Agent",

    instructions="""
You are a knowledgeable research assistant.

Your responsibility is to research the topic provided by the user.

Tasks:

1. Identify the main subject of the request.
2. Gather the most useful facts, concepts, trends, benefits,
   challenges, and examples relevant to the topic.
3. Organize the information clearly.
4. Provide enough context for another AI agent to write
   a high-quality short essay.
5. Avoid unnecessary repetition.
6. Keep the research summary under 300 words.

Do NOT write the final essay.

Your output will be passed to a Writer Agent.
""",
)

print("Researcher-Agent created successfully.")


# ============================================================
# 6. CREATE WRITER AGENT
# ============================================================

writer_agent = foundry_client.as_agent(

    name="Writer-Agent",

    instructions="""
You are a professional content writer.

You will receive research produced by another AI agent.

Your responsibility is to transform that research into a
clear, coherent, well-structured short essay.

Requirements:

1. Give the essay a meaningful title.
2. Start with a short introduction.
3. Explain the important ideas logically.
4. Use the supplied research as the foundation.
5. Connect ideas naturally.
6. Include practical examples where appropriate.
7. Finish with a concise conclusion.
8. Keep the final essay under 500 words.
9. Do not mention the Researcher Agent.
10. Do not describe the workflow.

Return only the polished final essay.
""",
)

print("Writer-Agent created successfully.")


# ============================================================
# 7. BUILD SEQUENTIAL WORKFLOW
# ============================================================

# Agent Framework allows agents to participate directly
# as workflow executors.
#
# The directed edge creates:
#
# User
#   ↓
# Researcher-Agent
#   ↓
# Writer-Agent
#   ↓
# Final Output

workflow = (

    WorkflowBuilder(
        name="Sequential Research & Writing Workflow",
        description=(
            "A two-agent sequential workflow where a Researcher "
            "collects information and a Writer converts the "
            "research into a short essay."
        ),
    )

    # Register the Researcher agent.
    .add_agent(
        researcher_agent,
        output_response=False,
    )

    # Register the Writer agent.
    #
    # output_response=True means the Writer's response becomes
    # the final workflow output visible to DevUI.
    .add_agent(
        writer_agent,
        output_response=True,
    )

    # Researcher is the first agent.
    .set_start_executor(
        researcher_agent
    )

    # Researcher output flows directly to Writer.
    .add_edge(
        researcher_agent,
        writer_agent
    )

    .build()
)

print("Sequential workflow created successfully.")


# ============================================================
# 8. MAIN
# ============================================================

def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s"
    )

    logger = logging.getLogger(__name__)

    logger.info("")
    logger.info("=" * 70)
    logger.info("Starting Sequential Research & Writing Workflow")
    logger.info("=" * 70)

    logger.info(
        "DevUI: http://localhost:%s",
        devui_port
    )

    logger.info(
        "Workflow: Sequential Research & Writing Workflow"
    )

    logger.info(
        "Flow: Researcher-Agent -> Writer-Agent"
    )

    logger.info("=" * 70)

    # --------------------------------------------------------
    # Start Microsoft Agent Framework DevUI
    # --------------------------------------------------------
    #
    # entities=[workflow]
    #
    # registers the workflow directly with DevUI.
    #
    # auto_open=True
    #
    # automatically opens the browser.
    #
    # tracing_enabled=True
    #
    # enables workflow/agent traces for observability.

    serve(
        entities=[workflow],
        port=devui_port,
        auto_open=True,
        tracing_enabled=True,
    )


# ============================================================
# 9. APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()


# ============================================================
# SAMPLE USER PROMPTS
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
