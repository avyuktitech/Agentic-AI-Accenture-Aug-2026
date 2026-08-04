"""Orchestrator Agent Workflow Sample for DevUI.

This sample demonstrates an orchestrator-led multi-agent workflow using the
Microsoft Agent Framework. The orchestrator turns a user request into a clear
brief, sends it to three specialist agents, and then combines their outputs.
"""

import os
import asyncio
import logging
import random
import socket
from typing import Any
from dotenv import load_dotenv
from agent_framework import (
    Executor,
    WorkflowBuilder,
    WorkflowContext,
    handler,
    WorkflowViz,
)
from azure.ai.projects.aio import AIProjectClient
from azure.identity.aio import AzureCliCredential
from agent_framework.devui import serve

# Load Azure AI Foundry configuration from .env. Both naming styles are
# supported so this file works with the existing project configuration.
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


def ensure_port_available(host: str, port: int) -> None:
    """Fail before creating Azure resources if the DevUI port is already busy."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        if sock.connect_ex((host, port)) == 0:
            raise RuntimeError(
                f"Port {port} is already in use. Stop the existing server or set "
                f"DEVUI_ORCHESTRATOR_PORT to another port, for example 8095."
            )


async def run_agent_with_retry(agent: Any, message, *, max_tokens: int = 800):
    """Run an agent and retry transient Azure rate-limit failures."""
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


class LazyFoundryAgent:
    """Creates and uses the Foundry client inside DevUI's request event loop."""

    def __init__(self, agent_name: str, agent_instructions: str):
        self.agent_name = agent_name
        self.agent_instructions = agent_instructions
        self.credential: AzureCliCredential | None = None
        self.project_client: AIProjectClient | None = None
        self.openai_client: Any | None = None

    async def run(self, message: str, *, max_tokens: int):
        if self.openai_client is None:
            self.credential = (
                AzureCliCredential(tenant_id=azure_tenant_id)
                if azure_tenant_id else AzureCliCredential()
            )
            self.project_client = AIProjectClient(endpoint=project_endpoint, credential=self.credential)
            self.openai_client = self.project_client.get_openai_client()
            print(f"{self.agent_name} client initialized in the DevUI event loop.")

        response = await self.openai_client.responses.create(
            model=model,
            input=f"Instructions:\n{self.agent_instructions}\n\nUser request:\n{message}",
        )
        if not response.output_text:
            raise RuntimeError(f"{self.agent_name} returned an empty response")
        return response.output_text


def create_agent(agent_name: str, agent_instructions: str) -> LazyFoundryAgent:
    return LazyFoundryAgent(agent_name, agent_instructions)


class OrchestratorBriefExecutor(Executor):
    """Turns the user's raw request into a specialist-ready task brief."""

    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, user_request: str, ctx: WorkflowContext[str]) -> None:
        response = await run_agent_with_retry(self.agent, user_request, max_tokens=500)
        await ctx.send_message(str(response))


class ResearchSpecialistExecutor(Executor):
    """Provides factual background and important context."""

    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, task_brief: str, ctx: WorkflowContext[str]) -> None:
        response = await run_agent_with_retry(self.agent, task_brief, max_tokens=550)
        await ctx.send_message(str(response))


class StrategySpecialistExecutor(Executor):
    """Suggests practical actions, structure, and implementation steps."""

    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, task_brief: str, ctx: WorkflowContext[str]) -> None:
        response = await run_agent_with_retry(self.agent, task_brief, max_tokens=550)
        await ctx.send_message(str(response))


class RiskSpecialistExecutor(Executor):
    """Identifies risks, tradeoffs, and mitigations."""

    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, task_brief: str, ctx: WorkflowContext[str]) -> None:
        response = await run_agent_with_retry(self.agent, task_brief, max_tokens=500)
        await ctx.send_message(str(response))


class OrchestratorFinalExecutor(Executor):
    """Combines specialist outputs into the final response."""

    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, specialist_results: list[str], ctx: WorkflowContext[str]) -> None:
        delay_seconds = int(os.getenv("ORCHESTRATOR_FINAL_DELAY_SECONDS", "10"))
        if delay_seconds > 0:
            print(f"Waiting {delay_seconds}s before final orchestrator call to avoid rate limits...")
            await asyncio.sleep(delay_seconds)

        final_brief = "\n\n".join(
            f"Specialist result {index}:\n{result}"
            for index, result in enumerate(specialist_results, start=1)
        )
        response = await run_agent_with_retry(self.agent, final_brief, max_tokens=900)
        await ctx.yield_output(str(response))


def build_workflow():
    """Create agents and connect them into an orchestrator-led workflow."""
    orchestrator_agent = create_agent(
        agent_name="Orchestrator-Agent",  #Agent-1
        agent_instructions=(
            "You are the orchestrator. First, convert the user request into a clear task brief "
            "for specialist agents. Later, combine specialist outputs into one concise final answer "
            "with clear recommendations. Avoid asking follow-up questions unless essential."
        ),
    )
    research_agent = create_agent(
        agent_name="Research-Specialist-Agent",  #Agent-2
        agent_instructions=(
            "You are a research specialist. Extract facts, context, assumptions, and useful examples "
            "from the task brief. Keep your response under 250 words."
        ),
    )
    strategy_agent = create_agent(
        agent_name="Strategy-Specialist-Agent", #Agent-3
        agent_instructions=(
            "You are a strategy specialist. Propose a practical plan, steps, and priorities based on "
            "the task brief. Keep your response under 250 words."
        ),
    )
    risk_agent = create_agent(
        agent_name="Risk-Specialist-Agent",  #Agent-4
        agent_instructions=(
            "You are a risk specialist. Identify risks, tradeoffs, missing assumptions, and mitigations. "
            "Keep your response under 220 words."
        ),
    )

    orchestrator_brief_executor = OrchestratorBriefExecutor(
        orchestrator_agent,
        id="OrchestratorBrief",
    )
    research_executor = ResearchSpecialistExecutor(
        research_agent,
        id="ResearchSpecialist",
    )
    strategy_executor = StrategySpecialistExecutor(
        strategy_agent,
        id="StrategySpecialist",
    )
    risk_executor = RiskSpecialistExecutor(
        risk_agent,
        id="RiskSpecialist",
    )
    orchestrator_final_executor = OrchestratorFinalExecutor(
        orchestrator_agent,
        id="OrchestratorFinal",
    )

    workflow = (
        WorkflowBuilder(
            name="Orchestrator Agent Workflow",
            description="An orchestrator coordinates research, strategy, and risk specialists.",
        )
        .set_start_executor(orchestrator_brief_executor)
        .add_fan_out_edges(orchestrator_brief_executor, [
            research_executor,
            strategy_executor,
            risk_executor,
        ])
        .add_fan_in_edges([
            research_executor,
            strategy_executor,
            risk_executor,
        ], orchestrator_final_executor)
        .build()
    )

    viz = WorkflowViz(workflow)
    mermaid_content = viz.to_mermaid()
    print("Mermaid Diagram:\n", mermaid_content)

    return workflow


def main():
    """Launch the orchestrator workflow in DevUI."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    devui_port = int(os.getenv("DEVUI_ORCHESTRATOR_PORT") or os.getenv("DEVUI_PORT", "8094"))
    ensure_port_available("127.0.0.1", devui_port)
    logger.info("Starting Orchestrator Agent Workflow")
    logger.info("Available at: http://localhost:%s", devui_port)
    logger.info("Entity ID: workflow_orchestrator_agent")

    workflow = build_workflow()
    serve(entities=[workflow], port=devui_port, auto_open=True, tracing_enabled=True)


if __name__ == "__main__":
    main()


# Sample DevUI test prompts:
#
# Create a practical plan for a small company to adopt generative AI safely over the next 90 days.
#
# Help a college build a student support chatbot. Include research, implementation strategy, and risks.
#
# Design a launch plan for an online grocery delivery service in a new city.
#
# Prepare a business proposal for using AI to improve customer service in a bank.
