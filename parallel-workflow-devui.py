"""Vacation Planning Workflow Sample for DevUI.

This sample demonstrates a multi-agent workflow for vacation planning using the Microsoft Agent Framework.
Agents include: Location Picker, Destination Recommender, Weather, Cuisine Suggestion, and Itinerary Planner.
"""

import os
import asyncio
import logging
import random
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

# Configuration is kept outside the source code in .env. This avoids hard-coding
# deployment-specific values and lets the workflow run in other environments.
load_dotenv()
project_endpoint = os.getenv("AI_FOUNDRY_PROJECT_ENDPOINT") or os.getenv("FOUNDRY_PROJECT_ENDPOINT")
model = os.getenv("AI_FOUNDRY_DEPLOYMENT_NAME") or os.getenv("MODEL_DEPLOYMENT_NAME")
azure_tenant_id = os.getenv("AZURE_TENANT_ID")

print("Project Endpoint: ", project_endpoint)
print("Model: ", model)

if not project_endpoint or not model:
    missing = []
    if not project_endpoint:
        missing.append("AI_FOUNDRY_PROJECT_ENDPOINT or FOUNDRY_PROJECT_ENDPOINT")
    if not model:
        missing.append("AI_FOUNDRY_DEPLOYMENT_NAME or MODEL_DEPLOYMENT_NAME")
    raise ValueError("Missing required .env value(s): " + ", ".join(missing))


async def run_agent_with_retry(agent: Any, message, *, max_tokens: int = 800):
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

class LazyFoundryAgent:
    """Calls the Foundry Responses API from the DevUI request loop."""

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
                if azure_tenant_id
                else AzureCliCredential()
            )
            self.project_client = AIProjectClient(
                endpoint=project_endpoint,
                credential=self.credential,
            )
            self.openai_client = self.project_client.get_openai_client()
            print(f"{self.agent_name} client initialized in the DevUI event loop.")

        response = await self.openai_client.responses.create(
            model=model,
            # This deployment returns empty content when the Responses API's
            # `instructions` field is used. Put the role guidance in the input
            # instead, which is supported reliably by the deployed model.
            input=f"Instructions:\n{self.agent_instructions}\n\nUser request:\n{message}",
        )
        if not response.output_text:
            raise RuntimeError(f"{self.agent_name} returned an empty response")
        return response.output_text


def create_agent(agent_name: str, agent_instructions: str) -> LazyFoundryAgent:
    """Return a lazily initialized Foundry agent for the workflow executor."""
    return LazyFoundryAgent(agent_name, agent_instructions)

# Executors are workflow nodes. A handler receives the upstream node's message,
# invokes its specialist agent, and publishes the result through the context.
class LocationSelectorExecutor(Executor):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, user_query: str, ctx: WorkflowContext[dict[str, str]]) -> None:
        response = await run_agent_with_retry(self.agent, user_query, max_tokens=500)
        # Keep the original request with the selected location. The downstream
        # agents and the final planner need both pieces of context.
        await ctx.send_message({
            "user_query": user_query,
            "location": str(response),
        })

class DestinationRecommenderExecutor(Executor):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, trip_context: dict[str, str], ctx: WorkflowContext[dict[str, str]]) -> None:
        prompt = (
            f"Original trip request:\n{trip_context['user_query']}\n\n"
            f"Selected destination:\n{trip_context['location']}"
        )
        response = await run_agent_with_retry(self.agent, prompt, max_tokens=500)
        await ctx.send_message({**trip_context, "specialist": "Destination", "result": str(response)})

class WeatherExecutor(Executor):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, trip_context: dict[str, str], ctx: WorkflowContext[dict[str, str]]) -> None:
        prompt = (
            f"Original trip request:\n{trip_context['user_query']}\n\n"
            f"Selected destination:\n{trip_context['location']}"
        )
        response = await run_agent_with_retry(self.agent, prompt, max_tokens=450)
        await ctx.send_message({**trip_context, "specialist": "Weather", "result": str(response)})

class CuisineSuggestionExecutor(Executor):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, trip_context: dict[str, str], ctx: WorkflowContext[dict[str, str]]) -> None:
        prompt = (
            f"Original trip request:\n{trip_context['user_query']}\n\n"
            f"Selected destination:\n{trip_context['location']}"
        )
        response = await run_agent_with_retry(self.agent, prompt, max_tokens=500)
        await ctx.send_message({**trip_context, "specialist": "Cuisine", "result": str(response)})

class ItineraryPlannerExecutor(Executor):
    def __init__(self, agent, **kwargs):
        super().__init__(**kwargs)
        self.agent = agent

    @handler
    async def handle(self, results: list[dict[str, str]], ctx: WorkflowContext[str]) -> None:
        # Fan-in supplies a list containing all three parallel branch results.
        delay_seconds = int(os.getenv("ITINERARY_DELAY_SECONDS", "10"))
        if delay_seconds > 0:
            print(f"Waiting {delay_seconds}s before final itinerary call to avoid rate limits...")
            await asyncio.sleep(delay_seconds)
        # Fan-in supplies a list, whereas ChatAgent.run expects a text prompt.
        # Include the original request and location even if a specialist call
        # happens to return empty content.
        trip_context = results[0] if results else {}
        notes = "\n\n".join(
            f"{item.get('specialist', 'Specialist')} note:\n{item.get('result') or '(No note returned)'}"
            for item in results
        )
        planning_brief = (
            f"Original trip request:\n{trip_context.get('user_query', '')}\n\n"
            f"Selected destination:\n{trip_context.get('location', '')}\n\n"
            f"Specialist notes:\n{notes}\n\n"
            "Create the requested itinerary now. Do not ask the user to paste notes or provide more details; "
            "make reasonable assumptions when a specialist note is unavailable."
        )
        response = await run_agent_with_retry(self.agent, planning_brief, max_tokens=900)
        # yield_output marks this value as the workflow's final result.
        await ctx.yield_output(str(response))

def build_workflow():
    """Create the specialist agents and connect them as a parallel workflow."""
    # Agent instructions establish a focused role for each workflow stage.
    location_picker_agent = create_agent(
        agent_name="Location-Picker-Agent", # AGENT-1
        agent_instructions=(
            "Choose exactly one concrete vacation destination from the user's request. "
            "State the city/region and country, then give a short reason it fits. "
            "Use reasonable assumptions for missing details; never ask a follow-up question. "
            "Keep the answer under 120 words."
        )
    )
    destination_recommender_agent = create_agent(
        agent_name="Destination-Recommender-Agent",  ## AGENT-2
        agent_instructions=(
            "You are a travel expert. The input already identifies a selected destination. "
            "Give practical attractions, neighborhoods, and budget tips for that destination. "
            "Make reasonable assumptions for missing details; do not ask follow-up questions. "
            "Keep the answer under 220 words."
        )
    )
    weather_agent = create_agent(
        agent_name="Weather-Agent",             ## AGENT-3
        agent_instructions=(
            "You are a weather expert. The input already identifies a selected destination. "
            "Summarize likely weather and packing advice using any season or duration in the user's request. "
            "Make reasonable assumptions for missing details; do not ask follow-up questions. "
            "Keep the answer under 180 words."
        )
    )
    cuisine_suggestion_agent = create_agent(
        agent_name="Cuisine-Suggestion-Agent",                   ## AGENT-4
        agent_instructions=(
            "You are a culinary expert. The input already identifies a selected destination. "
            "Suggest local foods and budget-friendly dining ideas for that destination. "
            "Make reasonable assumptions for missing details; do not ask follow-up questions. "
            "Keep the answer under 220 words."
        )
    )
    itinerary_planner_agent = create_agent(
        agent_name="Itinerary-Planner-Agent",                ### AGENT-5
        agent_instructions="You are an itinerary planning expert. Combine the destination, weather, and cuisine notes into a clear day-by-day itinerary. Keep the final answer concise and actionable."
    )

    # Executor IDs make nodes identifiable in traces and visualizations.
    location_selector_executor = LocationSelectorExecutor(location_picker_agent, id="LocationSelector")
    destination_recommender_executor = DestinationRecommenderExecutor(destination_recommender_agent, id="DestinationRecommender")
    weather_executor = WeatherExecutor(weather_agent, id="Weather")
    cuisine_suggestion_executor = CuisineSuggestionExecutor(cuisine_suggestion_agent, id="CuisineSuggestion")
    itinerary_planner_executor = ItineraryPlannerExecutor(itinerary_planner_agent, id="ItineraryPlanner")

    # State lets workflow infrastructure inspect or persist the objects
    # associated with an executor.
    for executor in [
        location_selector_executor,
        destination_recommender_executor,
        weather_executor,
        cuisine_suggestion_executor,
        itinerary_planner_executor,
    ]:
        executor.state = {
            "location_picker_agent": location_picker_agent,
            "destination_recommender_agent": destination_recommender_agent,
            "weather_agent": weather_agent,
            "cuisine_suggestion_agent": cuisine_suggestion_agent,
            "itinerary_planner_agent": itinerary_planner_agent,
        }

    # Data flow:
    # user -> location selector -> three concurrent specialists -> itinerary.
    workflow = (
        WorkflowBuilder(
            name="Vacation Planner Workflow",
            description="Multi-agent workflow for vacation planning with recommendations and itinerary."
        )
        .set_start_executor(location_selector_executor)
        # Fan-out starts independent branches from the same location message.
        .add_fan_out_edges(location_selector_executor, [
            destination_recommender_executor,
            weather_executor,
            cuisine_suggestion_executor
        ])
        # Fan-in waits for every branch before planning the itinerary.
        .add_fan_in_edges([
            destination_recommender_executor,
            weather_executor,
            cuisine_suggestion_executor
        ], itinerary_planner_executor)
        .build()
    )

    # Mermaid text makes the workflow graph easy to inspect or document.
    viz = WorkflowViz(workflow)
    mermaid_content = viz.to_mermaid()
    print("Mermaid Diagram:\n", mermaid_content)

    return workflow

def main():
    """Launch the vacation planning workflow in DevUI."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)
    devui_port = int(os.getenv("DEVUI_PORT", "8092"))
    logger.info("Starting Vacation Planning Workflow")
    logger.info("Available at: http://localhost:%s", devui_port)
    logger.info("Entity ID: workflow_vacation_planner")

    workflow = build_workflow()
    # DevUI provides an interactive browser client. Tracing records each node
    # and agent call so the execution can be inspected.
    serve(entities=[workflow], port=devui_port, auto_open=True, tracing_enabled=True)

if __name__ == "__main__":
    # This guard runs main only when the file is executed directly.
    main()



#######Suggested - user prompts 

#I want a 5-day vacation in India with beaches, good food, and a relaxed budget. Suggest a location and plan the trip.

#Plan a 4-day family vacation from Bangalore. We like nature, light adventure, and vegetarian food.

#I want 3-day trip in Europe during winter. Include destination ideas, weather, cuisine, and itinerary.

#Suggest a budget-friendly solo travel plan for 6 days. I like history, street food, and walkable cities.

