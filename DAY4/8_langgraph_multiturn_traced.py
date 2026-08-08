# langgraph_multiturn_traced.py
# Install below packages
#pip install opentelemetry-sdk

import os
from typing import TypedDict, Optional

# --- Tracing must be enabled BEFORE the Foundry client is created ---
os.environ["AZURE_EXPERIMENTAL_ENABLE_GENAI_TRACING"] = "true"

from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.telemetry import AIProjectInstrumentor

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
MODEL_DEPLOYMENT = os.environ["MODEL_DEPLOYMENT_NAME"]
AGENT_NAME = "sandeep-agent11"

# --- Step 1: Set up console tracing ---
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(tracer_provider)
AIProjectInstrumentor().instrument()
tracer = trace.get_tracer(__name__)

# --- Step 2: Connect to Foundry ---
credential = DefaultAzureCredential()
project_client = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=credential,
    allow_preview=True,
)
openai_client = project_client.get_openai_client(agent_name=AGENT_NAME)

# --- Step 3: LangGraph state — now carries response_id for multi-turn ---
class TicketState(TypedDict):
    message: str
    follow_up: str
    first_result: str
    final_result: str
    response_id: Optional[str]

# --- Step 4: Node 1 — initial triage turn ---
def triage_node(state: TicketState) -> TicketState:
    response = openai_client.responses.create(
        model=MODEL_DEPLOYMENT,
        input=state["message"],
    )
    return {
        **state,
        "first_result": response.output_text,
        "response_id": response.id,
    }

# --- Step 5: Node 2 — follow-up turn, chained via previous_response_id ---
def follow_up_node(state: TicketState) -> TicketState:
    response = openai_client.responses.create(
        model=MODEL_DEPLOYMENT,
        input=state["follow_up"],
        previous_response_id=state["response_id"],  # <-- this is the chaining
    )
    return {**state, "final_result": response.output_text}

# --- Step 6: Build the two-node graph ---
graph = StateGraph(TicketState)
graph.add_node("triage", triage_node)
graph.add_node("follow_up", follow_up_node)
graph.set_entry_point("triage")
graph.add_edge("triage", "follow_up")
graph.add_edge("follow_up", END)
app = graph.compile()

# --- Step 7: Run it ---
if __name__ == "__main__":
    with tracer.start_as_current_span("langgraph-multiturn-demo"):
        result = app.invoke({
            "message": "My invoice charged me twice this month.",
            "follow_up": "It's from Acme Corp, invoice #4521, charged $49 on both Aug 1 and Aug 2.",
            "first_result": "",
            "final_result": "",
            "response_id": None,
        })

    print("\n--- Turn 1 (triage) ---")
    print(result["first_result"])
    print("\n--- Turn 2 (follow-up, same conversation) ---")
    print(result["final_result"])