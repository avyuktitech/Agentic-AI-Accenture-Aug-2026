# langgraph_azure_agent_simple.py
# Install below packages
#python langgraph_azure_agent_simple.py

import os
from typing import TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient

load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
MODEL_DEPLOYMENT = os.environ["MODEL_DEPLOYMENT_NAME"]
AGENT_NAME = "sandeep-agent11"  # your existing deployed agent

# --- Step 1: Connect to Foundry ---
credential = DefaultAzureCredential()

# allow_preview=True is required because Agent-endpoint routing is still preview
project_client = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=credential,
    allow_preview=True,
)

# Get an OpenAI-protocol client pointed at your existing agent's endpoint
openai_client = project_client.get_openai_client(agent_name=AGENT_NAME)

# --- Step 2: LangGraph state ---
class TicketState(TypedDict):
    message: str
    result: str

# --- Step 3: LangGraph node that calls the Azure AI Agent via Responses API ---
def call_azure_agent(state: TicketState) -> TicketState:
    response = openai_client.responses.create(
        model=MODEL_DEPLOYMENT,
        input=state["message"],
    )
    return {"message": state["message"], "result": response.output_text}

# --- Step 4: Build the graph ---
graph = StateGraph(TicketState)
graph.add_node("triage", call_azure_agent)
graph.set_entry_point("triage")
graph.add_edge("triage", END)
app = graph.compile()

# --- Step 5: Run it ---
if __name__ == "__main__":
    result = app.invoke({
        "message": "My invoice charged me twice this month.",
        "result": "",
    })
    print("\n--- LangGraph result ---")
    print(result["result"])