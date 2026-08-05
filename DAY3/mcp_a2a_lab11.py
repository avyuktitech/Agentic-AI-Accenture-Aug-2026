# mcp_a2a_lab.py
#pip install azure-ai-projects azure-identity langgraph python-dotenv
import os
from typing import TypedDict, Optional
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import PromptAgentDefinition, MCPTool
from openai.types.responses.response_input_param import McpApprovalResponse

load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
MODEL_DEPLOYMENT = os.environ["MODEL_DEPLOYMENT_NAME"]

# --- Step 1: Connect to Foundry ---
credential = DefaultAzureCredential()
project_client = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=credential,
    allow_preview=True,
)
openai_client = project_client.get_openai_client()

# --- Step 2: Define the MCP tool (Microsoft Learn MCP server) ---
mcp_tool = MCPTool(
    server_label="microsoft-learn",         # matches your MCP_SERVER_NAME intent
    server_url="https://learn.microsoft.com/api/mcp",
    require_approval="always",              # safest default — approve each call explicitly
)

# --- Step 3: Create (or version) an agent with the MCP tool attached ---
agent = project_client.agents.create_version(
    agent_name="mcp-learn-agent",
    definition=PromptAgentDefinition(
        model=MODEL_DEPLOYMENT,
        instructions=(
            "You are a helpful assistant that can look up Microsoft Learn "
            "documentation using the connected MCP tool when relevant."
        ),
        tools=[mcp_tool],
    ),
)
print(f"Agent created: {agent.id} (name={agent.name}, version={agent.version})")

# --- Step 4: Helper — call the agent and auto-approve MCP tool calls ---
def call_agent_with_mcp_approval(user_input: str) -> str:
    response = openai_client.responses.create(
        model=MODEL_DEPLOYMENT,
        input=user_input,
        extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
    )

    # Loop: keep approving MCP tool calls until the agent produces final output
    while any(item.type == "mcp_approval_request" for item in response.output):
        approvals = [
            McpApprovalResponse(
                type="mcp_approval_response",
                approval_request_id=item.id,
                approve=True,
            )
            for item in response.output
            if item.type == "mcp_approval_request"
        ]
        response = openai_client.responses.create(
            model=MODEL_DEPLOYMENT,
            input=approvals,
            previous_response_id=response.id,
            extra_body={"agent_reference": {"name": agent.name, "type": "agent_reference"}},
        )

    return response.output_text

# --- Step 5: LangGraph wrapping ---
class LabState(TypedDict):
    question: str
    answer: str

def mcp_node(state: LabState) -> LabState:
    return {"question": state["question"], "answer": call_agent_with_mcp_approval(state["question"])}

graph = StateGraph(LabState)
graph.add_node("mcp_lookup", mcp_node)
graph.set_entry_point("mcp_lookup")
graph.add_edge("mcp_lookup", END)
app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({
        "question": "According to Microsoft Learn, what is Azure AI Foundry Agent Service?",
        "answer": "",
    })
    print("\n--- MCP-grounded answer ---")
    print(result["answer"])