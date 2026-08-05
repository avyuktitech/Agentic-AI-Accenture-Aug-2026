"""ReAct-style Azure AI Foundry agent with safe local business tools.

Prerequisites:
    pip install "azure-ai-projects>=2.0.0" azure-identity python-dotenv
    az login

Run:
    python react_foundry_agent.py
"""

import json
import os
from typing import Any

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import FunctionTool, PromptAgentDefinition
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv


load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
MODEL_DEPLOYMENT = os.environ["MODEL_DEPLOYMENT_NAME"]
AGENT_NAME = os.getenv("REACT_AGENT_NAME", "react-customer-support-agent")
MAX_TOOL_ROUNDS = 5


# Replace these demo functions with calls to approved enterprise systems.
ORDERS = {
    "1001": {"status": "Shipped", "eta": "2026-08-07", "carrier": "Contoso Express"},
    "1002": {"status": "Processing", "eta": "2026-08-09", "carrier": None},
    "1003": {"status": "Delivered", "eta": "2026-08-02", "carrier": "Contoso Express"},
}


def lookup_order(order_id: str) -> dict[str, Any]:
    """Return the current status for an order."""
    order = ORDERS.get(order_id)
    if order is None:
        return {"found": False, "message": f"No order exists for ID {order_id}."}
    return {"found": True, "order_id": order_id, **order}


def get_refund_policy() -> dict[str, Any]:
    """Return the published refund policy used by this demo."""
    return {
        "return_window_days": 30,
        "eligibility": "Unused items in their original condition are eligible.",
        "processing_time": "Approved refunds are processed within 5 business days.",
    }


TOOLS = [
    FunctionTool(
        name="lookup_order",
        description="Look up an order when the customer supplies an order ID.",
        parameters={
            "type": "object",
            "properties": {"order_id": {"type": "string", "description": "The order ID, for example 1002."}},
            "required": ["order_id"],
            "additionalProperties": False,
        },
        strict=True,
    ),
    FunctionTool(
        name="get_refund_policy",
        description="Retrieve the current refund policy before answering a policy question.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
        strict=True,
    ),
]

TOOL_HANDLERS = {
    "lookup_order": lookup_order,
    "get_refund_policy": get_refund_policy,
}

INSTRUCTIONS = """You are a concise customer-support agent.
Use the available tools whenever a question needs order or refund-policy data.
Never invent tool results. The tools are read-only: do not claim that you changed an
order, issued a refund, or performed any external action. Give a clear final answer
after you have the required facts. Do not offer to send notifications, create tickets,
or perform a follow-up action. Do not expose private chain-of-thought reasoning."""


def redact_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep the demo trace useful without later leaking sensitive argument names."""
    return {key: "[redacted]" if any(word in key.lower() for word in ("token", "key", "password")) else value
            for key, value in arguments.items()}


def get_or_create_react_agent(project_client: AIProjectClient) -> None:
    """Create the Foundry agent once; agent tools cannot be supplied per request."""
    try:
        project_client.agents.get(AGENT_NAME)
        print(f"Using Foundry agent: {AGENT_NAME}")
        return
    except ResourceNotFoundError:
        pass

    agent = project_client.agents.create_version(
        agent_name=AGENT_NAME,
        definition=PromptAgentDefinition(
            model=MODEL_DEPLOYMENT,
            instructions=INSTRUCTIONS,
            tools=TOOLS,
        ),
        description="ReAct customer support demo with read-only local tools.",
    )
    print(f"Created Foundry agent: {agent.name} (version {agent.version})")


def run_react_agent(question: str) -> str:
    """Run an action/observation loop until the Foundry agent returns final text."""
    project_client = AIProjectClient(
        endpoint=PROJECT_ENDPOINT,
        credential=DefaultAzureCredential(),
        allow_preview=True,
    )
    get_or_create_react_agent(project_client)
    agent_client = project_client.get_openai_client(agent_name=AGENT_NAME)

    response = agent_client.responses.create(
        model=MODEL_DEPLOYMENT,
        input=question,
        parallel_tool_calls=False,
    )

    for round_number in range(1, MAX_TOOL_ROUNDS + 1):
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            return response.output_text

        tool_outputs = []
        for call in calls:
            arguments = json.loads(call.arguments)
            handler = TOOL_HANDLERS.get(call.name)
            result = (
                handler(**arguments)
                if handler
                else {"error": f"Unsupported tool: {call.name}"}
            )
            print(f"[Action {round_number}] {call.name}({redact_arguments(arguments)})")
            print(f"[Observation] {json.dumps(result)}")
            tool_outputs.append(
                {"type": "function_call_output", "call_id": call.call_id, "output": json.dumps(result)}
            )

        response = agent_client.responses.create(
            model=MODEL_DEPLOYMENT,
            input=tool_outputs,
            previous_response_id=response.id,
            parallel_tool_calls=False,
        )

    raise RuntimeError(f"The agent exceeded the {MAX_TOOL_ROUNDS}-round safety limit.")


if __name__ == "__main__":
    prompt = "What is the status of order 1002, and when should it arrive?"
    print(f"Question: {prompt}\n")
    print("Final answer:")
    print(run_react_agent(prompt))
