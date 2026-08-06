import os
from typing import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_openai import AzureChatOpenAI


load_dotenv()


# ------------------------------------------------------------
# LLM
# ------------------------------------------------------------

llm = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
    api_version=os.getenv(
        "AZURE_OPENAI_API_VERSION",
        "2025-04-01-preview"
    ),
)


# ------------------------------------------------------------
# Structured output
# ------------------------------------------------------------

class Remittance(BaseModel):
    customer_name: str
    invoice_number: str
    payment_amount: float = Field(gt=0)
    payment_date: str
    payment_reference: str


extractor = llm.with_structured_output(Remittance)


# ------------------------------------------------------------
# LangGraph state
# ------------------------------------------------------------

class State(TypedDict, total=False):
    raw_text: str
    remittance: dict
    validation_status: str


# ------------------------------------------------------------
# Agent nodes
# ------------------------------------------------------------

def ingest(state: State):

    text = state["raw_text"].strip()

    if not text:
        raise ValueError("Empty remittance document")

    print("\n[INGESTION AGENT]")
    print("Document received")

    return {}


def extract(state: State):

    print("\n[REMITTANCE EXTRACTION AGENT]")

    result = extractor.invoke(
        f"""
        Extract remittance information from the following document.

        Document:
        {state['raw_text']}

        Do not invent values.
        """
    )

    return {
        "remittance": result.model_dump()
    }


def validate(state: State):

    print("\n[VALIDATION AGENT]")

    r = state["remittance"]

    required = [
        "customer_name",
        "invoice_number",
        "payment_amount",
        "payment_reference"
    ]

    missing = [
        field for field in required
        if not r.get(field)
    ]

    if missing:
        status = f"FAILED: Missing {missing}"
    else:
        status = "VALID"

    return {
        "validation_status": status
    }


# ------------------------------------------------------------
# Workflow
# ------------------------------------------------------------

builder = StateGraph(State)

builder.add_node("ingestion", ingest)
builder.add_node("extraction", extract)
builder.add_node("validation", validate)

builder.add_edge(START, "ingestion")
builder.add_edge("ingestion", "extraction")
builder.add_edge("extraction", "validation")
builder.add_edge("validation", END)

workflow = builder.compile()


# ------------------------------------------------------------
# Test
# ------------------------------------------------------------

if __name__ == "__main__":

    document = """
    Customer: ABC Retail Pvt Ltd
    Invoice Number: INV-1001
    Payment Amount: INR 98,500
    Payment Date: 05-Aug-2026
    Bank Reference: HDFC-998877
    """

    result = workflow.invoke({
        "raw_text": document
    })

    print("\nFINAL RESULT")
    print(result["remittance"])
    print(result["validation_status"])
