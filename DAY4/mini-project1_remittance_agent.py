import os
import json
import re
from typing import TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI


# ============================================================
# 1. LOAD ENVIRONMENT
# ============================================================

load_dotenv()

ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
MODEL = os.getenv("AZURE_OPENAI_DEPLOYMENT")


if not ENDPOINT:
    raise ValueError("AZURE_OPENAI_ENDPOINT missing from .env")

if not API_KEY:
    raise ValueError("AZURE_OPENAI_API_KEY missing from .env")

if not MODEL:
    raise ValueError("AZURE_OPENAI_DEPLOYMENT missing from .env")


ENDPOINT = ENDPOINT.rstrip("/") + "/"


print("=" * 70)
print("MINI PROJECT 1 - INTELLIGENT REMITTANCE PROCESSING")
print("=" * 70)

print("Endpoint :", ENDPOINT)
print("Model    :", MODEL)


# ============================================================
# 2. CREATE LLM
# ============================================================

llm = ChatOpenAI(
    base_url=ENDPOINT,
    api_key=API_KEY,
    model=MODEL,
    temperature=0,
)


# ============================================================
# 3. PYDANTIC DATA MODEL
# ============================================================

class Remittance(BaseModel):

    customer_name: str

    invoice_number: str

    payment_amount: float = Field(gt=0)

    payment_date: str

    payment_reference: str


# ============================================================
# 4. LANGGRAPH STATE
# ============================================================

class State(TypedDict, total=False):

    raw_text: str

    llm_response: str

    remittance: dict

    validation_status: str

    error: str


# ============================================================
# 5. INGESTION NODE
# ============================================================

def ingest(state: State):

    print("\n" + "=" * 60)
    print("[1] INGESTION NODE")
    print("=" * 60)

    text = state.get(
        "raw_text",
        ""
    ).strip()

    if not text:

        return {
            "error": "Empty remittance document",
            "validation_status": "FAILED"
        }

    print("Document received successfully.")

    print("\nDocument:")
    print(text)

    return {}


# ============================================================
# 6. HELPER - EXTRACT JSON FROM MODEL RESPONSE
# ============================================================

def extract_json_from_text(text: str):

    """
    Extract a JSON object from an LLM response.

    Handles responses such as:

    ```json
    {...}
    ```

    or

    Here is the result:
    {...}
    """

    text = text.strip()


    # --------------------------------------------------------
    # Remove Markdown code fences
    # --------------------------------------------------------

    text = re.sub(
        r"^```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"\s*```$",
        "",
        text
    )


    # --------------------------------------------------------
    # First try direct JSON parsing
    # --------------------------------------------------------

    try:

        return json.loads(text)

    except json.JSONDecodeError:

        pass


    # --------------------------------------------------------
    # Search for JSON object inside response
    # --------------------------------------------------------

    start = text.find("{")
    end = text.rfind("}")


    if start != -1 and end != -1 and end > start:

        json_text = text[start:end + 1]

        try:

            return json.loads(json_text)

        except json.JSONDecodeError as exc:

            raise ValueError(
                f"Invalid JSON returned by model: {exc}"
            )


    raise ValueError(
        "No valid JSON object found in model response."
    )


# ============================================================
# 7. REMITTANCE EXTRACTION NODE
# ============================================================

def extract(state: State):

    print("\n" + "=" * 60)
    print("[2] REMITTANCE EXTRACTION AGENT")
    print("=" * 60)


    if state.get("error"):

        return {}


    document = state["raw_text"]


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We are NOT using:
    #
    # llm.with_structured_output(...)
    #
    # because some Foundry-hosted models do not reliably
    # support the structured-output/tool schema expected
    # by LangChain.
    #
    # Instead:
    #
    # LLM
    #  ↓
    # Text JSON
    #  ↓
    # json.loads()
    #  ↓
    # Pydantic validation
    # --------------------------------------------------------


    prompt = f"""
You are an enterprise remittance extraction system.

Extract remittance information from the DOCUMENT below.

You MUST return exactly one JSON object.

Use EXACTLY these five keys:

customer_name
invoice_number
payment_amount
payment_date
payment_reference


IMPORTANT RULES:

1. Return ONLY JSON.
2. Do NOT use Markdown.
3. Do NOT use ```json code blocks.
4. Do NOT explain your answer.
5. Do NOT add any other fields.
6. payment_amount MUST be numeric.
7. Remove INR, currency symbols and commas from payment_amount.
8. Do not invent information.
9. Preserve invoice numbers and payment references exactly.


Example output:

{{
  "customer_name": "ABC Retail Pvt Ltd",
  "invoice_number": "INV-1001",
  "payment_amount": 98500,
  "payment_date": "05-Aug-2026",
  "payment_reference": "HDFC-998877"
}}


DOCUMENT:

----------------------------

{document}

----------------------------


Return the JSON object now:
"""


    try:

        response = llm.invoke(prompt)


        # ----------------------------------------------------
        # Extract text content
        # ----------------------------------------------------

        content = response.content


        # Some model integrations can return content blocks
        # rather than a simple string.

        if isinstance(content, list):

            text_parts = []

            for item in content:

                if isinstance(item, str):

                    text_parts.append(item)

                elif isinstance(item, dict):

                    if "text" in item:

                        text_parts.append(
                            str(item["text"])
                        )

            content = "".join(text_parts)


        content = str(content).strip()


        print("\nRAW MODEL RESPONSE:")
        print("-" * 50)
        print(content)
        print("-" * 50)


        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        parsed = extract_json_from_text(
            content
        )


        # ----------------------------------------------------
        # Validate using Pydantic
        # ----------------------------------------------------

        remittance = Remittance.model_validate(
            parsed
        )


        result = remittance.model_dump()


        print("\nStructured Remittance:")

        print(
            json.dumps(
                result,
                indent=2
            )
        )


        return {

            "llm_response": content,

            "remittance": result
        }


    except ValidationError as exc:

        error = (
            "Pydantic validation failed: "
            + str(exc)
        )

        print("\nERROR:")
        print(error)

        return {

            "error": error,

            "validation_status": "FAILED"
        }


    except Exception as exc:

        error = (
            "Extraction failed: "
            + str(exc)
        )

        print("\nERROR:")
        print(error)

        return {

            "error": error,

            "validation_status": "FAILED"
        }


# ============================================================
# 8. VALIDATION NODE
# ============================================================

def validate(state: State):

    print("\n" + "=" * 60)
    print("[3] VALIDATION NODE")
    print("=" * 60)


    if state.get("error"):

        print(
            "Validation skipped because "
            "extraction failed."
        )

        return {}


    remittance = state.get(
        "remittance"
    )


    if not remittance:

        return {

            "error":
                "No remittance data available",

            "validation_status":
                "FAILED"
        }


    # --------------------------------------------------------
    # Required fields
    # --------------------------------------------------------

    required_fields = [

        "customer_name",

        "invoice_number",

        "payment_amount",

        "payment_date",

        "payment_reference"
    ]


    missing_fields = [

        field

        for field in required_fields

        if remittance.get(field)
        in (None, "")
    ]


    if missing_fields:

        status = (
            "FAILED - Missing fields: "
            + ", ".join(missing_fields)
        )

        print(status)

        return {
            "validation_status": status
        }


    # --------------------------------------------------------
    # Business validation
    # --------------------------------------------------------

    if remittance["payment_amount"] <= 0:

        return {

            "validation_status":
                "FAILED - Invalid payment amount"
        }


    # --------------------------------------------------------
    # Display validated data
    # --------------------------------------------------------

    print(
        "Customer Name      :",
        remittance["customer_name"]
    )

    print(
        "Invoice Number     :",
        remittance["invoice_number"]
    )

    print(
        "Payment Amount     :",
        remittance["payment_amount"]
    )

    print(
        "Payment Date       :",
        remittance["payment_date"]
    )

    print(
        "Payment Reference  :",
        remittance["payment_reference"]
    )


    print("\nValidation PASSED")


    return {
        "validation_status": "VALID"
    }


# ============================================================
# 9. BUILD LANGGRAPH
# ============================================================

builder = StateGraph(State)


builder.add_node(
    "ingestion",
    ingest
)

builder.add_node(
    "extraction",
    extract
)

builder.add_node(
    "validation",
    validate
)


# ============================================================
# 10. DEFINE WORKFLOW
# ============================================================

builder.add_edge(
    START,
    "ingestion"
)

builder.add_edge(
    "ingestion",
    "extraction"
)

builder.add_edge(
    "extraction",
    "validation"
)

builder.add_edge(
    "validation",
    END
)


workflow = builder.compile()


# ============================================================
# 11. TEST
# ============================================================

if __name__ == "__main__":


    document = """

Customer: ABC Retail Pvt Ltd

Invoice Number: INV-1001

Payment Amount: INR 98,500

Payment Date: 05-Aug-2026

Bank Reference: HDFC-998877

"""


    print("\nINPUT DOCUMENT")
    print("=" * 60)

    print(
        document.strip()
    )


    result = workflow.invoke({

        "raw_text": document

    })


    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)


    if result.get("error"):

        print(
            "STATUS : FAILED"
        )

        print(
            "ERROR  :",
            result["error"]
        )


    else:

        print(
            json.dumps(
                result.get(
                    "remittance",
                    {}
                ),
                indent=2
            )
        )

        print(
            "\nValidation Status:",
            result.get(
                "validation_status"
            )
        )


    print("=" * 70)