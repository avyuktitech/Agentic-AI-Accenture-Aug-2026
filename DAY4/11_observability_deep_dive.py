# observability_deep_dive.py
import os
import time
from dotenv import load_dotenv

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    AzureAIDataSourceConfig,
    TestingCriterionAzureAIEvaluator,
    EvaluationRule,
    ContinuousEvaluationRuleAction,
    EvaluationRuleFilter,
    EvaluationRuleEventType,
)
from azure.ai.projects.telemetry import AIProjectInstrumentor

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor

# --- Step 0: Tracing must be enabled before the client is created ---
os.environ["AZURE_EXPERIMENTAL_ENABLE_GENAI_TRACING"] = "true"

load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
AGENT_NAME = "sandeep-agent11"

# --- Step 1: Wire up console tracing (the "Trace" pillar) ---
tracer_provider = TracerProvider()
tracer_provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
trace.set_tracer_provider(tracer_provider)
AIProjectInstrumentor().instrument()
tracer = trace.get_tracer(__name__)

# --- Step 2: Connect ---
credential = DefaultAzureCredential()
project_client = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=credential,
    allow_preview=True,
)
openai_client = project_client.get_openai_client()

# --- Step 3: Define what to evaluate continuously (the "Evaluate" pillar) ---
data_source_config = AzureAIDataSourceConfig(type="azure_ai_source", scenario="responses")

testing_criteria = [
    TestingCriterionAzureAIEvaluator(
        type="azure_ai_evaluator",
        name="violence_detection",
        evaluator_name="builtin.violence",
    )
]

eval_object = openai_client.evals.create(
    name="B6-Continuous-Safety-Eval",
    data_source_config=data_source_config,
    testing_criteria=testing_criteria,
)
print(f"Evaluation created (id: {eval_object.id})")

# --- Step 4: Create the continuous evaluation rule (the "Monitor" pillar) ---
continuous_eval_rule = project_client.evaluation_rules.create_or_update(
    id="b6-continuous-eval-rule",
    evaluation_rule=EvaluationRule(
        display_name="B6 Continuous Eval Rule",
        description="Runs safety evaluation automatically on every response from sandeep-agent11",
        action=ContinuousEvaluationRuleAction(eval_id=eval_object.id, max_hourly_runs=100),
        event_type=EvaluationRuleEventType.RESPONSE_COMPLETED,
        filter=EvaluationRuleFilter(agent_name=AGENT_NAME),
        enabled=True,
    ),
)
print(f"Continuous eval rule created: {continuous_eval_rule.display_name}\n")

# --- Step 5: Generate traffic — every response gets traced AND continuously evaluated ---
with tracer.start_as_current_span("b6-observability-demo"):
    conversation = openai_client.conversations.create(
        items=[{"type": "message", "role": "user", "content": "What is the size of France in square miles?"}],
    )
    print(f"Conversation started (id: {conversation.id})")

    response = openai_client.responses.create(
        conversation=conversation.id,
        extra_body={"agent_reference": {"name": AGENT_NAME, "type": "agent_reference"}},
    )
    print(f"Response: {response.output_text[:100]}...\n")

    MAX_QUESTIONS = 5
    for i in range(MAX_QUESTIONS):
        openai_client.conversations.items.create(
            conversation_id=conversation.id,
            items=[{"type": "message", "role": "user", "content": f"Question {i}: What is the capital city?"}],
        )
        response = openai_client.responses.create(
            conversation=conversation.id,
            extra_body={"agent_reference": {"name": AGENT_NAME, "type": "agent_reference"}},
        )
        print(f"[{i}] Response: {response.output_text[:80]}...")

# --- Step 6: Poll for the automatically-triggered eval runs ---
print("\nWaiting for continuous evaluation to pick up the traffic...")
for attempt in range(20):
    eval_run_list = openai_client.evals.runs.list(eval_id=eval_object.id, order="desc", limit=10)
    if eval_run_list.data and eval_run_list.data[0].report_url:
        run_report_url = eval_run_list.data[0].report_url
        report_url = "/".join(run_report_url.split("/")[:-2])
        print(f"\nContinuous eval runs are flowing. Open in Foundry portal:\n{report_url}")
        break
    print("  still waiting...")
    time.sleep(10)