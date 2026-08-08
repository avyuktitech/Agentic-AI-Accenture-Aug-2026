# evaluate_agent.py
#pip install "azure-ai-projects>=2.0.0" azure-identity python-dotenv

# evaluate_agent.py
import os
import time
from pprint import pprint
from dotenv import load_dotenv

from openai.types.evals.create_eval_jsonl_run_data_source_param import (
    CreateEvalJSONLRunDataSourceParam,
    SourceFileContent,
    SourceFileContentContent,
)
from openai.types.eval_create_params import DataSourceConfigCustom

from azure.identity import DefaultAzureCredential
from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import TestingCriterionAzureAIEvaluator

load_dotenv()

PROJECT_ENDPOINT = os.environ["FOUNDRY_PROJECT_ENDPOINT"]
MODEL_DEPLOYMENT = os.environ["MODEL_DEPLOYMENT_NAME"]
AGENT_NAME = "sandeep-agent11"

# --- Step 1: Two clients — agent-scoped and project-scoped ---
credential = DefaultAzureCredential()
project_client = AIProjectClient(
    endpoint=PROJECT_ENDPOINT,
    credential=credential,
    allow_preview=True,
)

agent_client = project_client.get_openai_client(agent_name=AGENT_NAME)  # responses only
eval_client = project_client.get_openai_client()                        # evals live here

# --- Step 2: Collect real agent outputs ---
test_queries = [
    "My invoice charged me twice this month.",
    "What's the status of order 1002?",
    "How do I reset my account password?",
    "Explain your refund policy in one sentence.",
]

print("Collecting live responses from the agent to evaluate...\n")
collected = []
for q in test_queries:
    response = agent_client.responses.create(model=MODEL_DEPLOYMENT, input=q)
    print(f"Q: {q}\nA: {response.output_text[:120]}...\n")
    collected.append({"query": q, "response": response.output_text})

# --- Step 3: Schema + evaluators ---
data_source_config = DataSourceConfigCustom(
    type="custom",
    item_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "response": {"type": "string"},
        },
        "required": [],
    },
    include_sample_schema=False,
)

testing_criteria = [
    TestingCriterionAzureAIEvaluator(
        type="azure_ai_evaluator",
        name="Coherence",
        evaluator_name="builtin.coherence",
        data_mapping={"query": "{{item.query}}", "response": "{{item.response}}"},
        initialization_parameters={"model": MODEL_DEPLOYMENT},
    ),
    TestingCriterionAzureAIEvaluator(
        type="azure_ai_evaluator",
        name="Fluency",
        evaluator_name="builtin.fluency",
        data_mapping={"query": "{{item.query}}", "response": "{{item.response}}"},
        initialization_parameters={"model": MODEL_DEPLOYMENT},
    ),
]

# --- Step 4: Create the evaluation (project-scoped client) ---
print("Creating evaluation...")
eval_object = eval_client.evals.create(
    name="sandeep-agent11-quality-check",
    data_source_config=data_source_config,
    testing_criteria=testing_criteria,  # type: ignore
)
print(f"Evaluation created (id: {eval_object.id})\n")

# --- Step 5: Run it against the collected outputs ---
print("Creating evaluation run with inline data...")
eval_run_object = eval_client.evals.runs.create(
    eval_id=eval_object.id,
    name="b4-agent-quality-run",
    metadata={"team": "track-b-lab", "scenario": "b4-testing-evaluating-agents"},
    data_source=CreateEvalJSONLRunDataSourceParam(
        type="jsonl",
        source=SourceFileContent(
            type="file_content",
            content=[SourceFileContentContent(item=item) for item in collected],
        ),
    ),
)
print(f"Eval run created (id: {eval_run_object.id})\n")

# --- Step 6: Poll until complete ---
print("Waiting for evaluation run to complete...")
while True:
    run = eval_client.evals.runs.retrieve(run_id=eval_run_object.id, eval_id=eval_object.id)
    if run.status in ("completed", "failed"):
        output_items = list(
            eval_client.evals.runs.output_items.list(run_id=run.id, eval_id=eval_object.id)
        )
        print("\n--- Evaluation results ---")
        pprint(output_items)
        print(f"\nFull report: {run.report_url}")
        break
    time.sleep(5)

eval_client.evals.delete(eval_id=eval_object.id)
print("\nEvaluation deleted.")