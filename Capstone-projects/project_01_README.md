# Project 01 — Multi-Agent Remittance Processing

This project demonstrates a production-shaped accounts-receivable workflow that reads remittance data and recommends an auditable invoice match.

`Email / Bank / ERP -> Ingestion Agent -> Remittance Agent -> Rules Agent -> Smart Matching Agent -> Auditable decision`

## Included files

- `project_01_remittance_multi_agent.py` — agent implementations and runnable sample workflow.
- `project_01_remittance_processing.ipynb` — executed walkthrough with hand-offs and result tables.
- `project_01_test_remittance_multi_agent.py` — smoke test for all matching paths.

## Agents and outcome

- **Data Ingestion Agent** validates bank CSV and email remittance notices.
- **Remittance Processing Agent** maps input fields to one canonical payment model.
- **Payment Matching Agent** performs deterministic 2-way and 3-way matching.
- **Smart Matching Agent** produces a confidence-ranked fuzzy suggestion and requires human approval.

The sample covers a 3-way exact match, 2-way exact match, reference-led match, and high-confidence fuzzy suggestion.

## Run

```powershell
.\b1-lab\Scripts\python.exe project_01_remittance_multi_agent.py
.\b1-lab\Scripts\python.exe -c "from project_01_test_remittance_multi_agent import test_demo_covers_each_matching_path; test_demo_covers_each_matching_path(); print('Smoke test passed')"
.\b1-lab\Scripts\jupyter.exe notebook project_01_remittance_processing.ipynb
```

## Production integration points

- Replace `ingest_bank_csv` and `ingest_email` with approved Outlook/SFTP/ERP connectors.
- Use Azure AI Document Intelligence or a Foundry extraction agent for PDFs, scans, and unstructured emails.
- Persist the `MatchDecision` audit record; only auto-post `matched` decisions, while `suggested` and `review` go to an analyst queue.
- Use the Azure Foundry settings already in `.env` to replace `SmartMatchingAgent` scoring with a governed model call if desired.
