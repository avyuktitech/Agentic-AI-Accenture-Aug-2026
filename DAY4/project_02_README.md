# Project 02 — Partial Payments, D365 Journals, and Reporting

`Payment -> Partial Payment Agent -> D365 Journal Agent -> Reporting Agent`

The workflow demonstrates a partial payment that leaves an invoice open, a full settlement, and an overpayment that becomes a customer credit. Every journal is balanced, but stays **pending approval** by default: no real D365 transaction is sent.

## Included files

- `project_02_partial_payment_erp_reporting_agents.py` — workflow, D365 journal preparation, and report generation.
- `project_02_partial_payment_erp_reporting.ipynb` — executed allocation, journal-preview, and report walkthrough.
- `project_02_test_partial_payment_erp_reporting_agents.py` — smoke test for allocation and reporting controls.

## Agents and outcome

- **Partial Payment Agent** applies receipts, keeps residual balances open, and creates customer credit for overpayments.
- **D365 Journal Agent** prepares and validates balanced customer-payment journals with voucher, account, currency, and dimension fields.
- **Reporting Agent** produces reconciliation totals and exposes posting exceptions.

The sample processes three receipts: one partial payment ($400 remains open), one full settlement, and one $50 customer credit. It prepares three balanced journals for approval.

## Run

```powershell
.\b1-lab\Scripts\python.exe project_02_partial_payment_erp_reporting_agents.py
.\b1-lab\Scripts\python.exe -c "from project_02_test_partial_payment_erp_reporting_agents import test_partial_payment_erp_and_reporting_flow; test_partial_payment_erp_and_reporting_flow(); print('Smoke test passed')"
.\b1-lab\Scripts\jupyter.exe notebook project_02_partial_payment_erp_reporting.ipynb
```

## D365 integration safety

To post for a controlled integration test, call `run(..., approve_posting=True)`. The current implementation still simulates posting; add your authenticated D365 Finance API adapter only after security and finance approval.
