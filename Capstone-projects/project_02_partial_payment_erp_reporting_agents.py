"""Multi-agent demo for partial payments, D365 journal preparation, and reporting.

This is an auditable local simulation.  `D365JournalAgent` deliberately records
posting intent rather than sending a real ERP transaction; replace its adapter
with an approved Dynamics 365 Finance API integration in production.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class OpenInvoice:
    invoice_id: str
    customer: str
    original_amount: float
    open_balance: float
    currency: str = "USD"


@dataclass
class Payment:
    payment_id: str
    customer: str
    amount: float
    payment_date: str
    currency: str = "USD"
    invoice_reference: str | None = None


@dataclass
class Allocation:
    payment_id: str
    invoice_id: str
    applied_amount: float
    remaining_invoice_balance: float
    variance_amount: float
    action: str
    note: str


@dataclass
class JournalEntry:
    voucher: str
    account: str
    debit: float
    credit: float
    currency: str
    description: str
    dimension: str


@dataclass
class PostingResult:
    voucher: str
    status: str
    message: str
    entries: list[JournalEntry] = field(default_factory=list)


class PartialPaymentAgent:
    """Allocates a payment against an invoice and classifies any variance."""

    tolerance = 0.01

    def allocate(self, payment: Payment, invoice: OpenInvoice) -> Allocation:
        if payment.customer.lower() != invoice.customer.lower():
            raise ValueError("Payment customer does not match invoice customer.")
        if payment.currency != invoice.currency:
            raise ValueError("Currency conversion is required before allocation.")
        applied = min(payment.amount, invoice.open_balance)
        residual = round(invoice.open_balance - applied, 2)
        overpayment = round(payment.amount - applied, 2)
        if overpayment > self.tolerance:
            action, variance, note = "create customer credit", overpayment, "Overpayment retained as customer credit."
        elif residual > self.tolerance:
            action, variance, note = "keep invoice open", residual, "Partial payment applied; residual stays open for collection."
        else:
            action, variance, note = "close invoice", 0.0, "Invoice is fully settled."
        invoice.open_balance = residual
        return Allocation(payment.payment_id, invoice.invoice_id, applied, residual, variance, action, note)


class D365JournalAgent:
    """Creates balanced D365 Finance-style customer-payment journal entries."""

    bank_account = "110100"
    accounts_receivable = "130100"
    customer_credit = "210250"

    def prepare(self, allocation: Allocation, payment: Payment) -> list[JournalEntry]:
        voucher = f"PAY-{payment.payment_id}"
        entries = [
            JournalEntry(voucher, self.bank_account, payment.amount, 0.0, payment.currency,
                         f"Bank receipt {payment.payment_id}", "BUSINESSUNIT-001"),
            JournalEntry(voucher, self.accounts_receivable, 0.0, allocation.applied_amount, payment.currency,
                         f"Settle {allocation.invoice_id}", "BUSINESSUNIT-001"),
        ]
        overpayment = round(payment.amount - allocation.applied_amount, 2)
        if overpayment > 0:
            entries.append(JournalEntry(voucher, self.customer_credit, 0.0, overpayment, payment.currency,
                                        f"Customer credit from {payment.payment_id}", "BUSINESSUNIT-001"))
        return entries

    def validate_and_post(self, entries: list[JournalEntry], *, approve: bool = False) -> PostingResult:
        voucher = entries[0].voucher
        debit = round(sum(entry.debit for entry in entries), 2)
        credit = round(sum(entry.credit for entry in entries), 2)
        if debit != credit:
            return PostingResult(voucher, "rejected", f"Journal is unbalanced: debit={debit}, credit={credit}.", entries)
        if not approve:
            return PostingResult(voucher, "pending approval", "Balanced D365 journal prepared; approval required before API posting.", entries)
        # Production adapter: call the approved D365 Finance journal API here.
        return PostingResult(voucher, "posted (simulated)", "Journal passed validation and would be posted to D365 Finance.", entries)


class ReportingAgent:
    """Produces a concise, immutable reconciliation report from workflow decisions."""

    def reconciliation_report(self, allocations: list[Allocation], postings: list[PostingResult]) -> dict[str, Any]:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "payments_processed": len(allocations),
            "amount_applied": round(sum(item.applied_amount for item in allocations), 2),
            "remaining_open_balance": round(sum(item.remaining_invoice_balance for item in allocations), 2),
            "customer_credit_created": round(sum(item.variance_amount for item in allocations if item.action == "create customer credit"), 2),
            "partial_payments": sum(item.action == "keep invoice open" for item in allocations),
            "journals_ready_or_posted": sum(item.status in {"pending approval", "posted (simulated)"} for item in postings),
            "exceptions": [item.message for item in postings if item.status == "rejected"],
        }


class FinanceOperationsOrchestrator:
    """Coordinates allocation, journal preparation, approval-safe posting, and reporting."""

    def __init__(self) -> None:
        self.partial_payments = PartialPaymentAgent()
        self.d365 = D365JournalAgent()
        self.reporting = ReportingAgent()

    def run(self, payments: list[Payment], invoices: list[OpenInvoice], *, approve_posting: bool = False) -> tuple[list[Allocation], list[PostingResult], dict[str, Any]]:
        invoice_by_id = {invoice.invoice_id: invoice for invoice in invoices}
        allocations, postings = [], []
        for payment in payments:
            if not payment.invoice_reference or payment.invoice_reference not in invoice_by_id:
                raise ValueError(f"Payment {payment.payment_id} needs a valid invoice reference for this demo.")
            allocation = self.partial_payments.allocate(payment, invoice_by_id[payment.invoice_reference])
            entries = self.d365.prepare(allocation, payment)
            allocations.append(allocation)
            postings.append(self.d365.validate_and_post(entries, approve=approve_posting))
        return allocations, postings, self.reporting.reconciliation_report(allocations, postings)


def demo_data() -> tuple[list[Payment], list[OpenInvoice]]:
    invoices = [
        OpenInvoice("INV-2001", "Northwind Traders", 1000.00, 1000.00),
        OpenInvoice("INV-2002", "Contoso Retail", 750.00, 750.00),
        OpenInvoice("INV-2003", "Fabrikam Ltd", 500.00, 500.00),
    ]
    payments = [
        Payment("RCPT-2001", "Northwind Traders", 600.00, "2026-08-06", invoice_reference="INV-2001"),
        Payment("RCPT-2002", "Contoso Retail", 750.00, "2026-08-06", invoice_reference="INV-2002"),
        Payment("RCPT-2003", "Fabrikam Ltd", 550.00, "2026-08-06", invoice_reference="INV-2003"),
    ]
    return payments, invoices


def run_demo() -> dict[str, Any]:
    payments, invoices = demo_data()
    allocations, postings, report = FinanceOperationsOrchestrator().run(payments, invoices, approve_posting=False)
    return {"allocations": [asdict(item) for item in allocations], "postings": [asdict(item) for item in postings], "report": report}


if __name__ == "__main__":
    result = run_demo()
    for allocation in result["allocations"]:
        print(allocation)
    print("Report:", result["report"])
