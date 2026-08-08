"""Runnable multi-agent remittance processing demonstration.

The module is deliberately dependency-free.  The agents are small, inspectable
business components; production adapters can replace the in-memory sources with
Outlook, SFTP, an ERP API, or Azure AI Document Intelligence.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from difflib import SequenceMatcher
from typing import Any, Iterable


@dataclass
class Remittance:
    remittance_id: str
    amount: float
    payer: str
    payment_date: str
    invoice_reference: str | None = None
    source: str = "unknown"


@dataclass
class Invoice:
    invoice_id: str
    customer: str
    amount_due: float
    invoice_date: str
    po_number: str | None = None


@dataclass
class MatchDecision:
    remittance_id: str
    status: str
    match_type: str
    invoice_ids: list[str] = field(default_factory=list)
    confidence: float = 0.0
    reason: str = ""


def _normalise(value: str | None) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


class DataIngestionAgent:
    """Validates and loads bank CSV plus email remittance notices."""

    required_csv_fields = {"payment_ref", "paid_amt", "payer", "payment_date"}

    def ingest_bank_csv(self, content: str) -> list[dict[str, str]]:
        rows = list(csv.DictReader(io.StringIO(content.strip())))
        headers = set(rows[0]) if rows else set()
        missing = self.required_csv_fields - headers
        if missing:
            raise ValueError(f"Bank file rejected; missing columns: {sorted(missing)}")
        if not rows:
            raise ValueError("Bank file rejected; no payment rows found.")
        return rows

    def ingest_email(self, body: str) -> dict[str, str]:
        fields = dict(re.findall(r"(?im)^\s*([a-z_ ]+)\s*:\s*(.+?)\s*$", body))
        normalised = {_normalise(key): value.strip() for key, value in fields.items()}
        required = {"paymentreference", "amount", "payer", "paymentdate"}
        missing = required - set(normalised)
        if missing:
            raise ValueError(f"Email notice rejected; missing fields: {sorted(missing)}")
        return normalised


class RemittanceProcessingAgent:
    """Converts differently shaped source records into a canonical remittance."""

    def from_bank_row(self, row: dict[str, str]) -> Remittance:
        return Remittance(
            remittance_id=row["payment_ref"].strip(), amount=float(row["paid_amt"]),
            payer=row["payer"].strip(), payment_date=row["payment_date"].strip(),
            invoice_reference=(row.get("invoice_ref") or "").strip() or None, source="bank_csv",
        )

    def from_email(self, fields: dict[str, str]) -> Remittance:
        return Remittance(
            remittance_id=fields["paymentreference"], amount=float(fields["amount"].replace(",", "")),
            payer=fields["payer"], payment_date=fields["paymentdate"],
            invoice_reference=fields.get("invoice") or None, source="email",
        )


class PaymentMatchingAgent:
    """Performs deterministic three-way, two-way, and invoice-reference matches."""

    @staticmethod
    def _same_customer(remittance: Remittance, invoice: Invoice) -> bool:
        return _normalise(remittance.payer) == _normalise(invoice.customer)

    def match(self, remittance: Remittance, invoices: list[Invoice]) -> MatchDecision | None:
        reference = _normalise(remittance.invoice_reference)
        if reference:
            for invoice in invoices:
                if _normalise(invoice.invoice_id) == reference:
                    if self._same_customer(remittance, invoice) and abs(remittance.amount - invoice.amount_due) < 0.01:
                        return MatchDecision(remittance.remittance_id, "matched", "3-way exact", [invoice.invoice_id], 1.0,
                            "Invoice reference, payer, and amount agree.")
                    return MatchDecision(remittance.remittance_id, "exception", "reference conflict", [invoice.invoice_id], 0.25,
                        "Invoice reference exists but the payer or amount disagrees.")

        candidates = [i for i in invoices if self._same_customer(remittance, i) and abs(remittance.amount - i.amount_due) < 0.01]
        if len(candidates) == 1:
            return MatchDecision(remittance.remittance_id, "matched", "2-way exact", [candidates[0].invoice_id], 0.96,
                "Payer and amount agree; no reliable invoice reference was supplied.")
        return None


class SmartMatchingAgent:
    """Ranks unresolved payments; auto-posting is limited to high-confidence cases."""

    def suggest(self, remittance: Remittance, invoices: list[Invoice]) -> MatchDecision:
        ranked: list[tuple[float, Invoice, list[str]]] = []
        for invoice in invoices:
            payer_score = SequenceMatcher(None, _normalise(remittance.payer), _normalise(invoice.customer)).ratio()
            amount_score = max(0.0, 1 - abs(remittance.amount - invoice.amount_due) / max(invoice.amount_due, 1))
            reference_score = SequenceMatcher(None, _normalise(remittance.invoice_reference), _normalise(invoice.invoice_id)).ratio() if remittance.invoice_reference else 0
            score = round(0.45 * payer_score + 0.35 * amount_score + 0.20 * reference_score, 3)
            reasons = [f"payer similarity {payer_score:.0%}", f"amount similarity {amount_score:.0%}"]
            if remittance.invoice_reference:
                reasons.append(f"reference similarity {reference_score:.0%}")
            ranked.append((score, invoice, reasons))

        ranked.sort(key=lambda item: item[0], reverse=True)
        best_score, best_invoice, reasons = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        status = "suggested" if best_score >= 0.82 and best_score - second_score >= 0.08 else "review"
        return MatchDecision(remittance.remittance_id, status, "AI-assisted fuzzy", [best_invoice.invoice_id], best_score,
            "; ".join(reasons) + ". Human approval required before posting.")


class RemittanceOrchestrator:
    """Coordinates specialist agents and emits an auditable decision log."""

    def __init__(self) -> None:
        self.ingestion = DataIngestionAgent()
        self.processing = RemittanceProcessingAgent()
        self.rules = PaymentMatchingAgent()
        self.smart = SmartMatchingAgent()

    def run(self, bank_csv: str, email_notice: str, invoices: list[Invoice]) -> tuple[list[Remittance], list[MatchDecision]]:
        bank_rows = self.ingestion.ingest_bank_csv(bank_csv)
        email_fields = self.ingestion.ingest_email(email_notice)
        remittances = [self.processing.from_bank_row(row) for row in bank_rows]
        remittances.append(self.processing.from_email(email_fields))
        decisions = []
        for payment in remittances:
            decisions.append(self.rules.match(payment, invoices) or self.smart.suggest(payment, invoices))
        return remittances, decisions


def demo_data() -> tuple[str, str, list[Invoice]]:
    bank_csv = """payment_ref,paid_amt,payer,payment_date,invoice_ref
PAY-1001,1250.00,Northwind Traders,2026-08-05,INV-1001
PAY-1002,875.50,Contoso Retail,2026-08-05,
PAY-1003,1440.00,Adventure Works,2026-08-05,INV-1004
"""
    email_notice = """Payment Reference: PAY-1004
Amount: 640.00
Payer: Fabrikam Ltd
Payment Date: 2026-08-05
Invoice: INV-1005A
"""
    invoices = [
        Invoice("INV-1001", "Northwind Traders", 1250.00, "2026-07-18", "PO-551"),
        Invoice("INV-1002", "Contoso Retail", 875.50, "2026-07-20", "PO-552"),
        Invoice("INV-1003", "AdventureWorks", 1450.00, "2026-07-22", "PO-553"),
        Invoice("INV-1004", "Adventure Works", 1440.00, "2026-07-23", "PO-554"),
        Invoice("INV-1005", "Fabrikam Limited", 640.00, "2026-07-25", "PO-555"),
    ]
    return bank_csv, email_notice, invoices


def run_demo() -> list[dict[str, Any]]:
    bank_csv, email_notice, invoices = demo_data()
    _, decisions = RemittanceOrchestrator().run(bank_csv, email_notice, invoices)
    return [asdict(decision) for decision in decisions]


if __name__ == "__main__":
    for result in run_demo():
        print(result)
