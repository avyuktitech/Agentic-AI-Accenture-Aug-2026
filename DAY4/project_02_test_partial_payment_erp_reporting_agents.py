from project_02_partial_payment_erp_reporting_agents import run_demo


def test_partial_payment_erp_and_reporting_flow():
    result = run_demo()
    assert [row["action"] for row in result["allocations"]] == [
        "keep invoice open", "close invoice", "create customer credit"
    ]
    assert result["report"]["amount_applied"] == 1850.0
    assert result["report"]["remaining_open_balance"] == 400.0
    assert result["report"]["customer_credit_created"] == 50.0
    assert result["report"]["journals_ready_or_posted"] == 3
