from project_01_remittance_multi_agent import run_demo


def test_demo_covers_each_matching_path():
    results = run_demo()
    assert [result["match_type"] for result in results] == [
        "3-way exact", "2-way exact", "3-way exact", "AI-assisted fuzzy"
    ]
    assert results[-1]["status"] == "suggested"
    assert results[-1]["invoice_ids"] == ["INV-1005"]
