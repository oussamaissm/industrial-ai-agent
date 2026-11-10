from src.evaluation.runner import evaluate_success

DIAG = {"type": "diagnosis", "ground_truth": {"machine_id": "CR-01", "parameter_any": ["differential pressure", "pressure"]}}
HEALTHY = {"type": "healthy", "ground_truth": {"expect_no_alarms": True}}


def test_diagnosis_accepts_id_variants():
    for ident in ("CR-01", "cr01", "CR 01"):
        ok, _ = evaluate_success(DIAG, {"answer": f"{ident} has low differential pressure.", "structured_data": None})
        assert ok
    ok, _ = evaluate_success(DIAG, {"answer": "Cleanroom pressure is low", "structured_data": {"machine_id": "CR-01"}})
    assert ok


def test_diagnosis_rejects_wrong_asset_or_missing_parameter():
    assert not evaluate_success(DIAG, {"answer": "BR-101 differential pressure low", "structured_data": None})[0]
    ok, why = evaluate_success(DIAG, {"answer": "CR-01 has a problem", "structured_data": None})
    assert not ok and "parameter" in why


def test_diagnosis_has_no_lenient_shortcut():
    # the legacy 'analysis' type passes on keywords alone; 'diagnosis' must not
    assert not evaluate_success(DIAG, {"answer": "Machine is down for maintenance", "structured_data": {"x": 1}})[0]


def test_healthy_task():
    assert evaluate_success(HEALTHY, {"answer": "There are no active alarms on the site.", "structured_data": None})[0]
    assert evaluate_success(HEALTHY, {"answer": "All clear: 0 active alarms.", "structured_data": None})[0]
    assert not evaluate_success(HEALTHY, {"answer": "CR-01 has a critical alarm.", "structured_data": None})[0]
    assert not evaluate_success(HEALTHY, {"answer": "I could not tell.", "structured_data": None})[0]


def test_simulator_tasks_unchanged():
    t1 = {"type": "lookup", "ground_truth": {"machine_id": "M001"}}
    assert evaluate_success(t1, {"answer": "M001 has the most downtime", "structured_data": None})[0]
    assert not evaluate_success(t1, {"answer": "M002", "structured_data": None})[0]
