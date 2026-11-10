from src.backends import get_backend
from src.backends.base import NotSupportedError
import pytest


def test_machine_status_matches_original_shape():
    b = get_backend("simulator")
    assert b.get_machine_status("M001") == {"id": "M001", "capacity_hours": 9, "status": "operational", "downtime_minutes": 379}
    assert len(b.get_machine_status()) == 10


def test_id_normalisation_and_unknown():
    b = get_backend("simulator")
    assert b.get_machine_status("m001")["id"] == "M001"
    assert b.get_machine_status("M999") == {"error": "Not found"}


def test_downtime_analysis_matches_benchmark_ground_truth():
    # data/benchmark_tasks.jsonl T001: M001 with 379 minutes
    result = get_backend("simulator").analyze_machine_downtime()
    assert result["highest_downtime_machine"] == "M001"
    assert result["ranking"][0]["downtime_minutes"] == 379


def test_schedule_and_missing_order():
    b = get_backend("simulator")
    assert b.calculate_schedule("O0010", "M001")["success"] is True
    assert b.calculate_schedule("O0010", "M003") == {"success": False, "reason": "Machine unavailable"}
    assert b.calculate_schedule("O9999", "M001")["success"] is False


def test_alarms_are_derived_from_status():
    alarms = get_backend("simulator").get_active_alarms()
    assert {a["asset_id"] for a in alarms} == {"M003", "M007", "M009", "M010"}
    assert get_backend("simulator").get_active_alarms("m003")[0]["severity"] == "critical"


def test_openfactory_has_no_orders(make_of_backend):
    backend, _ = make_of_backend()
    with pytest.raises(NotSupportedError):
        backend.get_order_details("O0001")
