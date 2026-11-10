import pytest

from src.backends.base import BackendError


def test_overview_healthy(make_of_backend):
    b, _ = make_of_backend("healthy")
    o = b.get_factory_overview()
    assert o["site"] == "Open Factory Demo Site"
    assert o["assets"] == 6
    assert o["assets_by_status"] == {"operational": 6}
    assert o["active_alarm_count"] == 0


def test_status_resolves_ids_flexibly(make_of_backend):
    b, _ = make_of_backend("healthy")
    for alias in ("CR-01", "cr01", "CR 01"):
        assert b.get_machine_status(alias)["machine_id"] == "CR-01"
    assert len(b.get_machine_status()) == 6


def test_fault_state_is_reported(make_of_backend):
    b, _ = make_of_backend("cr01_pressure_fault")
    status = b.get_machine_status("CR-01")
    assert status["status"] in ("warning", "alarm")
    assert "CR01_DP_LOW" in status["active_alarms"]
    assert "CR01.DifferentialPressure" in status["degraded_parameters"]

    alarms = b.get_active_alarms()
    assert [a["alarm_id"] for a in alarms] == ["CR01_DP_LOW"]
    assert b.get_active_alarms("BR-101") == []

    t = b.get_machine_telemetry("CR-01")
    dp = next(r for r in t["readings"] if r["name"] == "CR01.DifferentialPressure")
    assert dp["quality"] != "Good" and dp["limits"]["alarm_low"] == 7
    assert "CR01.DifferentialPressure" in t["out_of_range"]

    d = b.analyze_machine_downtime()
    assert d["all_clear"] is False and d["degraded_assets"][0]["machine_id"] == "CR-01"


def test_healthy_analysis_is_all_clear(make_of_backend):
    b, _ = make_of_backend("healthy")
    assert b.analyze_machine_downtime()["all_clear"] is True
    assert b.get_active_alarms() == []


def test_production_metrics(make_of_backend):
    b, _ = make_of_backend("healthy")
    m = b.get_production_metrics()
    assert {x["batch_id"] for x in m["batches"]} == {"BATCH-OFI-2401", "LOT-FL-401-2401"}
    assert m["filling_line"]["units_per_minute"]["unit"] == "units/min"


def test_unknown_asset_lists_known_ones(make_of_backend):
    b, _ = make_of_backend()
    with pytest.raises(BackendError, match="Known assets: BR-101"):
        b.get_machine_telemetry("XX-99")


def test_retries_transient_failures(make_of_backend):
    b, handler = make_of_backend()
    handler.fail_first = 2
    assert b.health()["status"] == "ok"
    assert handler.calls.count("/health") == 3


def test_gives_up_after_retries(make_of_backend):
    b, handler = make_of_backend()
    handler.fail_first = 99
    with pytest.raises(BackendError, match="unreachable or failing"):
        b.health()


def test_malformed_payload_is_rejected_not_passed_on(make_of_backend):
    b, handler = make_of_backend()
    handler.garbage = True
    with pytest.raises(BackendError, match="Unexpected site snapshot payload"):
        b.get_factory_overview()


def test_definitions_are_cached(make_of_backend):
    b, handler = make_of_backend("cr01_pressure_fault")
    b.get_machine_telemetry("CR-01")
    first = len(handler.calls)
    b.get_machine_telemetry("CR-01")
    assert sum(1 for p in handler.calls[first:] if p.startswith("/api/v1/tags/")) == 0
