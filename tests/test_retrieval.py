"""Accuracy tests for the deterministic retrieval/scoring engine.

These are hand-labeled: for each scenario we know, from the curated
dataset, what the intuitively-correct top pick is. This is the "verify
the accuracy claim" step - it would catch a regression in the scoring
weights/logic that silently made recommendations worse.
"""
from backend.catalog import get_catalog
from backend.retrieval import top_candidates


def test_exact_hardware_and_skill_match_wins():
    catalog = get_catalog()
    results = top_candidates(
        skill_level="beginner",
        available_components=["arduino_uno", "ultrasonic_hcsr04", "buzzer"],
        time_budget_hours=4,
        interests="obstacle alarms",
        k=3,
        catalog=catalog,
    )
    assert results[0].project["id"] == "proj-001"  # Ultrasonic Parking / Obstacle Distance Alarm
    assert results[0].hardware_score >= 0.75  # all required parts covered (optional lcd not owned)


def test_advanced_predictive_maintenance_match():
    catalog = get_catalog()
    results = top_candidates(
        skill_level="advanced",
        available_components=["esp32", "mpu6050_imu", "current_sensor_acs712"],
        time_budget_hours=12,
        interests="predictive maintenance and vibration analysis",
        k=3,
        catalog=catalog,
    )
    assert results[0].project["id"] == "proj-020"  # Vibration-Based Predictive Maintenance Node
    assert results[0].hardware_score >= 0.75  # all required parts covered (optional lora not owned)
    assert results[0].skill_score == 1.0


def test_skill_mismatch_is_penalized():
    catalog = get_catalog()
    # A beginner with advanced-only hardware (PLC) should not get full skill score
    # on the advanced PLC project.
    results = top_candidates(
        skill_level="beginner",
        available_components=["plc_basic", "relay_module", "buzzer"],
        time_budget_hours=10,
        interests="sequencing",
        k=5,
        catalog=catalog,
    )
    plc_result = next(r for r in results if r.project["id"] == "proj-021")
    assert plc_result.skill_score < 1.0


def test_time_budget_penalizes_long_projects():
    catalog = get_catalog()
    results = top_candidates(
        skill_level="advanced",
        available_components=[],
        time_budget_hours=2,  # far too short for any advanced build
        interests="",
        k=28,
        catalog=catalog,
    )
    by_id = {r.project["id"]: r for r in results}
    # proj-018 (16h CNC plotter) should score worse on time than a 4h beginner project
    assert by_id["proj-018"].time_score < 0.3


def test_missing_components_are_actually_missing():
    catalog = get_catalog()
    available = ["arduino_uno", "buzzer"]
    results = top_candidates(
        skill_level="beginner",
        available_components=available,
        time_budget_hours=5,
        interests="",
        k=28,
        catalog=catalog,
    )
    for r in results:
        # nothing "missing" should actually be in the available set
        for missing_id in r.missing_required:
            assert missing_id not in available
        # nothing "have" should be absent from the available set
        for have_id in r.have_required:
            assert have_id in available


def test_scores_are_sorted_descending():
    catalog = get_catalog()
    results = top_candidates(
        skill_level="intermediate",
        available_components=["esp32", "dht22_temp_humidity"],
        time_budget_hours=8,
        interests="iot",
        k=10,
        catalog=catalog,
    )
    scores = [r.total_score for r in results]
    assert scores == sorted(scores, reverse=True)
