"""Tests for free-text hardware normalization - this is the first line of
"human proofing": messy user input (typos, casing, extra words) must map to
real catalog components instead of silently failing or crashing."""
from backend.catalog import get_catalog


def test_alias_match():
    c = get_catalog()
    assert c.normalize_component("arduino") == "arduino_uno"
    assert c.normalize_component("HC-SR04") == "ultrasonic_hcsr04"


def test_case_insensitive():
    c = get_catalog()
    assert c.normalize_component("ESP32") == "esp32"
    assert c.normalize_component("esp32") == "esp32"


def test_unmatched_returns_none_not_garbage():
    c = get_catalog()
    assert c.normalize_component("a rocket engine") is None
    assert c.normalize_component("") is None


def test_normalize_many_dedupes_and_splits_commas():
    c = get_catalog()
    result = c.normalize_many(["arduino, esp32", "servo motor", "arduino"])
    assert result.count("arduino_uno") == 1
    assert "esp32" in result
    assert "servo_motor_sg90" in result
