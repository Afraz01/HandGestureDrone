import math
from types import SimpleNamespace

import pytest

from hand_throttle import (
    CLIMB_DAC,
    DESCEND_DAC,
    DOWN_VECTOR,
    EMASmoother,
    HOVER_DAC,
    UP_VECTOR,
    classify_gesture,
    classify_vector_direction,
    clamp_dac,
    is_within_direction,
    voltage_to_dac,
)


def vector_from_vertical(angle_deg, direction):
    angle_rad = math.radians(angle_deg)
    y_sign = -1.0 if direction == "UP" else 1.0
    return (math.sin(angle_rad), y_sign * math.cos(angle_rad))


def test_voltage_to_dac_calibration_values():
    assert voltage_to_dac(0.00) == 0
    assert voltage_to_dac(1.65) == 2048
    assert voltage_to_dac(3.30) == 4095
    assert DESCEND_DAC == 0
    assert HOVER_DAC == 2048
    assert CLIMB_DAC == 4095


@pytest.mark.parametrize("angle", [0.0, 30.0, 44.0, 45.0])
def test_up_angles_inside_inclusive_45_degree_cone(angle):
    vector = vector_from_vertical(angle, "UP")
    assert is_within_direction(vector, UP_VECTOR)
    assert classify_vector_direction(vector)["label"] == "UP"


def test_46_degrees_is_not_up():
    vector = vector_from_vertical(46.0, "UP")
    assert not is_within_direction(vector, UP_VECTOR)
    assert classify_vector_direction(vector)["label"] == "SIDE"


@pytest.mark.parametrize("angle", [0.0, 30.0, 44.0, 45.0])
def test_down_angles_inside_inclusive_45_degree_cone(angle):
    vector = vector_from_vertical(angle, "DOWN")
    assert is_within_direction(vector, DOWN_VECTOR)
    assert classify_vector_direction(vector)["label"] == "DOWN"


def test_46_degrees_is_not_down():
    vector = vector_from_vertical(46.0, "DOWN")
    assert not is_within_direction(vector, DOWN_VECTOR)
    assert classify_vector_direction(vector)["label"] == "SIDE"


@pytest.mark.parametrize("sideways", [(1.0, 0.0), (-1.0, 0.0)])
def test_sideways_vectors_are_ambiguous(sideways):
    assert not is_within_direction(sideways, UP_VECTOR)
    assert not is_within_direction(sideways, DOWN_VECTOR)
    assert classify_vector_direction(sideways)["label"] == "SIDE"


def test_dac_clamping():
    assert clamp_dac(-100) == 0
    assert clamp_dac(5000) == 4095
    assert voltage_to_dac(-1.0) == 0
    assert voltage_to_dac(5.0) == 4095


def test_ema_smoothing():
    smoother = EMASmoother(alpha=0.3, initial=HOVER_DAC)
    first = smoother.update(CLIMB_DAC)
    assert first == pytest.approx(0.3 * CLIMB_DAC + 0.7 * HOVER_DAC)
    second = smoother.update(CLIMB_DAC)
    assert second == pytest.approx(0.3 * CLIMB_DAC + 0.7 * first)
    assert HOVER_DAC < first < second < CLIMB_DAC


def make_hand(direction):
    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    finger_indices = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    x_positions = (0.35, 0.45, 0.55, 0.65)
    y_positions = (0.70, 0.55, 0.40, 0.25) if direction == "UP" else (0.30, 0.45, 0.60, 0.75)
    for indices, x in zip(finger_indices, x_positions):
        for index, y in zip(indices, y_positions):
            landmarks[index] = SimpleNamespace(x=x, y=y, z=0.0)
    return SimpleNamespace(landmark=landmarks)


def make_folded_hand():
    landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
    finger_indices = ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
    x_positions = (0.35, 0.45, 0.55, 0.65)
    # Each PIP and DIP forms a strong bend, independent of global orientation.
    offsets = ((0.00, 0.12), (0.00, 0.00), (0.09, 0.05), (0.02, 0.12))
    for indices, base_x in zip(finger_indices, x_positions):
        for index, (dx, dy) in zip(indices, offsets):
            landmarks[index] = SimpleNamespace(x=base_x + dx, y=0.45 + dy, z=0.0)
    return SimpleNamespace(landmark=landmarks)


def test_all_four_extended_up_classifies_climb():
    result = classify_gesture(make_hand("UP"), tracking_confidence=0.99)
    assert result["state"] == "CLIMB"
    assert all(result["extended"].values())


def test_all_four_extended_down_classifies_descend():
    result = classify_gesture(make_hand("DOWN"), tracking_confidence=0.99)
    assert result["state"] == "DESCEND"
    assert all(result["extended"].values())


def test_no_hand_and_low_confidence_fail_to_hover():
    assert classify_gesture(None)["state"] == "HOVER"
    result = classify_gesture(make_hand("UP"), tracking_confidence=0.1)
    assert result["state"] == "HOVER"
    assert result["reason"] == "Tracking confidence insufficient"


def test_folded_fist_has_hover_priority():
    result = classify_gesture(make_folded_hand(), tracking_confidence=0.99)
    assert result["state"] == "HOVER"
    assert "folded" in result["reason"].lower() or "fist" in result["reason"].lower()
