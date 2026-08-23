"""Webcam hand-gesture throttle control for an HS210 transmitter.

The desktop detects only three throttle states: CLIMB, HOVER, and DESCEND.
Anything uncertain is deliberately classified as HOVER.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any, Iterable, Sequence

import numpy as np

try:
    import cv2
except ImportError:  # Pure logic can still be imported by tests.
    cv2 = None  # type: ignore[assignment]

try:
    import mediapipe as mp
except ImportError:
    mp = None  # type: ignore[assignment]

try:
    import serial
    from serial.tools import list_ports
except ImportError:
    serial = None  # type: ignore[assignment]
    list_ports = None  # type: ignore[assignment]


# Initial calibration: change these three voltages after careful re-measurement.
VREF = 3.3

DESCEND_VOLTAGE = 0.00
HOVER_VOLTAGE = 1.65
CLIMB_VOLTAGE = 3.30

DAC_MIN = 0
DAC_MAX = 4095


def clamp_dac(value: int | float) -> int:
    """Round and clamp a DAC value to the MCP4728's 12-bit range."""
    return max(DAC_MIN, min(DAC_MAX, round(value)))


def voltage_to_dac(desired_voltage: float, vref: float = VREF) -> int:
    """Convert volts to a clamped 12-bit DAC code."""
    if not math.isfinite(desired_voltage):
        raise ValueError("desired_voltage must be finite")
    if not math.isfinite(vref) or vref <= 0.0:
        raise ValueError("vref must be a positive finite value")
    return clamp_dac((desired_voltage / vref) * DAC_MAX)


DESCEND_DAC = voltage_to_dac(DESCEND_VOLTAGE)
HOVER_DAC = voltage_to_dac(HOVER_VOLTAGE)
CLIMB_DAC = voltage_to_dac(CLIMB_VOLTAGE)

FINGER_ANGLE_THRESHOLD_DEG = 45.0
FINGER_EXTENSION_ANGLE_DEG = 150.0
FINGER_EXTENSION_LENGTH_RATIO = 1.45
MIN_HANDEDNESS_CONFIDENCE = 0.60

EMA_ALPHA = 0.3
SERIAL_BAUD = 115200
SERIAL_RATE_HZ = 30.0
SERIAL_RECONNECT_SECONDS = 2.0
SHUTDOWN_HOVER_SECONDS = 0.40

UP_VECTOR = (0.0, -1.0)  # MediaPipe image coordinates: +y points down.
DOWN_VECTOR = (0.0, 1.0)

FINGER_LANDMARKS = {
    "INDEX": {"mcp": 5, "pip": 6, "dip": 7, "tip": 8},
    "MIDDLE": {"mcp": 9, "pip": 10, "dip": 11, "tip": 12},
    "RING": {"mcp": 13, "pip": 14, "dip": 15, "tip": 16},
    "PINKY": {"mcp": 17, "pip": 18, "dip": 19, "tip": 20},
}

STATE_VOLTAGES = {
    "CLIMB": CLIMB_VOLTAGE,
    "HOVER": HOVER_VOLTAGE,
    "DESCEND": DESCEND_VOLTAGE,
}

STATE_COLORS = {
    "CLIMB": (40, 210, 40),       # BGR green
    "HOVER": (160, 160, 160),     # BGR gray
    "DESCEND": (40, 40, 230),     # BGR red
}

UI_PANEL_COLOR = (18, 21, 28)
UI_BORDER_COLOR = (75, 82, 94)
UI_TEXT_COLOR = (244, 246, 250)
UI_MUTED_COLOR = (174, 182, 194)
UI_AMBER_COLOR = (35, 185, 245)
DIRECTION_COLORS = {
    "UP": STATE_COLORS["CLIMB"],
    "DOWN": STATE_COLORS["DESCEND"],
    "SIDE": UI_AMBER_COLOR,
}


def normalize_vector(vector: Sequence[float]) -> np.ndarray | None:
    """Return a unit vector, or None for a zero/non-finite vector."""
    array = np.asarray(vector, dtype=float)
    magnitude = float(np.linalg.norm(array))
    if not math.isfinite(magnitude) or magnitude <= 1e-12:
        return None
    return array / magnitude


def angle_between_vectors(
    vector: Sequence[float], reference: Sequence[float]
) -> float:
    """Return the smaller angle in degrees between two vectors."""
    normalized_vector = normalize_vector(vector)
    normalized_reference = normalize_vector(reference)
    if normalized_vector is None or normalized_reference is None:
        return math.inf
    dot = float(np.dot(normalized_vector, normalized_reference))
    dot = float(np.clip(dot, -1.0, 1.0))
    return math.degrees(math.acos(dot))


def is_within_direction(
    vector: Sequence[float],
    reference: Sequence[float],
    threshold_deg: float = FINGER_ANGLE_THRESHOLD_DEG,
) -> bool:
    """Return True when vector is within threshold_deg of reference."""
    return angle_between_vectors(vector, reference) <= threshold_deg + 1e-9


def classify_vector_direction(
    vector: Sequence[float],
    threshold_deg: float = FINGER_ANGLE_THRESHOLD_DEG,
) -> dict[str, float | str]:
    """Describe a finger vector relative to vertical UP and DOWN."""
    up_angle = angle_between_vectors(vector, UP_VECTOR)
    down_angle = angle_between_vectors(vector, DOWN_VECTOR)
    if up_angle <= threshold_deg + 1e-9:
        label = "UP"
        nearest_angle = up_angle
    elif down_angle <= threshold_deg + 1e-9:
        label = "DOWN"
        nearest_angle = down_angle
    else:
        label = "SIDE"
        nearest_angle = min(up_angle, down_angle)
    return {
        "up": up_angle,
        "down": down_angle,
        "label": label,
        "nearest_angle": nearest_angle,
    }


def _landmark_point(hand_landmarks: Any, index: int, dimensions: int) -> np.ndarray:
    landmark = hand_landmarks.landmark[index]
    values = (landmark.x, landmark.y, landmark.z)
    return np.asarray(values[:dimensions], dtype=float)


def _joint_angle(point_a: np.ndarray, joint: np.ndarray, point_c: np.ndarray) -> float:
    return angle_between_vectors(point_a - joint, point_c - joint)


def is_finger_extended(hand_landmarks: Any, finger_name: str) -> bool:
    """Detect extension using PIP/DIP straightness plus length geometry.

    This calculation is orientation-independent; it does not assume that an
    extended finger has a smaller y coordinate than its joints.
    """
    indices = FINGER_LANDMARKS[finger_name]
    mcp = _landmark_point(hand_landmarks, indices["mcp"], 3)
    pip = _landmark_point(hand_landmarks, indices["pip"], 3)
    dip = _landmark_point(hand_landmarks, indices["dip"], 3)
    tip = _landmark_point(hand_landmarks, indices["tip"], 3)

    pip_angle = _joint_angle(mcp, pip, dip)
    dip_angle = _joint_angle(pip, dip, tip)
    proximal_length = float(np.linalg.norm(pip - mcp))
    reach = float(np.linalg.norm(tip - mcp))
    reach_ratio = reach / proximal_length if proximal_length > 1e-12 else 0.0

    return (
        pip_angle >= FINGER_EXTENSION_ANGLE_DEG
        and dip_angle >= FINGER_EXTENSION_ANGLE_DEG
        and reach_ratio >= FINGER_EXTENSION_LENGTH_RATIO
    )


def is_fist(extended: dict[str, bool]) -> bool:
    """Treat three or more folded main fingers as a fist/closed hand."""
    folded_count = sum(not extended[name] for name in FINGER_LANDMARKS)
    return folded_count >= 3


def get_finger_direction(hand_landmarks: Any, finger_name: str) -> np.ndarray:
    """Return the 2-D PIP-to-tip image vector for one main finger."""
    indices = FINGER_LANDMARKS[finger_name]
    pip = _landmark_point(hand_landmarks, indices["pip"], 2)
    tip = _landmark_point(hand_landmarks, indices["tip"], 2)
    return tip - pip


def hover_result(reason: str, confidence: float = 0.0) -> dict[str, Any]:
    """Construct a safe HOVER classification when no usable hand exists."""
    return {
        "state": "HOVER",
        "reason": reason,
        "finger_angles": {},
        "extended": {},
        "vectors": {},
        "confidence": confidence,
    }


def classify_gesture(
    hand_landmarks: Any,
    tracking_confidence: float = 1.0,
    angle_threshold_deg: float = FINGER_ANGLE_THRESHOLD_DEG,
) -> dict[str, Any]:
    """Classify one detected hand using the deterministic safety priority."""
    if hand_landmarks is None:
        return hover_result("No hand detected")
    if tracking_confidence < MIN_HANDEDNESS_CONFIDENCE:
        return hover_result(
            "Tracking confidence insufficient", confidence=tracking_confidence
        )

    extended = {
        name: is_finger_extended(hand_landmarks, name)
        for name in FINGER_LANDMARKS
    }
    vectors = {
        name: get_finger_direction(hand_landmarks, name)
        for name in FINGER_LANDMARKS
    }
    finger_angles = {
        name: classify_vector_direction(vector, angle_threshold_deg)
        for name, vector in vectors.items()
    }

    state = "HOVER"
    if is_fist(extended):
        reason = "Fist / most fingers folded"
    elif not all(extended.values()):
        reason = "One or more fingers are folded"
    elif all(
        float(finger_angles[name]["up"]) <= angle_threshold_deg + 1e-9
        for name in FINGER_LANDMARKS
    ):
        state = "CLIMB"
        reason = "All four fingers extended and inside UP cone"
    elif all(
        float(finger_angles[name]["down"]) <= angle_threshold_deg + 1e-9
        for name in FINGER_LANDMARKS
    ):
        state = "DESCEND"
        reason = "All four fingers extended and inside DOWN cone"
    else:
        labels = {str(data["label"]) for data in finger_angles.values()}
        if "UP" in labels and "DOWN" in labels:
            reason = "Finger directions do not agree"
        else:
            reason = f"Finger direction exceeds {angle_threshold_deg:.1f} degree threshold"

    return {
        "state": state,
        "reason": reason,
        "finger_angles": finger_angles,
        "extended": extended,
        "vectors": vectors,
        "confidence": tracking_confidence,
    }


class EMASmoother:
    """Exponential moving average initialized to the neutral HOVER command."""

    def __init__(self, alpha: float = EMA_ALPHA, initial: float = HOVER_DAC) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in the interval (0, 1]")
        self.alpha = alpha
        self.value = float(initial)

    def update(self, target: float) -> float:
        self.value = self.alpha * target + (1.0 - self.alpha) * self.value
        return self.value

    @property
    def rounded(self) -> int:
        return clamp_dac(self.value)

    def reset(self, value: float = HOVER_DAC) -> None:
        self.value = float(value)


def _available_ports() -> list[Any]:
    return list(list_ports.comports()) if list_ports is not None else []


def find_pico_port() -> str | None:
    """Find the most likely Pico/MicroPython USB serial port."""
    candidates: list[tuple[int, str]] = []
    for port in _available_ports():
        description = " ".join(
            str(value or "")
            for value in (
                getattr(port, "description", ""),
                getattr(port, "manufacturer", ""),
                getattr(port, "product", ""),
                getattr(port, "hwid", ""),
            )
        ).lower()
        score = 0
        if getattr(port, "vid", None) == 0x2E8A:  # Raspberry Pi USB VID
            score += 100
        if "pico" in description:
            score += 40
        if "micropython" in description:
            score += 30
        if "raspberry pi" in description:
            score += 20
        if score:
            candidates.append((score, port.device))
    return max(candidates, default=(0, None), key=lambda item: item[0])[1]


def print_available_ports() -> None:
    ports = _available_ports()
    print("Automatic Pico serial-port discovery failed.")
    if ports:
        print("Available serial ports:")
        for port in ports:
            print(f"  {port.device}: {port.description}")
    else:
        print("No serial ports were found.")
    print("Specify the Pico manually, for example: python hand_throttle.py --port COM5")


class SerialThrottleController:
    """Rate-limited, reconnecting USB serial throttle transport."""

    def __init__(
        self,
        enabled: bool,
        port: str | None = None,
        baud: int = SERIAL_BAUD,
        rate_hz: float = SERIAL_RATE_HZ,
    ) -> None:
        self.enabled = enabled
        self.requested_port = port
        self.baud = baud
        self.rate_hz = rate_hz
        self.connection: Any = None
        self.active_port: str | None = None
        self.last_connect_attempt = -math.inf
        self.last_send_time = -math.inf
        self.last_error = ""
        self._reported_missing_port = False
        if enabled:
            self.maybe_reconnect(time.monotonic(), force=True)

    @property
    def connected(self) -> bool:
        return self.connection is not None and bool(self.connection.is_open)

    @property
    def status(self) -> str:
        if not self.enabled:
            return "Disabled (--no-serial)"
        if self.connected:
            return f"Connected ({self.active_port})"
        if self.last_error:
            return f"Disconnected ({self.last_error})"
        return "Disconnected"

    def command_due(self, now: float) -> bool:
        return now - self.last_send_time >= 1.0 / self.rate_hz

    def maybe_reconnect(self, now: float, force: bool = False) -> None:
        if not self.enabled or self.connected:
            return
        if not force and now - self.last_connect_attempt < SERIAL_RECONNECT_SECONDS:
            return
        self.last_connect_attempt = now
        port = self.requested_port or find_pico_port()
        if not port:
            self.last_error = "no Pico port found"
            if not self._reported_missing_port:
                print_available_ports()
                self._reported_missing_port = True
            return
        try:
            self.connection = serial.Serial(
                port=port,
                baudrate=self.baud,
                timeout=0,
                write_timeout=0.2,
            )
            self.active_port = port
            self.last_error = ""
            self._reported_missing_port = False
            print(f"Serial connected: {port} at {self.baud} baud")
        except (OSError, serial.SerialException) as exc:
            self.connection = None
            self.active_port = None
            self.last_error = str(exc)
            print(f"Serial connection failed: {exc}")

    def send_throttle_command(
        self, dac_value: int | float, now: float | None = None, force: bool = False
    ) -> bool:
        """Send one validated THROTTLE line; return False on disconnection."""
        timestamp = time.monotonic() if now is None else now
        if not self.enabled:
            self.last_send_time = timestamp
            return True
        if not self.connected:
            return False
        if not force and not self.command_due(timestamp):
            return True

        packet = f"THROTTLE,{clamp_dac(dac_value)}\n".encode("ascii")
        try:
            self.connection.write(packet)
            self.last_send_time = timestamp
            return True
        except (OSError, serial.SerialException) as exc:
            self.last_error = str(exc)
            print(f"Serial connection lost: {exc}")
            self.close()
            return False

    def send_hover_burst(self, duration: float = SHUTDOWN_HOVER_SECONDS) -> None:
        """Best-effort neutral packets before closing the serial connection."""
        if not self.enabled or not self.connected:
            return
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline and self.connected:
            self.send_throttle_command(HOVER_DAC, force=True)
            time.sleep(0.05)

    def close(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            except (OSError, serial.SerialException):
                pass
        self.connection = None
        self.active_port = None


class HandGestureDetector:
    """MediaPipe Hands wrapper that prefers the highest-confidence right hand."""

    def __init__(self) -> None:
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.70,
            min_tracking_confidence=0.70,
        )
        self._drawing = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles

    def process(self, bgr_frame: np.ndarray) -> tuple[Any, str, float]:
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self._hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return None, "None", 0.0

        choices: list[tuple[bool, float, Any, str]] = []
        for index, landmarks in enumerate(results.multi_hand_landmarks):
            label = "Unknown"
            score = 0.0
            if results.multi_handedness and index < len(results.multi_handedness):
                classification = results.multi_handedness[index].classification[0]
                label = classification.label
                score = float(classification.score)
            choices.append((label.lower() == "right", score, landmarks, label))

        _, score, landmarks, label = max(
            choices, key=lambda choice: (choice[0], choice[1])
        )
        return landmarks, label, score

    def draw_landmarks(self, frame: np.ndarray, hand_landmarks: Any) -> None:
        if hand_landmarks is None:
            return
        self._drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp.solutions.hands.HAND_CONNECTIONS,
            self._styles.get_default_hand_landmarks_style(),
            self._styles.get_default_hand_connections_style(),
        )

    def close(self) -> None:
        self._hands.close()


def _pixel_point(hand_landmarks: Any, index: int, width: int, height: int) -> tuple[int, int]:
    landmark = hand_landmarks.landmark[index]
    return int(landmark.x * width), int(landmark.y * height)


def _shade_panel(
    frame: np.ndarray,
    bounds: tuple[int, int, int, int],
    alpha: float = 0.76,
    border_color: tuple[int, int, int] = UI_BORDER_COLOR,
) -> None:
    """Draw one translucent, bordered UI panel without copying the whole frame."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bounds
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    region = frame[y1:y2, x1:x2]
    tint = np.full_like(region, UI_PANEL_COLOR)
    cv2.addWeighted(tint, alpha, region, 1.0 - alpha, 0.0, region)
    cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), border_color, 1, cv2.LINE_AA)


def _wrapped_lines(text: str, max_characters: int, max_lines: int = 2) -> list[str]:
    """Wrap short status text for the compact on-camera panels."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_characters:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
            if len(lines) == max_lines - 1:
                break
    if current and len(lines) < max_lines:
        lines.append(current)
    consumed = " ".join(lines)
    if len(consumed) < len(text) and lines:
        lines[-1] = lines[-1][: max(1, max_characters - 3)].rstrip() + "..."
    return lines


def _serial_badge(status: str) -> tuple[str, tuple[int, int, int]]:
    if status.startswith("Connected"):
        port = status.partition("(")[2].rstrip(")") or "ONLINE"
        return f"SERIAL {port}", STATE_COLORS["CLIMB"]
    if status.startswith("Disabled"):
        return "SERIAL OFF", UI_MUTED_COLOR
    return "SERIAL LOST", STATE_COLORS["DESCEND"]


def draw_direction_cones(frame: np.ndarray, top: int = 98) -> None:
    """Draw a small, labeled UP/DOWN plus-or-minus-45-degree guide."""
    height, width = frame.shape[:2]
    if width < 480 or height < 300:
        return
    box_width = 112
    box_height = 142
    x1 = width - box_width - 12
    y1 = min(top, max(8, height - box_height - 42))
    _shade_panel(frame, (x1, y1, width - 12, y1 + box_height), alpha=0.68)

    cv2.putText(
        frame,
        "45 DEG GUIDE",
        (x1 + 10, y1 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        UI_MUTED_COLOR,
        1,
        cv2.LINE_AA,
    )
    origin = (x1 + box_width // 2, y1 + 76)
    length = 39
    cv2.circle(frame, origin, 3, UI_TEXT_COLOR, -1, cv2.LINE_AA)
    for sign, label, color in (
        (-1, "UP", STATE_COLORS["CLIMB"]),
        (1, "DOWN", STATE_COLORS["DESCEND"]),
    ):
        cv2.line(frame, origin, (origin[0], origin[1] + sign * length), color, 2, cv2.LINE_AA)
        for x_sign in (-1, 1):
            offset = int(length / math.sqrt(2))
            cv2.line(
                frame,
                origin,
                (origin[0] + x_sign * offset, origin[1] + sign * offset),
                color,
                1,
                cv2.LINE_AA,
            )
        label_y = y1 + 39 if sign < 0 else y1 + 132
        cv2.putText(
            frame,
            label,
            (x1 + 8, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            color,
            1,
            cv2.LINE_AA,
        )


def draw_debug_overlay(
    frame: np.ndarray,
    hand_landmarks: Any,
    result: dict[str, Any],
    target_voltage: float,
    target_dac: int,
    smoothed_dac: int,
    serial_status: str,
    handedness: str,
    fps: float,
    angle_threshold_deg: float,
) -> None:
    """Draw a compact dashboard while leaving the center of the camera visible."""
    height, width = frame.shape[:2]
    state = result["state"]
    state_color = STATE_COLORS[state]
    header_height = 88 if height >= 540 else 78
    footer_height = 34
    margin = 12

    left_width = min(370, max(270, int(width * 0.31)))
    right_width = min(390, max(285, int(width * 0.33)))
    if left_width + right_width + margin * 3 > width:
        available = max(300, width - margin * 3)
        left_width = available // 2
        right_width = available - left_width
    tracking_height = 150 if height >= 600 else 134
    fingers_height = 196 if height >= 600 else 180
    tracking_top = max(header_height + margin, height - footer_height - tracking_height - margin)
    fingers_top = max(header_height + margin, height - footer_height - fingers_height - margin)
    right_x = width - right_width - margin

    _shade_panel(frame, (0, 0, width, header_height), alpha=0.80, border_color=state_color)
    _shade_panel(
        frame,
        (margin, tracking_top, margin + left_width, tracking_top + tracking_height),
        alpha=0.73,
    )
    _shade_panel(
        frame,
        (right_x, fingers_top, right_x + right_width, fingers_top + fingers_height),
        alpha=0.73,
    )
    _shade_panel(frame, (0, height - footer_height, width, height), alpha=0.82)
    cv2.rectangle(frame, (0, header_height - 4), (width, header_height - 1), state_color, -1)

    compact = width < 820
    state_scale = 1.05 if compact else 1.25
    cv2.putText(
        frame,
        "HAND GESTURE DRONE",
        (18, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42 if compact else 0.48,
        UI_MUTED_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        state,
        (18, 63 if header_height >= 88 else 58),
        cv2.FONT_HERSHEY_DUPLEX,
        state_scale,
        state_color,
        2,
        cv2.LINE_AA,
    )

    target_x = int(width * 0.27)
    output_x = int(width * 0.48)
    status_x = int(width * 0.70)
    small_scale = 0.37 if compact else 0.43
    value_scale = 0.66 if compact else 0.78
    smoothed_voltage = (smoothed_dac / DAC_MAX) * VREF
    for label, value, detail, x in (
        ("TARGET", f"{target_voltage:.2f} V", f"DAC {target_dac}", target_x),
        ("OUTPUT", f"{smoothed_voltage:.2f} V", f"DAC {smoothed_dac}", output_x),
    ):
        cv2.putText(frame, label, (x, 20), cv2.FONT_HERSHEY_SIMPLEX, small_scale, UI_MUTED_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame, value, (x, 51), cv2.FONT_HERSHEY_DUPLEX, value_scale, UI_TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.putText(frame, detail, (x, 72), cv2.FONT_HERSHEY_SIMPLEX, small_scale, UI_MUTED_COLOR, 1, cv2.LINE_AA)

    serial_label, serial_color = _serial_badge(serial_status)
    cv2.circle(frame, (status_x, 18), 5, serial_color, -1, cv2.LINE_AA)
    cv2.putText(frame, serial_label, (status_x + 12, 23), cv2.FONT_HERSHEY_SIMPLEX, small_scale, serial_color, 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"{fps:4.1f} FPS  |  {handedness} {float(result['confidence']) * 100:.0f}%",
        (status_x, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        small_scale,
        UI_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"ANGLE LIMIT {angle_threshold_deg:.0f} DEG",
        (status_x, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        small_scale,
        UI_MUTED_COLOR,
        1,
        cv2.LINE_AA,
    )

    panel_x = margin + 14
    cv2.putText(frame, "TRACKING / SAFETY", (panel_x, tracking_top + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.47, UI_MUTED_COLOR, 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"Hand: {handedness}   Confidence: {float(result['confidence']) * 100:.0f}%",
        (panel_x, tracking_top + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        UI_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(frame, "WHY", (panel_x, tracking_top + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.38, UI_MUTED_COLOR, 1, cv2.LINE_AA)
    reason_width = max(18, int((left_width - 28) / 8))
    for line_index, line in enumerate(_wrapped_lines(str(result["reason"]), reason_width)):
        cv2.putText(
            frame,
            line,
            (panel_x, tracking_top + 100 + line_index * 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            state_color,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(frame, "FINGERS", (right_x + 14, fingers_top + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.47, UI_MUTED_COLOR, 1, cv2.LINE_AA)
    cv2.putText(frame, "DIR", (right_x + 104, fingers_top + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.34, UI_MUTED_COLOR, 1, cv2.LINE_AA)
    cv2.putText(frame, "ANGLE", (right_x + 162, fingers_top + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.34, UI_MUTED_COLOR, 1, cv2.LINE_AA)
    cv2.putText(frame, "SHAPE", (right_x + 228, fingers_top + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.34, UI_MUTED_COLOR, 1, cv2.LINE_AA)

    row_y = fingers_top + 55
    row_step = 33 if fingers_height >= 190 else 30
    for name in FINGER_LANDMARKS:
        data = result["finger_angles"].get(name)
        extended = bool(result["extended"].get(name))
        if data:
            direction = str(data["label"])
            angle_text = f"{float(data['nearest_angle']):.1f} deg"
            direction_color = DIRECTION_COLORS.get(direction, UI_MUTED_COLOR)
            shape_text = "EXT" if extended else "FOLD"
        else:
            direction = "--"
            angle_text = "--"
            direction_color = UI_MUTED_COLOR
            shape_text = "--"
        cv2.putText(frame, name, (right_x + 14, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, UI_TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.circle(frame, (right_x + 98, row_y - 5), 4, direction_color, -1, cv2.LINE_AA)
        cv2.putText(frame, direction, (right_x + 108, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, direction_color, 1, cv2.LINE_AA)
        cv2.putText(frame, angle_text, (right_x + 162, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.41, direction_color, 1, cv2.LINE_AA)
        cv2.putText(
            frame,
            shape_text,
            (right_x + 228, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.41,
            STATE_COLORS["CLIMB"] if extended else UI_MUTED_COLOR,
            1,
            cv2.LINE_AA,
        )
        row_y += row_step

    if hand_landmarks is not None:
        for name, indices in FINGER_LANDMARKS.items():
            pip = _pixel_point(hand_landmarks, indices["pip"], width, height)
            tip = _pixel_point(hand_landmarks, indices["tip"], width, height)
            data = result["finger_angles"].get(name)
            vector_color = UI_MUTED_COLOR
            if data and result["extended"].get(name):
                vector_color = DIRECTION_COLORS.get(str(data["label"]), UI_MUTED_COLOR)
            cv2.arrowedLine(frame, pip, tip, vector_color, 3, cv2.LINE_AA, tipLength=0.20)
            if data:
                angle_label = f"{float(data['nearest_angle']):.0f}"
                cv2.putText(
                    frame,
                    angle_label,
                    (tip[0] + 4, tip[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    (10, 10, 10),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    frame,
                    angle_label,
                    (tip[0] + 4, tip[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.46,
                    vector_color,
                    1,
                    cv2.LINE_AA,
                )

    draw_direction_cones(frame, top=header_height + margin)
    cv2.putText(
        frame,
        "Q / ESC   SAFE HOVER + EXIT",
        (margin, height - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        UI_TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    footer_right = "ALL FOUR FINGERS MUST PASS THE 45 DEG RULE"
    footer_size = cv2.getTextSize(footer_right, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)[0]
    cv2.putText(
        frame,
        footer_right,
        (max(margin, width - footer_size[0] - margin), height - 11),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        UI_MUTED_COLOR,
        1,
        cv2.LINE_AA,
    )


def print_debug(
    result: dict[str, Any],
    target_voltage: float,
    target_dac: int,
    smoothed_dac: int,
    serial_status: str,
    fps: float,
) -> None:
    print(f"Gesture: {result['state']}")
    if result["finger_angles"]:
        for name in FINGER_LANDMARKS:
            data = result["finger_angles"][name]
            print(f"{name.title()}: {float(data['nearest_angle']):.1f} deg {data['label']}")
    else:
        print(f"Reason: {result['reason']}")
    print(f"Target voltage: {target_voltage:.2f} V")
    print(f"Target DAC: {target_dac}")
    print(f"Smoothed DAC: {smoothed_dac}")
    print(f"Serial: {serial_status}")
    print(f"FPS: {fps:.1f}\n")


def run_camera(
    controller: SerialThrottleController,
    camera_index: int,
    angle_threshold_deg: float,
) -> None:
    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open webcam index {camera_index}")

    detector = HandGestureDetector()
    smoother = EMASmoother()
    last_control_update = -math.inf
    last_debug = -math.inf
    previous_frame_time = time.monotonic()
    fps = 0.0
    consecutive_camera_failures = 0

    try:
        while True:
            ok, frame = capture.read()
            now = time.monotonic()
            controller.maybe_reconnect(now)
            if not ok:
                consecutive_camera_failures += 1
                result = hover_result("Webcam frame unavailable")
                if now - last_control_update >= 1.0 / SERIAL_RATE_HZ:
                    smoother.update(HOVER_DAC)
                    controller.send_throttle_command(smoother.rounded, now)
                    last_control_update = now
                if consecutive_camera_failures >= 30:
                    raise RuntimeError(
                        "Webcam stopped providing frames; throttle returned to HOVER"
                    )
                time.sleep(0.01)
                continue
            consecutive_camera_failures = 0

            # MediaPipe handedness expects a mirrored/selfie image.
            frame = cv2.flip(frame, 1)
            hand_landmarks, handedness, confidence = detector.process(frame)
            result = classify_gesture(
                hand_landmarks,
                tracking_confidence=confidence,
                angle_threshold_deg=angle_threshold_deg,
            )
            if controller.enabled and not controller.connected:
                result = hover_result("Serial connection unavailable", confidence)

            target_voltage = STATE_VOLTAGES[result["state"]]
            target_dac = voltage_to_dac(target_voltage)

            if now - last_control_update >= 1.0 / SERIAL_RATE_HZ:
                smoother.update(target_dac)
                sent = controller.send_throttle_command(smoother.rounded, now)
                last_control_update = now
                if controller.enabled and not sent:
                    result = hover_result("Serial connection lost", confidence)
                    target_voltage = HOVER_VOLTAGE
                    target_dac = HOVER_DAC
                    smoother.update(HOVER_DAC)

            frame_delta = max(now - previous_frame_time, 1e-9)
            instantaneous_fps = 1.0 / frame_delta
            fps = instantaneous_fps if fps == 0.0 else 0.10 * instantaneous_fps + 0.90 * fps
            previous_frame_time = now

            detector.draw_landmarks(frame, hand_landmarks)
            draw_debug_overlay(
                frame,
                hand_landmarks,
                result,
                target_voltage,
                target_dac,
                smoother.rounded,
                controller.status,
                handedness,
                fps,
                angle_threshold_deg,
            )
            cv2.imshow("Hand Gesture Drone", frame)

            if now - last_debug >= 1.0:
                print_debug(
                    result,
                    target_voltage,
                    target_dac,
                    smoother.rounded,
                    controller.status,
                    fps,
                )
                last_debug = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        detector.close()
        capture.release()


def _draw_calibration_panel(
    state: str, controller: SerialThrottleController
) -> np.ndarray:
    panel = np.full((440, 720, 3), 25, dtype=np.uint8)
    voltage = STATE_VOLTAGES[state]
    dac_value = voltage_to_dac(voltage)
    color = STATE_COLORS[state]
    cv2.putText(panel, "Calibration Mode", (35, 62), cv2.FONT_HERSHEY_DUPLEX, 1.25, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(panel, f"State: {state}", (35, 135), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2, cv2.LINE_AA)
    cv2.putText(panel, f"Voltage: {voltage:.2f} V", (35, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(panel, f"DAC: {dac_value}", (35, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(panel, f"Serial: {controller.status}", (35, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(panel, "D = DESCEND   H = HOVER   U = CLIMB", (35, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(panel, "Q or ESC = safe HOVER and quit", (35, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (200, 200, 200), 1, cv2.LINE_AA)
    return panel


def run_calibration(controller: SerialThrottleController) -> None:
    state = "HOVER"
    last_send = -math.inf
    last_debug = -math.inf
    while True:
        now = time.monotonic()
        controller.maybe_reconnect(now)
        if controller.enabled and not controller.connected:
            state = "HOVER"
        dac_value = voltage_to_dac(STATE_VOLTAGES[state])
        if now - last_send >= 1.0 / SERIAL_RATE_HZ:
            if not controller.send_throttle_command(dac_value, now):
                state = "HOVER"
            last_send = now

        cv2.imshow("HS210 DAC Calibration", _draw_calibration_panel(state, controller))
        if now - last_debug >= 1.0:
            print(
                "Calibration Mode\n"
                f"State: {state}\n"
                f"Voltage: {STATE_VOLTAGES[state]:.2f} V\n"
                f"DAC: {voltage_to_dac(STATE_VOLTAGES[state])}\n"
                f"Serial: {controller.status}\n"
            )
            last_debug = now

        key = cv2.waitKey(10) & 0xFF
        if key in (ord("d"), ord("D")):
            state = "DESCEND"
        elif key in (ord("h"), ord("H")):
            state = "HOVER"
        elif key in (ord("u"), ord("U")):
            state = "CLIMB"
        elif key in (ord("q"), ord("Q"), 27):
            break


def require_runtime_dependencies(
    serial_required: bool, camera_tracking_required: bool = True
) -> None:
    missing: list[str] = []
    if cv2 is None:
        missing.append("opencv-python")
    if camera_tracking_required and mp is None:
        missing.append("mediapipe")
    if serial_required and serial is None:
        missing.append("pyserial")
    if missing:
        names = ", ".join(missing)
        raise RuntimeError(
            f"Missing runtime dependencies: {names}. "
            "Install them with: pip install -r requirements.txt"
        )


def parse_args(arguments: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control HS210 transmitter throttle from safe hand gestures."
    )
    parser.add_argument("--port", help="Pico serial port, such as COM5 or /dev/ttyACM0")
    parser.add_argument(
        "--no-serial",
        action="store_true",
        help="Run camera, classification, smoothing, and visualization without a Pico",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Bench-test exact D/H/U DAC values without opening the webcam",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index (default: 0)")
    parser.add_argument(
        "--angle-threshold",
        type=float,
        default=FINGER_ANGLE_THRESHOLD_DEG,
        help="UP/DOWN cone half-angle in degrees (default: 45)",
    )
    args = parser.parse_args(arguments)
    if args.no_serial and args.port:
        parser.error("--port and --no-serial cannot be used together")
    if not 0.0 <= args.angle_threshold <= 90.0:
        parser.error("--angle-threshold must be between 0 and 90 degrees")
    return args


def main(arguments: Iterable[str] | None = None) -> int:
    args = parse_args(arguments)
    require_runtime_dependencies(
        serial_required=not args.no_serial,
        camera_tracking_required=not args.calibrate,
    )
    controller = SerialThrottleController(enabled=not args.no_serial, port=args.port)
    print(
        "Calibration DAC values: "
        f"DESCEND={DESCEND_DAC}, HOVER={HOVER_DAC}, CLIMB={CLIMB_DAC}"
    )

    try:
        if args.calibrate:
            run_calibration(controller)
        else:
            run_camera(controller, args.camera_index, args.angle_threshold)
    except KeyboardInterrupt:
        print("Keyboard interrupt: returning to HOVER.")
    finally:
        print("Sending HOVER before shutdown...")
        controller.send_hover_burst()
        controller.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
