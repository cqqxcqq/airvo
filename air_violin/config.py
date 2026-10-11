"""Validated, locally persisted normalized calibration; no camera images saved."""

from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        values = (self.x0, self.y0, self.x1, self.y1)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError("Region boundaries must be finite numbers")
        if not (0 <= self.x0 < self.x1 <= 1 and 0 <= self.y0 < self.y1 <= 1):
            raise ValueError("Region boundaries must be ordered within [0, 1]")
        if self.x1 - self.x0 < 0.15 or self.y1 - self.y0 < 0.12:
            raise ValueError("Region is too small: use at least 15% width and 12% height")

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1

    def relative(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.x0) / (self.x1 - self.x0), (y - self.y0) / (self.y1 - self.y0)


@dataclass(frozen=True)
class FingerCalibration:
    """User-specific comfortable lifted/pressed geometry, one value per finger."""

    lifted: tuple[float, ...]
    pressed: tuple[float, ...]
    engage: float = 0.65
    release: float = 0.35
    dwell: float = 0.045
    unknown_timeout: float = 0.080

    def __post_init__(self) -> None:
        # JSON arrays become immutable tuples when loaded.
        object.__setattr__(self, "lifted", tuple(self.lifted))
        object.__setattr__(self, "pressed", tuple(self.pressed))
        if len(self.lifted) != 4 or len(self.pressed) != 4:
            raise ValueError("Calibrate all four fingers")
        if not all(math.isfinite(v) for v in (*self.lifted, *self.pressed)):
            raise ValueError("Finger poses must be finite")
        if any(abs(b - a) < 0.06 for a, b in zip(self.lifted, self.pressed)):
            raise ValueError("Lifted and pressed finger poses are too similar")
        if not 0 <= self.release < self.engage <= 1:
            raise ValueError("Invalid finger hysteresis")
        if not 0 <= self.dwell <= 0.1 or not 0 < self.unknown_timeout <= 0.080:
            raise ValueError("Invalid finger timing")


@dataclass(frozen=True)
class Calibration:
    version: int = 1
    mirrored: bool = True
    left_label: str = "Left"
    right_label: str = "Right"
    board: Rect = field(default_factory=lambda: Rect(0.08, 0.22, 0.48, 0.78))
    bow: Rect = field(default_factory=lambda: Rect(0.53, 0.22, 0.94, 0.82))
    bow_full_speed: float = 0.8  # normalized screen widths per second.
    pinch_on: float = 0.35
    pinch_off: float = 0.55
    fingers: FingerCalibration | None = None

    def __post_init__(self) -> None:
        if self.version not in (1, 2) or not isinstance(self.mirrored, bool):
            raise ValueError("Unsupported calibration version or mirror setting")
        if {self.left_label, self.right_label} != {"Left", "Right"}:
            raise ValueError("Calibrate distinct physical left and right hands")
        values = (self.bow_full_speed, self.pinch_on, self.pinch_off)
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
            raise ValueError("Calibration thresholds must be finite numbers")
        if not 0.15 <= self.bow_full_speed <= 4.0:
            raise ValueError("Bow speed must be 0.15..4 screen widths/second")
        if not 0 < self.pinch_on < self.pinch_off <= 6:
            raise ValueError("Pinch engagement must be smaller than disengagement")
        if self.fingers is not None and not isinstance(self.fingers, FingerCalibration):
            raise ValueError("Invalid finger calibration")


@dataclass(frozen=True)
class Controls:
    confidence: float = 0.65
    position_tau: float = 0.035
    velocity_tau: float = 0.040
    note_hold: float = 0.050
    hysteresis: float = 0.12  # fraction of one note zone.
    deadzone: float = 0.045
    reversal_grace: float = 0.085
    tracking_timeout: float = 0.180
    release_time: float = 0.250

    def __post_init__(self) -> None:
        if not 0.3 <= self.confidence <= 0.99:
            raise ValueError("Confidence must be 0.3..0.99")
        if not 0.005 <= self.position_tau <= 0.25 or not 0.005 <= self.velocity_tau <= 0.25:
            raise ValueError("Smoothing must be 5..250 ms")
        if not 0 <= self.note_hold <= 0.2 or not 0 <= self.hysteresis <= 0.4:
            raise ValueError("Invalid note dwell or hysteresis")
        if not 0.005 <= self.deadzone <= 0.3:
            raise ValueError("Deadzone must be 0.005..0.3 screen widths/second")
        if not 0 <= self.reversal_grace < self.tracking_timeout <= 0.3:
            raise ValueError("Reversal grace must be shorter than tracking timeout (max 300 ms)")


def default_config_path() -> Path:
    base = Path(os.environ["LOCALAPPDATA"]) if os.name == "nt" and "LOCALAPPDATA" in os.environ else Path.home()
    return base / ".air-violin-studio" / "calibration.json"


def load_calibration(path: Path, mirrored: bool) -> Calibration | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    data["board"] = Rect(**data["board"])
    data["bow"] = Rect(**data["bow"])
    if data.get("fingers") is not None:
        data["fingers"] = FingerCalibration(**data["fingers"])
    calibration = Calibration(**data)
    # Mirroring changes model inputs as well as x coordinates. Recalibrate labels.
    if calibration.mirrored != mirrored:
        return None
    return calibration


def save_calibration(path: Path, calibration: Calibration) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(asdict(calibration), f, indent=2, allow_nan=False)
            f.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
