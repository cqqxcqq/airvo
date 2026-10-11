"""Official versioned model location and local integrity validation."""

import hashlib
from pathlib import Path

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
# Computed from the official HTTPS object after checking its MD5 and size against
# Google's object metadata API, generation 1682480004222387 (2023-04-26).
MODEL_SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "hand_landmarker.task"


def verify_model(path: Path) -> str:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError(f"Model or integrity sidecar missing at {path}; run python tools/fetch_model.py")
    fields = sidecar.read_text(encoding="ascii").split()
    if not fields:
        raise ValueError("Empty model integrity sidecar. Re-download with tools/fetch_model.py --force")
    expected = fields[0]
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected != MODEL_SHA256 or actual != MODEL_SHA256:
        raise ValueError("Hand model checksum mismatch. Re-download with tools/fetch_model.py --force")
    return actual
