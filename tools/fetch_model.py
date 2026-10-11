"""Install Google's official hand model and optional person model over HTTPS.

The official version-1 bundle is checked against a pinned SHA-256 digest before
installation. A sidecar also detects later corruption. No TLS bypasses.
"""

import argparse
import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from air_violin.models import (DEFAULT_MODEL, MODEL_SHA256, MODEL_URL, verify_model,
                               DEFAULT_SEGMENTER, SEGMENTER_SHA256, SEGMENTER_URL, verify_segmenter)


def fetch_model(path: Path, force: bool = False) -> str:
    if path.exists() and not force:
        return verify_model(path)
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "AirViolinStudio/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = response.read(20_000_001)
    digest = hashlib.sha256(data).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError("Pinned model SHA-256 mismatch; download was not installed")
    if not 1_000_000 < len(data) <= 20_000_000:
        raise ValueError("Unexpected model size; download was not installed")
    with zipfile.ZipFile(io.BytesIO(data)) as bundle:
        if bundle.testzip() is not None or len(bundle.namelist()) < 2:
            raise ValueError("Invalid Hand Landmarker task bundle")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(data)
        os.replace(temporary, path)
        path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return verify_model(path)


def fetch_segmenter(path: Path = DEFAULT_SEGMENTER, force: bool = False) -> str:
    if path.exists() and not force:
        return verify_segmenter(path)
    request = urllib.request.Request(SEGMENTER_URL, headers={"User-Agent": "AirViolinStudio/0.2"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(2_000_001)
    digest = hashlib.sha256(data).hexdigest()
    if digest != SEGMENTER_SHA256 or data[4:8] != b"TFL3":
        raise ValueError("Person model checksum / TFLite header mismatch; download not installed")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(data)
        os.replace(temporary, path)
        path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return verify_segmenter(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--force", action="store_true", help="Replace an existing or corrupt model")
    parser.add_argument("--all", action="store_true", help="Also install the optional person-segmentation model")
    args = parser.parse_args()
    try:
        print(f"Model verified: {args.output}\nSHA-256: {fetch_model(args.output, args.force)}")
        if args.all:
            print(f"Person model verified: {DEFAULT_SEGMENTER}\nSHA-256: {fetch_segmenter(force=args.force)}")
        return 0
    except (OSError, ValueError, urllib.error.URLError, zipfile.BadZipFile, IndexError) as exc:
        print(f"Model installation failed: {exc}\nAllow storage.googleapis.com on HTTPS; do not disable certificate verification.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
