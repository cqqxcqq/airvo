#!/usr/bin/env bash
# Codex cloud: CPython 3.12, Debian 13 x86_64. Laptop setup is in README.md.
set -euo pipefail
cd /workspace/airvo
mkdir -p .local/matplotlib .local/cache
export MPLCONFIGDIR="$PWD/.local/matplotlib"
export XDG_CACHE_HOME="$PWD/.local/cache"
export PIP_CACHE_DIR="$PWD/.local/cache/pip"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use CPython 3.12"'
python3 -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check --only-binary=:all: -r requirements.txt

# Retained local libraries do not need reinstallation. No system writes or sudo.
AIRVO_AUDIO_LIB="$PWD/.local/portaudio/usr/lib/x86_64-linux-gnu"
export LD_LIBRARY_PATH="$AIRVO_AUDIO_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
if ! .venv/bin/python -c 'import sounddevice' >/dev/null 2>&1; then
    test "$(uname -m)" = x86_64
    test -f /usr/share/keyrings/debian-archive-keyring.gpg
    mkdir -p .local/apt/lists/partial .local/apt/cache/archives/partial .local/packages .local/portaudio
    cat > .local/apt/sources.list <<'SOURCES'
deb [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] https://deb.debian.org/debian trixie main
SOURCES
    cat > .local/apt/apt.conf <<'CONFIG'
Dir::Etc::main "-";
Dir::Etc::parts "-";
Dir::Etc::sourcelist "/workspace/airvo/.local/apt/sources.list";
Dir::Etc::sourceparts "-";
Dir::State::lists "/workspace/airvo/.local/apt/lists";
Dir::Cache "/workspace/airvo/.local/apt/cache";
APT::Get::List-Cleanup "0";
Acquire::Languages "none";
CONFIG
    APT_CONFIG="$PWD/.local/apt/apt.conf" /usr/bin/apt-get update
    (
        cd .local/packages
        APT_CONFIG=/workspace/airvo/.local/apt/apt.conf /usr/bin/apt-get download libportaudio2=19.6.0-1.2+b3
        dpkg-deb -x libportaudio2_19.6.0-1.2+b3_amd64.deb /workspace/airvo/.local/portaudio
    )
    # ctypes.find_library needs the linker name when using a local LD path.
    ln -sfn libportaudio.so.2 "$AIRVO_AUDIO_LIB/libportaudio.so"
fi
.venv/bin/python -m pip check
.venv/bin/python -c 'import mediapipe, cv2, numpy, sounddevice; print("Desktop dependencies import successfully")'
.venv/bin/python -m unittest discover -s tests -v
if [[ "${1:-}" != --skip-model ]]; then
    .venv/bin/python tools/fetch_model.py --all
    .venv/bin/python tools/validate.py --require-model
    .venv/bin/python tools/validate_studio.py --require-segmenter
else
    # Explicit diagnostic option for testing shared prerequisites while network
    # configuration is pending. It does not establish webcam-mode readiness.
    .venv/bin/python tools/validate.py
fi
