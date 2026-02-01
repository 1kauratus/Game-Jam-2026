# Game-Jam-2026
Game jam bla bla bla

Just run these pip commands in your folder:

python -m pip install --upgrade pip setuptools wheel
pip install "numpy<2" opencv-python mediapipe pygame sounddevice

---
## exact thing to run if you are using window:
```
$ErrorActionPreference = "Stop"

$VENV = ".venv312"

# Create venv with Python 3.12 (requires Python 3.12 to be installed)
if (!(Test-Path $VENV)) {
  py -3.12 -m venv $VENV
}

$PY = Join-Path $VENV "Scripts\python.exe"

# Verify Python version is 3.12.x
$ver = & $PY -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
if (-not $ver.StartsWith("3.12.")) { throw "Expected Python 3.12.x, got $ver" }

# Install pinned deps
& $PY -m pip install --upgrade pip
& $PY -m pip install "mediapipe==0.10.21" "pygame==2.6.1" opencv-python numpy sounddevice

# Run with arguments (DirectShow backend, camera 0)
& $PY ".\merged-main-v3-mask.py" --backend dshow --cam 0

```
