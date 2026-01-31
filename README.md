# Game-Jam-2026
Game jam bla bla bla

Just run these pip commands in your folder:

python -m pip install --upgrade pip setuptools wheel
pip install "numpy<2" opencv-python mediapipe pygame sounddevice

---
## exact thing to run if you are using window:
# ---- Game Jam 2026: quick run (Windows PowerShell) ----

# Prefer Python 3.11 (best compatibility with mediapipe==0.10.21)
$py = $null
try { $py = (py -3.11 -c "import sys; print(sys.executable)" 2>$null).Trim() } catch {}
if (-not $py) { $py = (Get-Command python -ErrorAction SilentlyContinue).Source }

if (-not $py) {
  Write-Host "Python not found. Install Python 3.11 from python.org, then re-run this script."
  exit 1
}

# Create venv (idempotent)
if (-not (Test-Path ".\.venv")) { & $py -m venv .venv }

# Activate venv
. .\.venv\Scripts\Activate.ps1

# Upgrade pip tooling
python -m pip install --upgrade pip setuptools wheel

# Install deps (pin mediapipe to required version)
pip uninstall -y mediapipe *> $null
pip install --no-cache-dir opencv-python numpy pygame sounddevice mediapipe==0.10.21

# Quick sanity check
python -c "import mediapipe as mp; assert mp.__version__=='0.10.21' and hasattr(mp,'solutions'); print('mediapipe OK:', mp.__version__)"

# Run game (DirectShow backend for Windows)
python .\webcam-main.py --backend dshow --cam 0

