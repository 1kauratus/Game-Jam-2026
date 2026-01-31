# Game-Jam-2026
Game jam bla bla bla

## How to run on vscode (for webcam-main.py)
python -m pip uninstall -y mediapipe
python -m pip install --upgrade pip
python -m pip install --force-reinstall mediapipe==0.10.21
python -m pip install opencv-python numpy pygame sounddevice

python webcam-main.py --cam 0 --backend dshow

- Change the number from 0 to 1 depends on which camera on your device you wanna use
    - 0 is for external webcam in my case