import cv2
import numpy as np
import math
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------
# Face Landmarker (Tasks API ONLY)
# -------------------------
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1
)

face_landmarker = vision.FaceLandmarker.create_from_options(options)

# -------------------------
# Manual mesh connections (triangles from MediaPipe Face Mesh 468 landmarks)
# -------------------------
# Simple tesselation edges (example subset for demo)
# You can expand this list for full mesh
TESSELATION = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
    (12, 13), (13, 14), (14, 15), (15, 16), (16, 17),
    # ... add more connections for full mesh
]

LIPS = [
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
    (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
    (291, 61)
]

LEFT_EYE = [
    (33, 246), (246, 161), (161, 160), (160, 159),
    (159, 158), (158, 157), (157, 173), (173, 133)
]

RIGHT_EYE = [
    (362, 398), (398, 384), (384, 385), (385, 386),
    (386, 387), (387, 388), (388, 466), (466, 263)
]

# -------------------------
# Helpers
# -------------------------
def neon_color(t):
    r = int(128 + 127 * math.sin(t))
    g = int(128 + 127 * math.sin(t + 2))
    b = int(128 + 127 * math.sin(t + 4))
    return (b, g, r)

def draw_connections(frame, landmarks, connections, color, thickness):
    h, w, _ = frame.shape
    for a, b in connections:
        pa = landmarks[a]
        pb = landmarks[b]
        x1, y1 = int(pa.x * w), int(pa.y * h)
        x2, y2 = int(pb.x * w), int(pb.y * h)
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

# -------------------------
# Webcam
# -------------------------
cap = cv2.VideoCapture(0)
frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    result = face_landmarker.detect_for_video(mp_image, frame_id)
    glow = np.zeros_like(frame)
    color = neon_color(frame_id * 0.05)

    if result.face_landmarks:
        face = result.face_landmarks[0]

        # ---- Full mesh glow
        draw_connections(glow, face, TESSELATION, color, 2)

        # ---- Mouth emphasis
        draw_connections(frame, face, LIPS, (0, 255, 255), 2)

        # ---- Eye emphasis
        draw_connections(frame, face, LEFT_EYE, (255, 255, 0), 2)
        draw_connections(frame, face, RIGHT_EYE, (255, 255, 0), 2)

        # ---- Main mesh
        draw_connections(frame, face, TESSELATION, color, 1)

    # Glow blend
    frame = cv2.addWeighted(frame, 1.0, glow, 0.6, 0)

    cv2.imshow("🔥 Neon Face Mesh (Tasks API)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
