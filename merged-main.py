# main.py
# Subway Surfers–style AR runner with:
# - Camera feed as background
# - MediaPipe hand mesh overlay
# - Calibration screen (both hands steady + face visible)
# - Hand lift controls lane switching (left/right)
# - BOTH hands lift -> SLIDE (under pipes)
# - Head tilt UP -> "JUMP" (WALL collision immunity + visual jump)  [UPDATED]
# - Head tilt DOWN -> CROUCH (hold to crouch)  [MODIFIED]
# - Voice loudness -> speed gain
#
# Run examples:
#   python main.py
#   python main.py --cam 1
#   python main.py --cam 1 --backend dshow
#   python main.py --backend msmf
#
# Install (in a venv recommended):
#   pip install opencv-python mediapipe numpy pygame sounddevice

import argparse
import cv2
import time
import numpy as np
import pygame
import mediapipe as mp
from collections import deque
import sounddevice as sd
import threading

# -----------------------------
# CONFIG
# -----------------------------
CAM_INDEX = 0  # default; can be overridden by --cam

WIN_W, WIN_H = 960, 540
FPS = 60

LANES = 3
LANE_X = [int(WIN_W * 0.35), int(WIN_W * 0.50), int(WIN_W * 0.65)]  # left, mid, right
GROUND_Y = int(WIN_H * 0.78)

PLAYER_W = 60
PLAYER_H_RUN = 90
PLAYER_H_SLIDE = 55

# Obstacles
OB_W = 70
WALL_H = 95
PIPE_H = 42
PIPE_Y = int(GROUND_Y - PLAYER_H_RUN + 20)  # overhead pipe height

SPAWN_INTERVAL = 1.25     # seconds (lower = harder)
SPEED_START = 220         # pixels/sec
SPEED_GROWTH = 4          # increase over time

# Hand control tuning
DEADZONE = 0.03
LIFT_TRIGGER = 0.06       # lift threshold to trigger lane change / slide
COOLDOWN = 0.35           # seconds between lane changes
SMOOTHING = 0.80          # smooth lift values

# Jump tuning (kept, but WALL avoidance is now gesture-based immunity)
GRAVITY = 2200.0          # px/s^2
JUMP_VEL = -820.0         # px/s
JUMP_COOLDOWN = 0.35

# Slide tuning
SLIDE_DURATION = 0.70
SLIDE_COOLDOWN = 0.40

# -----------------------------
# Head movement detection (Face Mesh) [MODIFIED]
# -----------------------------
# In image coords: y DECREASES when your nose goes UP, y INCREASES when your nose goes DOWN.
HEAD_SMOOTHING = 0.85

# Jump when nose goes UP by this amount (kept)
LOOK_UP_TRIGGER = 0.035
LOOK_UP_REARM = 0.030

# Crouch when nose goes DOWN by this amount (hold crouch)
LOOK_DOWN_TRIGGER = 0.020
LOOK_DOWN_RELEASE = 0.010

# -----------------------------
# WALL "JUMP" IMMUNITY LOGIC [NEW]
# -----------------------------
# While head is UP enough, WALL collisions are ignored (visual jump shown).
# Hysteresis prevents flicker.
WALL_IMMUNE_ON = 0.030
WALL_IMMUNE_OFF = 0.020
VISUAL_JUMP_OFFSET_PX = 40  # how high the player looks like it jumps

# Calibration (hands + face)
CALIBRATION_BUFFER_SEC = 1.2
CALIBRATION_STABLE_TIME = 1.0
CALIBRATION_MAX_STD = 0.008
CONFIRM_HOLD = 1.0

SHOW_HAND_MESH = True
SHOW_FACE_MESH = False
SHOW_DEBUG = True

# -----------------------------
# VOICE SPEED CONTROL
# -----------------------------
VOICE_ENABLED = True

AUDIO_SR = 16000
AUDIO_BLOCK = 1024
VOICE_CALIBRATE_SEC = 1.0

VOICE_MIN_GAIN = 1.00
VOICE_MAX_GAIN = 1.80

VOICE_ATTACK = 0.35
VOICE_RELEASE = 0.08

# -----------------------------
# DISTANCE / WIN CONDITION
# -----------------------------
# Speed is in pixels/sec; convert to meters with this scale.
# Tune PIXELS_PER_METER until the on-screen distance feels right.
PIXELS_PER_METER = 85.0
WIN_DISTANCE_M = 100.0

# -----------------------------
# MediaPipe
# -----------------------------
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

WRIST = 0
NOSE_TIP = 1  # face mesh landmark index commonly referenced as nose tip

# -----------------------------
# Camera helpers
# -----------------------------
_BACKEND_MAP = {
    "default": None,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
}

def _open_camera(index: int, backend_name: str, width: int, height: int, warmup_reads: int = 10):
    """
    Open a camera and verify it can actually return frames.
    Returns (cap, ok_read) where cap is released if unusable.
    """
    backend = _BACKEND_MAP.get(backend_name, None)
    if backend is None:
        cap = cv2.VideoCapture(index)
    else:
        cap = cv2.VideoCapture(index, backend)

    if not cap.isOpened():
        cap.release()
        return None, False

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    ok = False
    frame = None
    for _ in range(warmup_reads):
        ok, frame = cap.read()
        if ok and frame is not None and frame.size > 0:
            break
        time.sleep(0.03)

    if not ok:
        cap.release()
        return None, False

    return cap, True

def select_camera(preferred_index: int, backend_name: str, width: int, height: int, scan_max: int = 6):
    """
    Try preferred_index first, then scan 0..scan_max-1 for a working camera.
    """
    cap, ok = _open_camera(preferred_index, backend_name, width, height)
    if ok:
        return cap, preferred_index

    for i in range(scan_max):
        if i == preferred_index:
            continue
        cap, ok = _open_camera(i, backend_name, width, height)
        if ok:
            return cap, i

    return None, None

# -----------------------------
# Helpers
# -----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def cvframe_to_pygame_surface(frame_bgr, target_size):
    frame = cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_LINEAR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surf = pygame.image.frombuffer(frame_rgb.tobytes(), target_size, "RGB")
    return surf

def draw_text(screen, text, x, y, size=26, color=(255, 255, 255)):
    font = pygame.font.SysFont("Arial", size, bold=True)
    img = font.render(text, True, color)
    screen.blit(img, (x, y))

# -----------------------------
# Voice helper
# -----------------------------
class VoiceMeter:
    """
    Measures mic loudness (RMS) in the background and outputs a smooth 0..1 voice level.
    Auto-calibrates a noise floor for the first VOICE_CALIBRATE_SEC seconds.
    """
    def __init__(self, sr=AUDIO_SR, block=AUDIO_BLOCK):
        self.sr = sr
        self.block = block
        self.lock = threading.Lock()

        self.rms = 0.0
        self.level = 0.0  # 0..1
        self.noise_floor = None
        self.calib_samples = []
        self.calib_start = time.time()

        self.stream = sd.InputStream(
            samplerate=self.sr,
            channels=1,
            blocksize=self.block,
            callback=self._callback
        )

    def _callback(self, indata, frames, time_info, status):
        if status:
            return
        x = indata[:, 0].astype(np.float32)
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))

        with self.lock:
            self.rms = rms

            if self.noise_floor is None:
                self.calib_samples.append(rms)
                if (time.time() - self.calib_start) >= VOICE_CALIBRATE_SEC and len(self.calib_samples) > 10:
                    base = float(np.median(self.calib_samples))
                    self.noise_floor = base * 1.5
            else:
                raw = (rms - self.noise_floor) / (self.noise_floor * 4.0)
                raw = clamp(raw, 0.0, 1.0)

                a = VOICE_ATTACK if raw > self.level else VOICE_RELEASE
                self.level = (1 - a) * self.level + a * raw

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()
        self.stream.close()

    def get_level(self):
        with self.lock:
            return float(self.level), float(self.rms), self.noise_floor

# -----------------------------
# Game objects
# -----------------------------
class Obstacle:
    def __init__(self, lane_idx, y, kind):
        self.lane = lane_idx
        self.x = LANE_X[lane_idx]
        self.y = float(y)
        self.kind = kind  # "WALL" or "PIPE"

        if self.kind == "WALL":
            w, h = OB_W, WALL_H
        else:
            w, h = OB_W + 25, PIPE_H

        self.rect = pygame.Rect(0, 0, w, h)
        self.update_rect()

    def update_rect(self):
        if self.kind == "WALL":
            # Wall stands on the ground
            self.rect.midbottom = (self.x, int(self.y))
        else:
            # Pipe hangs overhead
            self.rect.midtop = (self.x, int(self.y))

    def update(self, dt, speed):
        self.y += speed * dt
        self.update_rect()

    def draw(self, screen):
        if self.kind == "WALL":
            fill = (255, 80, 80)      # red
        else:
            fill = (80, 255, 120)     # green

        pygame.draw.rect(screen, fill, self.rect, border_radius=10)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, width=3, border_radius=10)

        if self.kind == "PIPE":
            lip = self.rect.copy()
            lip.height = max(10, self.rect.height // 3)
            pygame.draw.rect(screen, (0, 0, 0), lip, width=0, border_radius=10)

class Player:
    def __init__(self):
        self.lane = 1
        self.x = LANE_X[self.lane]
        self.y = float(GROUND_Y)

        self.vy = 0.0
        self.state = "RUN"   # RUN, JUMP, SLIDE
        self.slide_t = 0.0

        self.rect = pygame.Rect(0, 0, PLAYER_W, PLAYER_H_RUN)
        self.update_rect()

    def height(self):
        return PLAYER_H_SLIDE if self.state == "SLIDE" else PLAYER_H_RUN

    def on_ground(self):
        return abs(self.y - GROUND_Y) < 0.5 and self.vy == 0.0 and self.state != "JUMP"

    def update_rect(self):
        self.x = LANE_X[self.lane]
        self.rect.size = (PLAYER_W, self.height())
        self.rect.midbottom = (int(self.x), int(self.y))

    def move_left(self):
        self.lane = max(0, self.lane - 1)
        self.update_rect()

    def move_right(self):
        self.lane = min(LANES - 1, self.lane + 1)
        self.update_rect()

    def jump(self):
        if self.state != "JUMP" and self.on_ground() and self.state != "SLIDE":
            self.state = "JUMP"
            self.vy = JUMP_VEL

    def slide(self):
        if self.state != "SLIDE" and self.on_ground() and self.state != "JUMP":
            self.state = "SLIDE"
            self.slide_t = SLIDE_DURATION
            self.update_rect()

    def update(self, dt):
        if self.state == "SLIDE":
            self.slide_t -= dt
            if self.slide_t <= 0:
                self.state = "RUN"
                self.slide_t = 0.0
                self.update_rect()

        if self.state == "JUMP":
            self.vy += GRAVITY * dt
            self.y += self.vy * dt

            if self.y >= GROUND_Y:
                self.y = float(GROUND_Y)
                self.vy = 0.0
                self.state = "RUN"

        self.update_rect()

    def draw(self, screen, wall_immune=False):
        # Visual "jump" while wall_immune is active (hitbox stays the same)
        offset_y = -VISUAL_JUMP_OFFSET_PX if wall_immune else 0

        body = self.rect.copy()
        body.y += offset_y

        pygame.draw.rect(screen, (80, 200, 255), body, border_radius=25)
        pygame.draw.rect(screen, (0, 0, 0), body, width=3, border_radius=25)

        head_r = 18
        head_center = (body.centerx, body.top + 18)
        pygame.draw.circle(screen, (255, 230, 200), head_center, head_r)
        pygame.draw.circle(screen, (0, 0, 0), head_center, head_r, 2)

# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cam", type=int, default=CAM_INDEX, help="Preferred camera index (try 1 for external webcam)")
    parser.add_argument("--backend", type=str, default="dshow", choices=["default", "dshow", "msmf"],
                        help="OpenCV capture backend (Windows: dshow often works best)")
    parser.add_argument("--scan", action="store_true", help="Scan camera indices if preferred cam fails (enabled by default)")
    args = parser.parse_args()

    pygame.init()
    pygame.display.set_caption("AR Runner - Hands (lane/slide) + Head (jump/crouch) + Voice speed")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()

    cap, used_index = select_camera(args.cam, args.backend, WIN_W, WIN_H, scan_max=6)
    if cap is None:
        raise RuntimeError(
            f"Could not open any working camera (tried preferred={args.cam} plus scan 0..5) "
            f"with backend={args.backend}. Try: --cam 1 --backend dshow"
        )

    print(f"[camera] using index={used_index} backend={args.backend}")

    # -----------------------------
    # Sound FX (from webcam-main.py)
    # -----------------------------
    death_sfx = None
    try:
        pygame.mixer.init()  # initialize audio
        death_sfx = pygame.mixer.Sound("assets/sfx/death.wav")
        death_sfx.set_volume(0.8)  # 0.0 to 1.0
    except Exception as e:
        print("Sound FX disabled (mixer or file error):", e)
        death_sfx = None


    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65
    )

    face = mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    voice = None
    if VOICE_ENABLED:
        try:
            voice = VoiceMeter()
            voice.start()
        except Exception as e:
            print("Voice input disabled (mic error):", e)
            voice = None

    # Calibration state
    state = "CALIBRATING"  # CALIBRATING -> CONFIRM -> PLAYING -> GAMEOVER
    confirm_start = None

    base_left_y = None
    base_right_y = None
    base_nose_y = None

    left_hist = deque()
    right_hist = deque()
    nose_hist = deque()

    smooth_left_lift = 0.0
    smooth_right_lift = 0.0

    smooth_nose_y = None
    jump_armed = True  # re-arm for head-up jump

    last_switch_time = 0.0
    left_armed = True
    right_armed = True

    last_jump_time = 0.0
    last_slide_time = 0.0

    # Track whether crouch is being forced by head-down
    forced_crouch = False

    # WALL immunity state (head up)
    wall_immune = False

    # Game state
    player = Player()
    obstacles = []
    last_spawn = time.time()
    start_time = time.time()
    score = 0.0
    speed = SPEED_START
    distance_m = 0.0
    # If the camera stops producing frames, do NOT freeze the window.
    cam_fail_count = 0
    CAM_FAIL_REOPEN_AFTER = 30  # frames
    placeholder_frame = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        now = time.time()

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE] or keys[pygame.K_q]:
            running = False

        if keys[pygame.K_r]:
            # Recalibrate and restart run
            state = "CALIBRATING"
            confirm_start = None
            base_left_y = base_right_y = base_nose_y = None
            left_hist.clear()
            right_hist.clear()
            nose_hist.clear()

            smooth_left_lift = smooth_right_lift = 0.0
            smooth_nose_y = None
            jump_armed = True

            last_switch_time = 0.0
            left_armed = right_armed = True
            last_jump_time = 0.0
            last_slide_time = 0.0

            forced_crouch = False
            wall_immune = False

            player = Player()
            obstacles = []
            last_spawn = time.time()
            start_time = time.time()
            score = 0.0
            speed = SPEED_START
            distance_m = 0.0

        # Grab camera frame (keep responsive on failure)
        ok, frame = cap.read()
        if not ok or frame is None or frame.size == 0:
            cam_fail_count += 1
            frame = placeholder_frame.copy()

            if cam_fail_count >= CAM_FAIL_REOPEN_AFTER:
                cap.release()
                cap, used_index = select_camera(args.cam, args.backend, WIN_W, WIN_H, scan_max=6)
                cam_fail_count = 0
                if cap is None:
                    cap = cv2.VideoCapture()  # dummy closed capture
        else:
            cam_fail_count = 0
            frame = cv2.flip(frame, 1)

        do_detection = (cam_fail_count == 0) and (frame is not placeholder_frame) and (frame is not None)

        left_y = None
        right_y = None
        nose_y = None

        if do_detection:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Hands
            res_h = hands.process(rgb)
            if res_h.multi_hand_landmarks and res_h.multi_handedness:
                for hand_lms, handedness in zip(res_h.multi_hand_landmarks, res_h.multi_handedness):
                    label = handedness.classification[0].label  # "Left"/"Right"
                    wrist_y = hand_lms.landmark[WRIST].y

                    if SHOW_HAND_MESH:
                        mp_draw.draw_landmarks(
                            frame,
                            hand_lms,
                            mp_hands.HAND_CONNECTIONS,
                            mp_styles.get_default_hand_landmarks_style(),
                            mp_styles.get_default_hand_connections_style()
                        )

                    if label == "Left":
                        left_y = wrist_y
                    elif label == "Right":
                        right_y = wrist_y

            # Face (nose)
            res_f = face.process(rgb)
            if res_f.multi_face_landmarks:
                lms = res_f.multi_face_landmarks[0].landmark
                if NOSE_TIP < len(lms):
                    nose_y = lms[NOSE_TIP].y

                if SHOW_FACE_MESH:
                    mp_draw.draw_landmarks(
                        frame,
                        res_f.multi_face_landmarks[0],
                        mp_face.FACEMESH_TESSELATION,
                        None,
                        None
                    )

            # Smooth nose value
            if nose_y is not None:
                if smooth_nose_y is None:
                    smooth_nose_y = nose_y
                else:
                    smooth_nose_y = HEAD_SMOOTHING * smooth_nose_y + (1 - HEAD_SMOOTHING) * nose_y

        # -------------- CALIBRATION --------------
        if state == "CALIBRATING":
            wall_immune = False  # ensure off during calibration

            if left_y is not None and right_y is not None and smooth_nose_y is not None:
                left_hist.append((now, left_y))
                right_hist.append((now, right_y))
                nose_hist.append((now, smooth_nose_y))

            while left_hist and (now - left_hist[0][0] > CALIBRATION_BUFFER_SEC):
                left_hist.popleft()
            while right_hist and (now - right_hist[0][0] > CALIBRATION_BUFFER_SEC):
                right_hist.popleft()
            while nose_hist and (now - nose_hist[0][0] > CALIBRATION_BUFFER_SEC):
                nose_hist.popleft()

            if len(left_hist) >= 2 and len(right_hist) >= 2 and len(nose_hist) >= 2:
                span = min(
                    left_hist[-1][0] - left_hist[0][0],
                    right_hist[-1][0] - right_hist[0][0],
                    nose_hist[-1][0] - nose_hist[0][0],
                )
                if span >= CALIBRATION_STABLE_TIME:
                    lvals = np.array([v for _, v in left_hist], dtype=np.float32)
                    rvals = np.array([v for _, v in right_hist], dtype=np.float32)
                    nvals = np.array([v for _, v in nose_hist], dtype=np.float32)

                    if (float(lvals.std()) <= CALIBRATION_MAX_STD and
                        float(rvals.std()) <= CALIBRATION_MAX_STD and
                        float(nvals.std()) <= CALIBRATION_MAX_STD):
                        base_left_y = float(lvals.mean())
                        base_right_y = float(rvals.mean())
                        base_nose_y = float(nvals.mean())
                        state = "CONFIRM"
                        confirm_start = now

        elif state == "CONFIRM":
            wall_immune = False  # keep off until playing
            if (now - confirm_start) >= CONFIRM_HOLD:
                state = "PLAYING"
                start_time = time.time()
                last_spawn = time.time()
                score = 0.0
                speed = SPEED_START
                wall_immune = False

        # -------------- PLAYING --------------
        if state == "PLAYING":
            # Keyboard fallback
            if keys[pygame.K_LEFT]:
                player.move_left()
            if keys[pygame.K_RIGHT]:
                player.move_right()
            if keys[pygame.K_SPACE] and (now - last_jump_time) >= JUMP_COOLDOWN:
                player.jump()
                last_jump_time = now
            if keys[pygame.K_DOWN] and (now - last_slide_time) >= SLIDE_COOLDOWN:
                player.slide()
                last_slide_time = now

            # Difficulty + voice speed
            score = now - start_time
            base_speed = SPEED_START + SPEED_GROWTH * score

            voice_gain = 1.0
            voice_level = 0.0
            if voice is not None:
                voice_level, rms, nf = voice.get_level()
                voice_gain = VOICE_MIN_GAIN + (VOICE_MAX_GAIN - VOICE_MIN_GAIN) * voice_level

            speed = base_speed * voice_gain
            distance_m += (speed * dt) / PIXELS_PER_METER
            # Update player physics
            player.update(dt)

            # Compute hand lifts (from baseline)
            left_lift = 0.0
            right_lift = 0.0
            if left_y is not None and base_left_y is not None:
                left_lift = max(0.0, base_left_y - left_y)
            if right_y is not None and base_right_y is not None:
                right_lift = max(0.0, base_right_y - right_y)

            if left_lift < DEADZONE:
                left_lift = 0.0
            if right_lift < DEADZONE:
                right_lift = 0.0

            smooth_left_lift = SMOOTHING * smooth_left_lift + (1 - SMOOTHING) * left_lift
            smooth_right_lift = SMOOTHING * smooth_right_lift + (1 - SMOOTHING) * right_lift

            both_hands_up = (smooth_left_lift >= LIFT_TRIGGER) and (smooth_right_lift >= LIFT_TRIGGER)

            # Re-arm edges when hands go down
            if smooth_left_lift <= (LIFT_TRIGGER * 0.35):
                left_armed = True
            if smooth_right_lift <= (LIFT_TRIGGER * 0.35):
                right_armed = True

            # Slide on BOTH hands up
            if both_hands_up and (now - last_slide_time) >= SLIDE_COOLDOWN:
                player.slide()
                last_slide_time = now

            # Lane switching on single-hand up (with cooldown)
            can_switch = (now - last_switch_time) >= COOLDOWN
            if can_switch and (not both_hands_up):
                if smooth_left_lift >= LIFT_TRIGGER and left_armed:
                    player.move_left()
                    left_armed = False
                    last_switch_time = now
                elif smooth_right_lift >= LIFT_TRIGGER and right_armed:
                    player.move_right()
                    right_armed = False
                    last_switch_time = now

            # -----------------------------
            # Head control:
            # - DOWN -> hold crouch (forces SLIDE)
            # - UP -> WALL immunity ON (visual jump) [NEW]
            # -----------------------------
            if base_nose_y is not None and smooth_nose_y is not None:
                nose_up = (base_nose_y - smooth_nose_y)      # positive when nose goes UP
                nose_down = (smooth_nose_y - base_nose_y)    # positive when nose goes DOWN

                # WALL immunity toggle based on head UP (hysteresis)
                if nose_up >= WALL_IMMUNE_ON:
                    wall_immune = True
                elif nose_up <= WALL_IMMUNE_OFF:
                    wall_immune = False

                # Jump logic kept (optional / you can still use it), but not required for wall avoidance now
                if nose_up <= LOOK_UP_REARM:
                    jump_armed = True

                if jump_armed and nose_up >= LOOK_UP_TRIGGER and (now - last_jump_time) >= JUMP_COOLDOWN:
                    player.jump()
                    last_jump_time = now
                    jump_armed = False
                    forced_crouch = False

                # Hold crouch while head DOWN (only when not jumping)
                if player.state != "JUMP":
                    if nose_down >= LOOK_DOWN_TRIGGER:
                        forced_crouch = True

                    if forced_crouch:
                        player.state = "SLIDE"
                        player.slide_t = SLIDE_DURATION
                        player.update_rect()

                        if nose_down <= LOOK_DOWN_RELEASE:
                            forced_crouch = False
                            player.state = "RUN"
                            player.slide_t = 0.0
                            player.update_rect()

            # Spawn obstacles
            if (now - last_spawn) >= SPAWN_INTERVAL:
                lane = int(np.random.randint(0, LANES))
                kind = "WALL" if np.random.rand() < 0.6 else "PIPE"
                if kind == "WALL":
                    spawn_y = -20  # uses midbottom
                else:
                    spawn_y = -PIPE_H - 20  # uses midtop

                obstacles.append(Obstacle(lane, y=spawn_y, kind=kind))
                last_spawn = now

            # Update obstacles
            for ob in obstacles:
                ob.update(dt, speed)

            obstacles = [ob for ob in obstacles if ob.y < WIN_H + 220]

            # Collision rules:
            # - WALL: ignored while wall_immune is True (head up)
            # - PIPE: if sliding, ignore collision; otherwise collide
            for ob in obstacles:
                if player.rect.colliderect(ob.rect):
                    if ob.kind == "PIPE" and player.state == "SLIDE":
                        continue
                    if ob.kind == "WALL" and wall_immune:
                        continue
                    if death_sfx is not None:
                        try:
                            death_sfx.play()
                        except Exception:
                            pass
                    state = "GAMEOVER"
                    break
            if state in ("PLAYING", "GAMEOVER") and distance_m >= WIN_DISTANCE_M:
                distance_m = WIN_DISTANCE_M
                state = "WIN"
        # -------------- RENDER --------------
        bg = cvframe_to_pygame_surface(frame, (WIN_W, WIN_H))
        screen.blit(bg, (0, 0))

        for x in LANE_X:
            pygame.draw.line(screen, (255, 255, 255), (x, 0), (x, WIN_H), 2)
        pygame.draw.line(screen, (255, 255, 255), (0, GROUND_Y), (WIN_W, GROUND_Y), 2)

        player.draw(screen, wall_immune if state == "PLAYING" else False)
        for ob in obstacles:
            ob.draw(screen)

        if cam_fail_count > 0:
            draw_text(screen, "CAMERA NOT READY (check webcam index / permissions)", 20, 18, 26, (255, 150, 0))
            draw_text(screen, "Try: python main.py --cam 1 --backend dshow", 20, 50, 22, (255, 150, 0))

        if state == "CALIBRATING":
            draw_text(screen, "CALIBRATION", 20, 90, 34, (255, 255, 0))
            draw_text(screen, "Show BOTH hands LOW & steady + keep face visible.", 20, 132, 26)
            draw_text(screen, "This sets hand rest position + head baseline.", 20, 164, 24)
            draw_text(screen, "Press R to recalibrate. Esc/Q to quit.", 20, 198, 22)

            if SHOW_DEBUG:
                l_ok = "OK" if left_y is not None else "---"
                r_ok = "OK" if right_y is not None else "---"
                f_ok = "OK" if smooth_nose_y is not None else "---"
                draw_text(screen, f"Hands: L={l_ok}  R={r_ok}   Face={f_ok}", 20, 235, 22, (200, 255, 200))

        elif state == "CONFIRM":
            draw_text(screen, "Calibration complete!", 20, 18, 36, (0, 255, 0))
            draw_text(screen, "Left hand: left lane | Right hand: right lane", 20, 62, 26)
            draw_text(screen, "Both hands: slide | Head UP: jump (walls) | Head DOWN: crouch", 20, 92, 26)

        elif state == "PLAYING":
            draw_text(screen, f"Time: {score:0.2f}s", 20, 18, 30)
            draw_text(screen, f"Speed: {int(speed)}", 20, 52, 24)
            draw_text(screen, f"State: {player.state}", 20, 80, 24)
            draw_text(screen, f"WallImmune: {wall_immune}", 20, 104, 22, (255, 255, 0))
            draw_text(screen, "Walls=HEAD UP  Pipes=SLIDE", 20, WIN_H - 40, 22, (255, 255, 0))
            draw_text(screen, f"Distance: {distance_m:0.1f}m / {WIN_DISTANCE_M:0.0f}m", 20, 44, 24)
            if voice is not None:
                draw_text(screen, f"Voice: {voice_level:.2f}  gain: {voice_gain:.2f}x", 20, 130, 20, (200, 255, 200))

            if SHOW_DEBUG and base_left_y is not None and base_right_y is not None and base_nose_y is not None:
                draw_text(screen, f"liftL={smooth_left_lift:.3f} liftR={smooth_right_lift:.3f}", 20, 155, 20, (200, 255, 200))
                if smooth_nose_y is not None:
                    draw_text(screen, f"noseUp={(base_nose_y-smooth_nose_y):.3f} noseDown={(smooth_nose_y-base_nose_y):.3f}",
                              20, 178, 20, (200, 255, 200))
        elif state == "WIN":
            draw_text(screen, "YOU WIN!", WIN_W // 2 - 115, WIN_H // 2 - 70, 54, (0, 255, 120))
            draw_text(screen, f"Distance: {distance_m:0.1f}m", WIN_W // 2 - 140, WIN_H // 2 - 5, 32, (255, 255, 255))
            draw_text(screen, f"Time: {score:0.2f}s", WIN_W // 2 - 120, WIN_H // 2 + 35, 28, (255, 255, 255))
            draw_text(screen, "Press R to play again (recalibrates).", WIN_W // 2 - 240, WIN_H // 2 + 80, 24,
                      (255, 255, 0))

        elif state == "GAMEOVER":
            panel = pygame.Rect(WIN_W // 2 - 260, WIN_H // 2 - 90, 520, 180)
            pygame.draw.rect(screen, (0, 0, 0), panel)              # filled background
            pygame.draw.rect(screen, (255, 255, 255), panel, 3)

            draw_text(screen, "lock in twin", WIN_W // 2 - 130, WIN_H // 2 - 60, 48, (255, 80, 80))
            draw_text(screen, f"Survived: {score:0.2f}s", WIN_W // 2 - 145, WIN_H // 2, 30, (255, 255, 255))
            draw_text(screen, "Press R to restart (recalibrates).", WIN_W // 2 - 220, WIN_H // 2 + 45, 24, (255, 255, 0))

        pygame.display.flip()

    hands.close()
    face.close()
    try:
        cap.release()
    except Exception:
        pass

    if voice is not None:
        voice.stop()

    pygame.quit()

if __name__ == "__main__":
    main()
