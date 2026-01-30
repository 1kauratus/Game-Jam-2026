# ar_runner.py
# Subway Surfers–style AR runner with:
# - Camera feed as background (fixed: no half-black)
# - MediaPipe hand mesh overlay
# - Calibration screen (must see both hands steady)
# - Hand lift controls lane switching (left/right)
#
# Install:
#   pip install opencv-python mediapipe numpy pygame
# Run:
#   python ar_runner.py

import cv2
import time
import numpy as np
import pygame
import mediapipe as mp
from collections import deque

# -----------------------------
# CONFIG
# -----------------------------
CAM_INDEX = 0

WIN_W, WIN_H = 960, 540
FPS = 60

LANES = 3
LANE_X = [int(WIN_W * 0.35), int(WIN_W * 0.50), int(WIN_W * 0.65)]  # left, mid, right
PLAYER_Y = int(WIN_H * 0.78)

PLAYER_W, PLAYER_H = 60, 90

OB_W, OB_H = 70, 80
SPAWN_INTERVAL = 1.25     # seconds (lower = harder)
SPEED_START = 260         # pixels/sec
SPEED_GROWTH = 6         # increase over time

# Hand control tuning
DEADZONE = 0.03
LIFT_TRIGGER = 0.06       # lift threshold to trigger lane change
COOLDOWN = 0.35           # seconds between lane changes
SMOOTHING = 0.80          # smooth lift values

# Calibration
CALIBRATION_BUFFER_SEC = 1.2
CALIBRATION_STABLE_TIME = 1.0
CALIBRATION_MAX_STD = 0.008
CONFIRM_HOLD = 1.0

SHOW_HAND_MESH = True
SHOW_DEBUG = True

# MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
WRIST = 0

# -----------------------------
# Helpers
# -----------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def cvframe_to_pygame_surface(frame_bgr, target_size):
    """
    Reliable OpenCV(BGR) -> Pygame Surface conversion.
    Fixes the half-black / wrong-stride issue by using frombuffer().
    """
    frame = cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_LINEAR)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surf = pygame.image.frombuffer(frame_rgb.tobytes(), target_size, "RGB")
    return surf

def draw_text(screen, text, x, y, size=26, color=(255, 255, 255)):
    font = pygame.font.SysFont("Arial", size, bold=True)
    img = font.render(text, True, color)
    screen.blit(img, (x, y))

# -----------------------------
# Game objects
# -----------------------------
class Obstacle:
    def __init__(self, lane_idx, y):
        self.lane = lane_idx
        self.x = LANE_X[lane_idx]
        self.y = y
        self.rect = pygame.Rect(0, 0, OB_W, OB_H)
        self.update_rect()

    def update_rect(self):
        self.rect.center = (self.x, int(self.y))

    def update(self, dt, speed):
        self.y += speed * dt
        self.update_rect()

    def draw(self, screen):
        pygame.draw.rect(screen, (255, 80, 80), self.rect, border_radius=10)
        pygame.draw.rect(screen, (0, 0, 0), self.rect, width=3, border_radius=10)

class Player:
    def __init__(self):
        self.lane = 1
        self.x = LANE_X[self.lane]
        self.y = PLAYER_Y
        self.rect = pygame.Rect(0, 0, PLAYER_W, PLAYER_H)
        self.update_rect()

    def update_rect(self):
        self.x = LANE_X[self.lane]
        self.rect.center = (self.x, self.y)

    def move_left(self):
        self.lane = max(0, self.lane - 1)
        self.update_rect()

    def move_right(self):
        self.lane = min(LANES - 1, self.lane + 1)
        self.update_rect()

    def draw(self, screen):
        body = pygame.Rect(0, 0, PLAYER_W, PLAYER_H)
        body.center = self.rect.center
        pygame.draw.rect(screen, (80, 200, 255), body, border_radius=25)
        pygame.draw.rect(screen, (0, 0, 0), body, width=3, border_radius=25)

        head_r = 18
        head_center = (self.rect.centerx, self.rect.centery - PLAYER_H // 2 + 18)
        pygame.draw.circle(screen, (255, 230, 200), head_center, head_r)
        pygame.draw.circle(screen, (0, 0, 0), head_center, head_r, 2)

# -----------------------------
# Main
# -----------------------------
def main():
    pygame.init()
    pygame.display.set_caption("AR Runner (Subway Surfers Style) - Hand Controlled")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock = pygame.time.Clock()

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try changing CAM_INDEX.")

    # Make camera deliver something close to our window size (optional; helps some webcams)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIN_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65
    )

    # Calibration state
    state = "CALIBRATING"  # CALIBRATING -> CONFIRM -> PLAYING -> GAMEOVER
    confirm_start = None

    base_left_y = None
    base_right_y = None

    left_hist = deque()
    right_hist = deque()

    smooth_left_lift = 0.0
    smooth_right_lift = 0.0

    last_switch_time = 0.0
    left_armed = True   # edge trigger
    right_armed = True

    # Game state
    player = Player()
    obstacles = []
    last_spawn = time.time()
    start_time = time.time()
    score = 0.0
    speed = SPEED_START

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
            base_left_y = None
            base_right_y = None
            left_hist.clear()
            right_hist.clear()
            smooth_left_lift = 0.0
            smooth_right_lift = 0.0
            left_armed = True
            right_armed = True
            last_switch_time = 0.0

            player = Player()
            obstacles = []
            last_spawn = time.time()
            start_time = time.time()
            score = 0.0
            speed = SPEED_START

        # Grab camera frame
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)  # mirror for natural control

        # Hand tracking
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        left_y = None
        right_y = None

        # Draw mesh on OpenCV frame
        if res.multi_hand_landmarks and res.multi_handedness:
            for hand_lms, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
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

        # -------------- CALIBRATION --------------
        if state == "CALIBRATING":
            # collect only when BOTH hands visible
            if left_y is not None and right_y is not None:
                left_hist.append((now, left_y))
                right_hist.append((now, right_y))

            # Trim history
            while left_hist and (now - left_hist[0][0] > CALIBRATION_BUFFER_SEC):
                left_hist.popleft()
            while right_hist and (now - right_hist[0][0] > CALIBRATION_BUFFER_SEC):
                right_hist.popleft()

            # If enough stable data, finalize
            if len(left_hist) >= 2 and len(right_hist) >= 2:
                lspan = left_hist[-1][0] - left_hist[0][0]
                rspan = right_hist[-1][0] - right_hist[0][0]
                if lspan >= CALIBRATION_STABLE_TIME and rspan >= CALIBRATION_STABLE_TIME:
                    lvals = np.array([v for _, v in left_hist], dtype=np.float32)
                    rvals = np.array([v for _, v in right_hist], dtype=np.float32)
                    if float(lvals.std()) <= CALIBRATION_MAX_STD and float(rvals.std()) <= CALIBRATION_MAX_STD:
                        base_left_y = float(lvals.mean())
                        base_right_y = float(rvals.mean())
                        state = "CONFIRM"
                        confirm_start = now

        elif state == "CONFIRM":
            if (now - confirm_start) >= CONFIRM_HOLD:
                state = "PLAYING"
                start_time = time.time()
                last_spawn = time.time()
                score = 0.0
                speed = SPEED_START

        # -------------- PLAYING --------------
        if state == "PLAYING":
            # Update difficulty
            score = now - start_time
            speed = SPEED_START + SPEED_GROWTH * score

            # Compute lifts
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

            # Edge-triggered lane switching
            can_switch = (now - last_switch_time) >= COOLDOWN

            # Re-arm when hand goes back down
            if smooth_left_lift <= (LIFT_TRIGGER * 0.35):
                left_armed = True
            if smooth_right_lift <= (LIFT_TRIGGER * 0.35):
                right_armed = True

            if can_switch:
                both = (smooth_left_lift >= LIFT_TRIGGER) and (smooth_right_lift >= LIFT_TRIGGER)
                if not both:
                    if smooth_left_lift >= LIFT_TRIGGER and left_armed:
                        player.move_left()
                        left_armed = False
                        last_switch_time = now
                    elif smooth_right_lift >= LIFT_TRIGGER and right_armed:
                        player.move_right()
                        right_armed = False
                        last_switch_time = now

            # Spawn obstacles
            if (now - last_spawn) >= SPAWN_INTERVAL:
                lane = np.random.randint(0, LANES)
                obstacles.append(Obstacle(lane, y=-OB_H))
                last_spawn = now

            # Update obstacles
            for ob in obstacles:
                ob.update(dt, speed)

            # Remove off-screen obstacles
            obstacles = [ob for ob in obstacles if ob.y < WIN_H + 120]

            # Collision
            for ob in obstacles:
                if player.rect.colliderect(ob.rect):
                    state = "GAMEOVER"
                    break

        # -------------- RENDER --------------
        bg = cvframe_to_pygame_surface(frame, (WIN_W, WIN_H))
        screen.blit(bg, (0, 0))

        # Lane lines
        for x in LANE_X:
            pygame.draw.line(screen, (255, 255, 255), (x, 0), (x, WIN_H), 2)

        # Game elements
        player.draw(screen)
        for ob in obstacles:
            ob.draw(screen)

        # HUD
        if state == "CALIBRATING":
            draw_text(screen, "CALIBRATION", 20, 18, 34, (255, 255, 0))
            draw_text(screen, "Hold BOTH hands LOW and STEADY (rest position).", 20, 60, 26)
            draw_text(screen, "When stable, it will auto-confirm and start.", 20, 92, 24)
            draw_text(screen, "Press R to recalibrate anytime. Esc/Q to quit.", 20, 126, 22)

            if SHOW_DEBUG:
                l_ok = "OK" if left_y is not None else "---"
                r_ok = "OK" if right_y is not None else "---"
                draw_text(screen, f"Hands: L={l_ok}  R={r_ok}", 20, 165, 22, (200, 255, 200))

        elif state == "CONFIRM":
            draw_text(screen, "Calibration complete!", 20, 18, 36, (0, 255, 0))
            draw_text(screen, "Lift LEFT hand to go left, RIGHT hand to go right.", 20, 62, 26)

        elif state == "PLAYING":
            draw_text(screen, f"Time: {score:0.2f}s", 20, 18, 30)
            draw_text(screen, f"Speed: {int(speed)}", 20, 52, 24)

            if SHOW_DEBUG and base_left_y is not None and base_right_y is not None:
                draw_text(screen, f"baseL={base_left_y:.3f} baseR={base_right_y:.3f}", 20, 84, 20, (200, 255, 200))
                draw_text(screen, f"liftL={smooth_left_lift:.3f} liftR={smooth_right_lift:.3f}", 20, 110, 20, (200, 255, 200))
                draw_text(screen, f"Lane: {player.lane+1}/{LANES}", 20, 136, 20, (200, 255, 200))

        elif state == "GAMEOVER":
            draw_text(screen, "GAME OVER", WIN_W // 2 - 130, WIN_H // 2 - 60, 48, (255, 80, 80))
            draw_text(screen, f"Survived: {score:0.2f}s", WIN_W // 2 - 145, WIN_H // 2, 30, (255, 255, 255))
            draw_text(screen, "Press R to restart (recalibrates).", WIN_W // 2 - 220, WIN_H // 2 + 45, 24, (255, 255, 0))

        pygame.display.flip()

    hands.close()
    cap.release()
    pygame.quit()

if __name__ == "__main__":
    main()
