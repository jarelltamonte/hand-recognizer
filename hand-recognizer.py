import cv2
import mediapipe as mp
import time
import pyautogui
from pynput.keyboard import Key, Controller

# ─────────────────────────────────────────
# Init
# ─────────────────────────────────────────
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
keyboard = Controller()

cap = cv2.VideoCapture(0)

# ── Timings ──────────────────────────────
COOLDOWN_NAV  = 1.5   # seconds between next / prev
VOLUME_RATE   = 0.12  # seconds between each volume tick while held
NO_HAND_RESET = 0.4   # seconds before nav lock resets after hand leaves

# ── Nav cooldown ─────────────────────────
last_action_time  = 0
last_no_hand_time = None

# ── Volume ───────────────────────────────
volume_level     = 50   # internal 0–100 tracker
VOLUME_STEP      = 2
last_volume_time = 0

# ── Gesture overlay ──────────────────────
gesture_text    = ""
gesture_time    = 0
GESTURE_DISPLAY = 1.0

# ── Detection tolerances ─────────────────
THRESHOLD = 0.025   # finger up/down sensitivity

prev_time = 0
pyautogui.FAILSAFE = True


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────
def finger_up(tip, pip):
    """Tip clearly above pip = extended upward."""
    return tip.y < pip.y - THRESHOLD

def index_pointing_up(lm):
    """
    Index clearly extended upward AND roughly vertical:
    tip is above pip (y), and tip x is close to mcp x (not tilted sideways).
    """
    tip, pip, mcp = lm[8], lm[6], lm[5]
    extended  = tip.y < pip.y - THRESHOLD
    vertical  = abs(tip.x - mcp.x) < 0.06   # not tilted sideways
    return extended and vertical

def index_pointing_sideways(lm):
    """
    Index is horizontal: tip is at roughly the same height as pip
    but significantly to the left or right of mcp.
    Returns 'right', 'left', or None.
    """
    tip, pip, mcp = lm[8], lm[6], lm[5]
    # Tip should NOT be above pip (not pointing up)
    not_up    = tip.y >= pip.y - 0.01
    # Horizontal displacement is significant
    dx        = tip.x - mcp.x
    if not_up and dx > 0.07:
        return "right"
    elif not_up and dx < -0.07:
        return "left"
    return None

def get_hand_side(lm, frame_width):
    return "right" if lm[0].x * frame_width > frame_width / 2 else "left"

def show_gesture(text, t):
    global gesture_text, gesture_time
    gesture_text = text
    gesture_time = t

def draw_volume_bar(frame, volume, frame_w, frame_h):
    """Vertical volume bar on the right edge."""
    bx        = frame_w - 52
    bar_top   = 80
    bar_bot   = frame_h - 80
    bar_h     = bar_bot - bar_top
    bar_w     = 26
    filled_h  = int(bar_h * volume / 100)

    # Track background
    cv2.rectangle(frame, (bx, bar_top), (bx + bar_w, bar_bot), (40, 40, 40), -1)
    cv2.rectangle(frame, (bx, bar_top), (bx + bar_w, bar_bot), (80, 80, 80), 1)

    # Filled portion (bottom-up), color shifts blue→green→red
    if filled_h > 0:
        ratio = volume / 100
        r = int(min(255, ratio * 2 * 255))
        g = int(min(255, (1 - abs(ratio - 0.5) * 2) * 255 + 80))
        b = int(max(0, (1 - ratio * 2) * 255))
        cv2.rectangle(frame,
                      (bx, bar_bot - filled_h),
                      (bx + bar_w, bar_bot),
                      (b, g, r), -1)

    # Tick marks every 25%
    for pct in [25, 50, 75]:
        ty = bar_bot - int(bar_h * pct / 100)
        cv2.line(frame, (bx - 4, ty), (bx, ty), (140, 140, 140), 1)

    # Percentage label above bar
    label = f"{volume}%"
    (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.putText(frame, label,
                (bx + bar_w // 2 - lw // 2, bar_top - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    cv2.putText(frame, "VOL",
                (bx + 1, bar_bot + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)


# ─────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_h, frame_w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results   = hands.process(frame_rgb)

    current_time = time.time()

    # ─────────────────────────────────────
    # No hand detected
    # ─────────────────────────────────────
    if not results.multi_hand_landmarks:
        if last_no_hand_time is None:
            last_no_hand_time = current_time
        elif current_time - last_no_hand_time > NO_HAND_RESET:
            # Stamp now so next/prev can't fire the instant hand reappears
            last_action_time = current_time

    else:
        last_no_hand_time = None

        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

            lm = hand_landmarks.landmark

            index_up_flag = finger_up(lm[8], lm[6])
            middle_open   = finger_up(lm[12], lm[10])
            ring_open     = finger_up(lm[16], lm[14])
            pinky_open    = finger_up(lm[20], lm[18])

            # ── Gesture classification ──────────────
            is_open_palm   = index_up_flag and middle_open and ring_open and pinky_open
            # Index alone vertical (not tilted)
            is_vol_up      = (index_pointing_up(lm)
                              and not middle_open and not ring_open and not pinky_open)
            # Index tilted sideways (left or right)
            side           = index_pointing_sideways(lm)
            is_vol_down    = (side is not None
                              and not middle_open and not ring_open and not pinky_open)

            hand_side = get_hand_side(lm, frame_w)

            # ─────────────────────────────────────
            # ☝  VOLUME UP — index straight up
            # ─────────────────────────────────────
            if is_vol_up and not is_open_palm:
                if current_time - last_volume_time > VOLUME_RATE:
                    keyboard.press(Key.media_volume_up)
                    keyboard.release(Key.media_volume_up)
                    volume_level     = min(100, volume_level + VOLUME_STEP)
                    last_volume_time = current_time
                    show_gesture(f"🔊 VOL UP  {volume_level}%", current_time)

            # ─────────────────────────────────────
            # 👉  VOLUME DOWN — index tilted sideways
            # ─────────────────────────────────────
            elif is_vol_down:
                if current_time - last_volume_time > VOLUME_RATE:
                    keyboard.press(Key.media_volume_down)
                    keyboard.release(Key.media_volume_down)
                    volume_level     = max(0, volume_level - VOLUME_STEP)
                    last_volume_time = current_time
                    show_gesture(f"🔉 VOL DOWN  {volume_level}%", current_time)

            # ─────────────────────────────────────
            # 🖐  NEXT / PREV — open palm by position
            # ─────────────────────────────────────
            can_nav = (current_time - last_action_time) > COOLDOWN_NAV

            if is_open_palm and can_nav:
                if hand_side == "right":
                    pyautogui.click(500, 500)
                    pyautogui.hotkey("shift", "n")
                    show_gesture("NEXT ▶▶", current_time)
                else:
                    pyautogui.click(500, 500)
                    pyautogui.hotkey("shift", "p")
                    show_gesture("◀◀ PREV", current_time)
                last_action_time = current_time

    # ── Volume bar ────────────────────────
    draw_volume_bar(frame, volume_level, frame_w, frame_h)

    # ── Gesture overlay ───────────────────
    if current_time - gesture_time < GESTURE_DISPLAY:
        (tw, th), _ = cv2.getTextSize(gesture_text, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
        rx1, ry1 = 10, frame_h - 100
        rx2, ry2 = rx1 + tw + 20, ry1 + th + 20
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 0), -1)
        cv2.putText(frame, gesture_text, (rx1 + 10, ry2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 255), 3)

    # ── FPS ───────────────────────────────
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # ── Legend ────────────────────────────
    legend = [
        "Gestures:",
        "☝ Index up      -> Vol Up",
        "👉 Index sideways -> Vol Down",
        "🖐 Open palm (R)  -> Next",
        "🖐 Open palm (L)  -> Prev",
    ]
    for i, line in enumerate(legend):
        cv2.putText(frame, line, (frame_w - 400, 30 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.imshow("Hand Gesture Controller", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()