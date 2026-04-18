import cv2
import mediapipe as mp
import time
import pyautogui

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Camera
cap = cv2.VideoCapture(0)

# Timing
prev_time = 0
last_action_time = 0
cooldown = 1  # seconds

# Gesture display
gesture_text = ""
gesture_time = 0

# Swipe tracking
prev_x = 0
swipe_threshold = 0.08

# Finger tolerance
threshold = 0.02

# Safety
pyautogui.FAILSAFE = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    current_time = time.time()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            # Landmarks
            index_tip = hand_landmarks.landmark[8]
            index_pip = hand_landmarks.landmark[6]

            middle_tip = hand_landmarks.landmark[12]
            middle_pip = hand_landmarks.landmark[10]

            ring_tip = hand_landmarks.landmark[16]
            ring_pip = hand_landmarks.landmark[14]

            pinky_tip = hand_landmarks.landmark[20]
            pinky_pip = hand_landmarks.landmark[18]

            # =========================
            # ✊ PAUSE (ALL 4 FINGERS DOWN)
            # =========================
            if (index_tip.y > index_pip.y - threshold and
                middle_tip.y > middle_pip.y - threshold and
                ring_tip.y > ring_pip.y - threshold and
                pinky_tip.y > pinky_pip.y - threshold):

                if current_time - last_action_time > cooldown:
                    pyautogui.click(500, 500)  # focus browser
                    pyautogui.press('space')   # pause/play

                    last_action_time = current_time
                    gesture_text = "PAUSE"
                    gesture_time = current_time

            # =========================
            # 👉 SWIPE (USE WRIST)
            # =========================
            wrist = hand_landmarks.landmark[0]
            current_x = wrist.x

            if prev_x != 0:
                dx = current_x - prev_x

                if current_time - last_action_time > cooldown:

                    if dx > swipe_threshold:
                        pyautogui.click(500, 500)
                        pyautogui.press('l')   # forward

                        last_action_time = current_time
                        gesture_text = "NEXT ▶"
                        gesture_time = current_time

                    elif dx < -swipe_threshold:
                        pyautogui.click(500, 500)
                        pyautogui.press('j')   # rewind

                        last_action_time = current_time
                        gesture_text = "◀ PREV"
                        gesture_time = current_time

            prev_x = current_x

    # =========================
    # SHOW GESTURE TEXT (0.5 sec)
    # =========================
    if current_time - gesture_time < 0.5:
        cv2.putText(frame, gesture_text, (20, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # =========================
    # FPS DISPLAY
    # =========================
    if prev_time == 0:
        fps = 0
    else:
        fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    # Show window
    cv2.imshow('Hand Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()