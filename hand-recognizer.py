import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Initialize MediaPipe Drawing module for drawing landmarks
mp_drawing = mp.solutions.drawing_utils

# Open a video capture object (0 for the default camera)
cap = cv2.VideoCapture(0)

prev_time = 0

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect hands
    results = hands.process(frame_rgb)
    
    # Check if hands are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumbs_tip = hand_landmarks.landmark[4]
            thumbs_pip = hand_landmarks.landmark[3]
                
            if (thumbs_tip.x < thumbs_pip.x) or (thumbs_tip.y < thumbs_pip.y):
                cv2.putText(frame, "THUMB UP", (20, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

            index_tip = hand_landmarks.landmark[8]
            index_pip = hand_landmarks.landmark[6]

            if index_tip.y < index_pip.y:
                cv2.putText(frame, "INDEX UP", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
                
            middle_tip = hand_landmarks.landmark[12]
            middle_pip = hand_landmarks.landmark[10]
                
            if middle_tip.y < middle_pip.y:
                cv2.putText(frame, "MIDDLE UP", (20, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
            ring_tip = hand_landmarks.landmark[16]
            ring_pip = hand_landmarks.landmark[14]
                
            if ring_tip.y < ring_pip.y:
                cv2.putText(frame, "RING UP", (20, 270),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
                
            pinky_tip = hand_landmarks.landmark[20]
            pinky_pip = hand_landmarks.landmark[18]
                
            if pinky_tip.y < pinky_pip.y:
                cv2.putText(frame, "PINKY UP", (20, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    current_time = time.time()

    if prev_time == 0:
        fps = 0
    else:
        fps = 1 / (current_time - prev_time)

    prev_time = current_time

    cv2.putText(frame, f'FPS: {int(fps)}', (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)
    
    # Display the frame with hand landmarks
    cv2.imshow('Hand Recognition', frame)
    
    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()