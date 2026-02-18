import cv2
import numpy as np
import mediapipe as mp
import screen_brightness_control as sbc
from math import hypot
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL       
import win32api           # imports for media control
import win32con

# Functions to send media and arrow key events  
def send_media_key(vk_code):
    """Send a media or arrow key press/release"""
    hwcode = win32api.MapVirtualKey(vk_code, 0)
    win32api.keybd_event(vk_code, hwcode, 0, 0)          # press
    win32api.keybd_event(vk_code, hwcode, win32con.KEYEVENTF_KEYUP, 0)  # release

# Send arrow keys
def send_arrow_key(vk_code):
    """Send arrow key (left/right)"""
    send_media_key(vk_code)

# Alternative using pyautogui 
# import pyautogui
# def send_media_key(vk_code):
#     if vk_code == win32con.VK_MEDIA_PLAY_PAUSE:
#         pyautogui.press('playpause')   # or 'space' for many players
#     elif vk_code == win32con.VK_RIGHT:
#         pyautogui.press('right')
#     elif vk_code == win32con.VK_LEFT:
#         pyautogui.press('left')

def main():
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    volRange = volume.GetVolumeRange()
    minVol, maxVol, _ = volRange

    mpHands = mp.solutions.hands
    hands = mpHands.Hands(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        max_num_hands=2)

    draw = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)

    # Debounce / state
    last_gesture = None
    gesture_cooldown = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed = hands.process(frameRGB)

            left_landmark_list, right_landmark_list = get_left_right_landmarks(frame, processed, draw, mpHands)

            both_hands_detected = bool(left_landmark_list and right_landmark_list)
            only_right_hand = bool(right_landmark_list and not left_landmark_list)

            # ── Brightness ── always when left hand is present
            if left_landmark_list:
                left_distance = get_distance(frame, left_landmark_list)
                if left_distance is not None:
                    b_level = np.interp(left_distance, [50, 220], [0, 100])
                    sbc.set_brightness(int(b_level))
                    cv2.putText(frame, f'Brightness: {int(b_level)}%', 
                                (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

            # ── Right hand section ──
            if right_landmark_list:
                # Volume pinching → only when BOTH hands are detected
                if both_hands_detected:
                    right_distance = get_distance(frame, right_landmark_list)
                    if right_distance is not None:
                        vol = np.interp(right_distance, [50, 220], [minVol, maxVol])
                        volume.SetMasterVolumeLevel(vol, None)
                        vol_percent = int(np.interp(vol, [minVol, maxVol], [0, 100]))
                        cv2.putText(frame, f'Volume: {vol_percent}%', 
                                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                        cv2.putText(frame, "Mode: VOLUME", (10, 30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

                # Media gestures → only when RIGHT HAND ALONE
                if only_right_hand:
                    # Show mode
                    cv2.putText(frame, "Mode: MEDIA", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 100, 100), 2)

                    # Finger counting logic
                    if processed.multi_hand_landmarks:
                        for hand_landmarks, handedness in zip(processed.multi_hand_landmarks, processed.multi_handedness):
                            if handedness.classification[0].label == "Right":
                                lm = hand_landmarks.landmark
                                h, w, _ = frame.shape

                                fingers_up = 0
                                finger_tips = [8, 12, 16, 20]  # index, middle, ring, pinky

                                # Thumb (approximate - tip left of IP joint)
                                if lm[4].x < lm[3].x:
                                    fingers_up += 1

                                # Other fingers: tip higher than PIP
                                for tip in finger_tips:
                                    if lm[tip].y < lm[tip - 2].y:
                                        fingers_up += 1

                                cv2.putText(frame, f'Fingers up: {fingers_up}', 
                                            (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

                                # Gesture execution with cooldown
                                if gesture_cooldown > 0:
                                    gesture_cooldown -= 1
                                else:
                                    current_gesture = fingers_up
                                    if current_gesture != last_gesture:
                                        if current_gesture == 1:
                                            send_media_key(win32con.VK_MEDIA_PLAY_PAUSE)
                                            cv2.putText(frame, "Play/Pause", (10, 180), 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                            gesture_cooldown = 15

                                        elif current_gesture == 2:
                                            send_arrow_key(win32con.VK_RIGHT)
                                            cv2.putText(frame, "Forward 10s", (10, 180), 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                            gesture_cooldown = 15

                                        elif current_gesture == 3:
                                            send_arrow_key(win32con.VK_LEFT)
                                            cv2.putText(frame, "Backward 10s", (10, 180), 
                                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                                            gesture_cooldown = 15

                                        last_gesture = current_gesture

            cv2.imshow('Image', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

# Get landmarks for left and right hands
def get_left_right_landmarks(frame, processed, draw, mpHands):
    left_landmark_list = []
    right_landmark_list = []

    if processed.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(processed.multi_hand_landmarks, processed.multi_handedness):
            hand_label = handedness.classification[0].label  # "Left" or "Right"

            draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            for idx, lm in enumerate(hand_landmarks.landmark):
                if idx == 4 or idx == 8:
                    h, w, _ = frame.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    if hand_label == "Left":
                        left_landmark_list.append([idx, x, y])
                    else:
                        right_landmark_list.append([idx, x, y])

    return left_landmark_list, right_landmark_list

# Calculate distance between thumb and index finger
def get_distance(frame, landmark_list):
    if len(landmark_list) < 2:
        return None
    (x1, y1), (x2, y2) = (landmark_list[0][1], landmark_list[0][2]), \
                         (landmark_list[1][1], landmark_list[1][2])
    cv2.circle(frame, (x1, y1), 7, (0, 255, 0), cv2.FILLED)
    cv2.circle(frame, (x2, y2), 7, (0, 255, 0), cv2.FILLED)
    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    L = hypot(x2 - x1, y2 - y1)
    return L

# Entry point
if __name__ == '__main__':
    main()