# 🖐️ Real Time Hand Gesture Multimedia & System Controller

A real-time computer vision system that enables touchless control of system volume, screen brightness, and media playback using hand gestures detected via webcam. The application leverages hand landmark detection and maps gestures to system-level commands for intuitive human–computer interaction

# 📌 Features

- 🎯 Real-time hand tracking using MediaPipe

- 🔊 Dynamic system volume control using pinch gestures

- 💡 Screen brightness adjustment via left-hand gestures

- ⏯ Media playback control (Play/Pause, Forward, Backward)

- 🧠 Finger-count–based gesture recognition

- ⛔ Cooldown mechanism to prevent repeated triggers

- 🖥 Seamless integration with Windows system APIs

# 🛠 Tech Stack

- Python

- OpenCV

- MediaPipe

- NumPy

- pycaw (Windows Core Audio API)

- screen_brightness_control

- Win32 API (win32api, win32con)

# 🧠 How It Works

1. The webcam captures real-time video input.
2. MediaPipe detects hand landmarks and classifies left/right hands.
3. The distance between thumb and index finger is calculated.
4. Distance values are interpolated to:
   - Adjust brightness (Left Hand)
   - Adjust volume (Both Hands Detected)
5. When only the right hand is visible:
   - 1 Finger → Play/Pause
   - 2 Fingers → Forward
   - 3 Fingers → Backward

A gesture cooldown mechanism ensures stable command execution.

# 💻 Installation
- 1️⃣ Clone the Repository
  - git clone [https://github.com/kushalkrd92/Real-Time-Hand-Gesture-Multimedia-and-System-Controller]
  - cd gesture-control-system

- 2️⃣ Create Virtual Environment (Recommended)
  - python -m venv myenv
  - myenv/bin/activate   # Mac/Linux
  - venv\Scripts\activate      # Windows

- 3️⃣ Install Dependencies
  - pip install -r requirements.txt

# ▶️ Usage

Run the main script:

- python merged.py
  
Press q to exit the application.

# 🎮 Controls
| Gesture                | Action            |
| ---------------------- | ----------------- |
| Left-hand pinch        | Adjust Brightness |
| Both-hands pinch       | Adjust Volume     |
| 1 Finger (Right hand)  | Play/Pause        |
| 2 Fingers (Right hand) | Forward           |
| 3 Fingers (Right hand) | Backward          |

# 📋 System Requirements

- Windows OS (for pycaw and Win32 API support)
- Python 3.8+
- Webcam
