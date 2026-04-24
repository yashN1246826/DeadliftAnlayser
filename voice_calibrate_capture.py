import cv2
import os
import time
import numpy as np
import threading
import speech_recognition as sr

# Save folders
os.makedirs("data/calib_cam0", exist_ok=True)
os.makedirs("data/calib_cam1", exist_ok=True)

# Cameras
cap0 = cv2.VideoCapture(0)  # Front camera (Logitech C920)
cap1 = cv2.VideoCapture(1)  # Side camera (iPhone)

# Match same camera settings as the rest of the project
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap0.set(cv2.CAP_PROP_FPS, 30)

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap1.set(cv2.CAP_PROP_FPS, 30)

if not cap0.isOpened() or not cap1.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# Print actual resolutions
w0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
h0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera 0 (Front - Logitech): {w0}x{h0}")
print(f"Camera 1 (Side - iPhone): {w1}x{h1}")

# Chessboard settings
CHESS_COLS = 9
CHESS_ROWS = 6

TARGET_PAIRS = 20
COOLDOWN_SECONDS = 2.0
MIN_DETECTION_STREAK = 6

img_count = 0
last_capture_time = 0
detection_streak = 0

# Voice trigger flags
capture_requested = False
stop_requested = False
lock = threading.Lock()

def voice_listener():
    global capture_requested, stop_requested

    recognizer = sr.Recognizer()

    # Much less aggressive than before
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 80
    recognizer.pause_threshold = 0.5
    recognizer.non_speaking_duration = 0.2
    recognizer.phrase_threshold = 0.2

    # Force MacBook microphone
    mic = sr.Microphone(device_index=2)

    with mic as source:
        print("Calibrating microphone...")
        recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Voice listener ready.")
        print("Say 'capture now' to save, 'quit' to stop.")

    while not stop_requested:
        try:
            with mic as source:
                print("Listening...")
                audio = recognizer.listen(
                    source,
                    timeout=5,
                    phrase_time_limit=3
                )

            text = recognizer.recognize_google(audio).lower().strip()
            print(f"[Heard] {text}")

            with lock:
                if "capture now" in text or "capture" in text:
                    capture_requested = True
                    print("Capture requested.")
                elif "quit" in text or "exit" in text or "stop" in text:
                    stop_requested = True

        except sr.WaitTimeoutError:
            pass

        except sr.UnknownValueError:
            print("Didn't catch that clearly.")

        except sr.RequestError as e:
            print(f"Speech recognition error: {e}")
            break

        except Exception as e:
            print(f"Voice listener error: {e}")
            break

print("=== VOICE CALIBRATION CAPTURE ===")
print("Camera 0 = Front view (Logitech C920)")
print("Camera 1 = Side view (iPhone)")
print("Hold the chessboard so BOTH cameras can see it.")
print("Say 'capture now' when both views look good.")
print("Say 'quit' or press q to stop.\n")

listener_thread = threading.Thread(target=voice_listener, daemon=True)
listener_thread.start()

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0 or not ret1:
        print("Error reading from cameras.")
        break

    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    found0, corners0 = cv2.findChessboardCorners(gray0, (CHESS_COLS, CHESS_ROWS), None)
    found1, corners1 = cv2.findChessboardCorners(gray1, (CHESS_COLS, CHESS_ROWS), None)

    display0 = frame0.copy()
    display1 = frame1.copy()

    if found0:
        cv2.drawChessboardCorners(display0, (CHESS_COLS, CHESS_ROWS), corners0, found0)
    if found1:
        cv2.drawChessboardCorners(display1, (CHESS_COLS, CHESS_ROWS), corners1, found1)

    status0 = "FOUND" if found0 else "NOT FOUND"
    status1 = "FOUND" if found1 else "NOT FOUND"

    color0 = (0, 255, 0) if found0 else (0, 0, 255)
    color1 = (0, 255, 0) if found1 else (0, 0, 255)

    cv2.putText(display0, "FRONT - LOGITECH", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display0, f"Chessboard: {status0}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color0, 2)
    cv2.putText(display0, f"Captured: {img_count}/{TARGET_PAIRS}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(display1, "SIDE - IPHONE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display1, f"Chessboard: {status1}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color1, 2)
    cv2.putText(display1, f"Captured: {img_count}/{TARGET_PAIRS}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    now = time.time()

    if found0 and found1:
        detection_streak += 1
    else:
        detection_streak = 0

    with lock:
        do_capture = capture_requested

    ready_to_capture = (
        do_capture and
        found0 and found1 and
        detection_streak >= MIN_DETECTION_STREAK and
        (now - last_capture_time) >= COOLDOWN_SECONDS
    )

    if ready_to_capture:
        filename = f"img_{img_count:03d}.png"
        cv2.imwrite(os.path.join("data/calib_cam0", filename), frame0)
        cv2.imwrite(os.path.join("data/calib_cam1", filename), frame1)

        img_count += 1
        last_capture_time = now
        detection_streak = 0

        with lock:
            capture_requested = False

        print(f"Captured pair {img_count}/{TARGET_PAIRS}")

    remaining = max(0, COOLDOWN_SECONDS - (now - last_capture_time))
    cv2.putText(display0, f"Cooldown: {remaining:.1f}s", (10, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

    display0 = cv2.resize(display0, (640, 480))
    display1 = cv2.resize(display1, (640, 480))
    combined = np.hstack((display0, display1))

    cv2.imshow("Voice Calibration Capture", combined)

    if img_count >= TARGET_PAIRS:
        print("\nFinished capturing target pairs.")
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nStopped by keyboard.")
        break

    with lock:
        if stop_requested:
            print("\nStopped by voice command.")
            break

cap0.release()
cap1.release()
cv2.destroyAllWindows()
print("Done.")