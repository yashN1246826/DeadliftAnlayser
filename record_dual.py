import cv2
import numpy as np
import time
import os

os.makedirs("data/recordings", exist_ok=True)

# Camera setup
cap0 = cv2.VideoCapture(0)  # Front camera (Logitech C920)
cap1 = cv2.VideoCapture(1)  # Side camera (iPhone)

# Force both cameras to the same resolution for consistency
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap0.set(cv2.CAP_PROP_FPS, 30)

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap1.set(cv2.CAP_PROP_FPS, 30)

if not cap0.isOpened() or not cap1.isOpened():
    print("Error: Could not open cameras")
    exit()

# Get properties after applying settings
fps = 30.0
w0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
h0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera 0 (Front - Logitech): {w0}x{h0}")
print(f"Camera 1 (Side - iPhone): {w1}x{h1}")

# Video writers - save at full resolution
timestamp = time.strftime("%Y%m%d_%H%M%S")
session_dir = f"data/recordings/{timestamp}"
os.makedirs(session_dir, exist_ok=True)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out0 = cv2.VideoWriter(f"{session_dir}/cam0.mp4", fourcc, fps, (w0, h0))
out1 = cv2.VideoWriter(f"{session_dir}/cam1.mp4", fourcc, fps, (w1, h1))

recording = False
frame_count = 0

print("\n=== DUAL VIDEO RECORDER ===")
print("Camera 0 = Front view (Logitech C920)")
print("Camera 1 = Side view (iPhone)")
print("Press 'r' to start/stop recording")
print("Press 'q' to quit\n")

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0 or not ret1:
        print("Error: Failed to read from one or both cameras")
        break

    if recording:
        out0.write(frame0)
        out1.write(frame1)
        frame_count += 1

    # Display copies only
    d0 = cv2.resize(frame0, (640, 480))
    d1 = cv2.resize(frame1, (640, 480))

    status = "RECORDING" if recording else "READY"
    colour = (0, 0, 255) if recording else (0, 255, 0)

    # Front camera labels
    cv2.putText(d0, "FRONT - LOGITECH", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(d0, status, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    cv2.putText(d0, f"Frames: {frame_count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Side camera labels
    cv2.putText(d1, "SIDE - IPHONE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(d1, status, (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
    cv2.putText(d1, f"Frames: {frame_count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    combined = np.hstack((d0, d1))
    cv2.imshow("Dual Recorder - R to record, Q to quit", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('r'):
        recording = not recording
        if recording:
            print("Recording started!")
            frame_count = 0
        else:
            print(f"Recording stopped. {frame_count} frames captured.")
    elif key == ord('q'):
        break

out0.release()
out1.release()
cap0.release()
cap1.release()
cv2.destroyAllWindows()

print(f"Videos saved to {session_dir}/")