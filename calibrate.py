import cv2
import os
import numpy as np

# Create directories for calibration images
os.makedirs("data/calib_cam0", exist_ok=True)
os.makedirs("data/calib_cam1", exist_ok=True)

# Camera setup
cap0 = cv2.VideoCapture(0)  # Front camera (Logitech C920)
cap1 = cv2.VideoCapture(1)  # Side camera (iPhone)

# Force both cameras to the same resolution for stereo consistency
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap0.set(cv2.CAP_PROP_FPS, 30)

cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap1.set(cv2.CAP_PROP_FPS, 30)

if not cap0.isOpened() or not cap1.isOpened():
    print("Error: Could not open one or both cameras")
    exit()

# Print actual camera resolutions after applying settings
w0 = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH))
h0 = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))
w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Camera 0 (Front - Logitech): {w0}x{h0}")
print(f"Camera 1 (Side - iPhone): {w1}x{h1}")

# Chessboard inner corners (9 columns x 6 rows)
CHESS_ROWS = 6
CHESS_COLS = 9

img_count = 0
print("\n=== CALIBRATION IMAGE CAPTURE ===")
print("Camera 0 = Front view (Logitech C920)")
print("Camera 1 = Side view (iPhone)")
print("Hold the chessboard so BOTH cameras can see it.")
print("Press SPACE to capture when corners are detected in both views.")
print("Capture at least 20 image pairs from different angles.")
print("Press 'q' to finish.\n")

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0 or not ret1:
        print("Error reading cameras")
        break

    # Convert to grayscale for corner detection
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Try to find chessboard corners in both images
    found0, corners0 = cv2.findChessboardCorners(gray0, (CHESS_COLS, CHESS_ROWS), None)
    found1, corners1 = cv2.findChessboardCorners(gray1, (CHESS_COLS, CHESS_ROWS), None)

    # Draw corners on display copies
    display0 = frame0.copy()
    display1 = frame1.copy()

    if found0:
        cv2.drawChessboardCorners(display0, (CHESS_COLS, CHESS_ROWS), corners0, found0)
    if found1:
        cv2.drawChessboardCorners(display1, (CHESS_COLS, CHESS_ROWS), corners1, found1)

    # Status text
    status0 = "FOUND" if found0 else "NOT FOUND"
    status1 = "FOUND" if found1 else "NOT FOUND"

    colour0 = (0, 255, 0) if found0 else (0, 0, 255)
    colour1 = (0, 255, 0) if found1 else (0, 0, 255)

    cv2.putText(display0, "FRONT - LOGITECH", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display0, f"Chessboard: {status0}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour0, 2)
    cv2.putText(display0, f"Captured: {img_count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(display1, "SIDE - IPHONE", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display1, f"Chessboard: {status1}", (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour1, 2)
    cv2.putText(display1, f"Captured: {img_count}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Resize for display only
    display0 = cv2.resize(display0, (640, 480))
    display1 = cv2.resize(display1, (640, 480))
    combined = np.hstack((display0, display1))
    cv2.imshow("Calibration - Press SPACE to capture, Q to quit", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' ') and found0 and found1:
        cv2.imwrite(f"data/calib_cam0/img_{img_count:03d}.png", frame0)
        cv2.imwrite(f"data/calib_cam1/img_{img_count:03d}.png", frame1)
        img_count += 1
        print(f"Captured pair {img_count}!")

    elif key == ord(' ') and (not found0 or not found1):
        print("Cannot capture - chessboard not detected in both cameras!")

    elif key == ord('q'):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()

print(f"\nDone! Captured {img_count} image pairs.")
print("Need at least 20 good pairs for reliable calibration.")