import cv2

cap0 = cv2.VideoCapture(0)  # MacBook
cap1 = cv2.VideoCapture(1)  # iPhone (change to 2 if that's where it was found)

if not cap0.isOpened():
    print("Error: Camera 0 (MacBook) not found")
    exit()
if not cap1.isOpened():
    print("Error: Camera 1 (iPhone) not found")
    exit()

print("Both cameras opened!")

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0 or not ret1:
        print("Error reading from one of the cameras")
        break

    # Resize both to same height for side-by-side display
    frame0 = cv2.resize(frame0, (640, 480))
    frame1 = cv2.resize(frame1, (640, 480))

    # Show side by side
    import numpy as np
    combined = np.hstack((frame0, frame1))
    cv2.imshow("Left: iphone | Right: macbook", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()