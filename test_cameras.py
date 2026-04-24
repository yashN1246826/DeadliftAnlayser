import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera 0")
    exit()

print("Press q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    cv2.imshow("Camera 0 Live", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()