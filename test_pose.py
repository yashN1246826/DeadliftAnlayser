import cv2
import mediapipe as mp

# Initialise MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

print("MediaPipe Pose running! Stand back so your full body is visible.")
print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipe needs RGB, OpenCV gives BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # Draw landmarks if a body is detected
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
            mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        # Print a few key joint positions to confirm data extraction
        landmarks = results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        
        print(f"Shoulder: ({left_shoulder.x:.2f}, {left_shoulder.y:.2f}) | "
              f"Hip: ({left_hip.x:.2f}, {left_hip.y:.2f}) | "
              f"Knee: ({left_knee.x:.2f}, {left_knee.y:.2f})", end='\r')
    else:
        print("No body detected - stand further back", end='\r')

    cv2.imshow("MediaPipe Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.close()
cap.release()
cv2.destroyAllWindows()
print("\nDone!")