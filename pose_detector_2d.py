import cv2
import mediapipe as mp
import numpy as np
import os
import multiprocessing

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def extract_landmarks(video_path, output_dir, camera_name):
    """
    Run MediaPipe BlazePose on a video and extract 2D landmarks per frame.
    Saves landmarks as a .npz file and an annotated preview video.

    Safe for multiprocessing — creates its own Pose instance so it can
    run inside a separate Process without any pickling issues.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return None

    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n[{camera_name}] Starting — {width}x{height}, "
          f"{fps:.0f} fps, {total_frames} frames")

    # Each process creates its own Pose instance (cannot be shared/pickled)
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    all_landmarks    = []
    frames_with_pose = 0

    os.makedirs(output_dir, exist_ok=True)
    preview_path = os.path.join(output_dir, f"{camera_name}_pose_preview.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(preview_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        frame_data = np.zeros((33, 4))   # x, y, z, visibility

        if results.pose_landmarks:
            frames_with_pose += 1
            for i, lm in enumerate(results.pose_landmarks.landmark):
                frame_data[i] = [lm.x, lm.y, lm.z, lm.visibility]

            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0),
                                       thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        all_landmarks.append(frame_data)
        out.write(frame)

        frame_idx += 1
        if frame_idx % 30 == 0:
            pct = frame_idx / total_frames * 100
            print(f"  [{camera_name}] {frame_idx}/{total_frames} "
                  f"({pct:.0f}%)", end='\r')

    pose.close()
    cap.release()
    out.release()

    landmarks_arr = np.array(all_landmarks)

    landmarks_path = os.path.join(output_dir, f"{camera_name}_landmarks.npz")
    np.savez(
        landmarks_path,
        landmarks=landmarks_arr,
        fps=fps,
        width=width,
        height=height,
        total_frames=len(landmarks_arr),
        frames_with_pose=frames_with_pose,
    )

    rate = frames_with_pose / len(landmarks_arr) * 100
    print(f"\n  [{camera_name}] Done — {frames_with_pose}/{len(landmarks_arr)} "
          f"frames ({rate:.1f}% detected)")
    print(f"  [{camera_name}] Saved: {landmarks_path}")
    return landmarks_arr


# ── Worker: top-level function so multiprocessing can pickle it ──

def _worker(video_path, output_dir, camera_name):
    extract_landmarks(video_path, output_dir, camera_name)


# ── Main parallel entry point ────────────────────────────────

def process_recording_parallel(recording_dir):
    """
    Find cam0 and cam1 videos and run MediaPipe on BOTH simultaneously.
    Time taken = max(cam0_time, cam1_time) instead of cam0_time + cam1_time.
    """
    files     = os.listdir(recording_dir)
    cam0_file = next(
        (f for f in files if "cam0" in f and f.endswith(".mp4")), None
    )
    cam1_file = next(
        (f for f in files if "cam1" in f and f.endswith(".mp4")), None
    )

    if not cam0_file or not cam1_file:
        print(f"Error: cam0/cam1 not found in {recording_dir}")
        print(f"Files present: {files}")
        return None, None

    output_dir = os.path.join(recording_dir, "processed")
    os.makedirs(output_dir, exist_ok=True)

    cam0_path = os.path.join(recording_dir, cam0_file)
    cam1_path = os.path.join(recording_dir, cam1_file)

    print("=" * 50)
    print("2D POSE ESTIMATION — PARALLEL (cam0 + cam1 simultaneously)")
    print("=" * 50)

    p0 = multiprocessing.Process(
        target=_worker, args=(cam0_path, output_dir, "cam0")
    )
    p1 = multiprocessing.Process(
        target=_worker, args=(cam1_path, output_dir, "cam1")
    )

    p0.start()
    p1.start()
    p0.join()
    p1.join()

    if p0.exitcode != 0:
        print("Error: cam0 processing failed (exit code "
              f"{p0.exitcode})")
        return None, None
    if p1.exitcode != 0:
        print("Error: cam1 processing failed (exit code "
              f"{p1.exitcode})")
        return None, None

    # Verify outputs
    f0 = os.path.join(output_dir, "cam0_landmarks.npz")
    f1 = os.path.join(output_dir, "cam1_landmarks.npz")
    if not os.path.exists(f0) or not os.path.exists(f1):
        print("Error: landmark .npz files missing after processing")
        return None, None

    d0 = np.load(f0)
    d1 = np.load(f1)
    len0, len1 = len(d0['landmarks']), len(d1['landmarks'])

    print(f"\nFrame count — Cam0: {len0}, Cam1: {len1}")
    if len0 != len1:
        print(f"Trimming to {min(len0, len1)} frames for synchronisation")

    print("\n2D pose estimation complete. Ready for 3D triangulation.")
    return len0, len1


if __name__ == "__main__":
    # Required for multiprocessing on macOS (uses 'spawn' start method)
    multiprocessing.freeze_support()

    from utils import get_latest_session
    session_dir = get_latest_session()
    process_recording_parallel(session_dir)
