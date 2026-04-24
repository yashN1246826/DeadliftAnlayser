import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Landmark indices - split by which view sees them best
LANDMARK_NAMES = {
    11: "left_shoulder",
    12: "right_shoulder",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    15: "left_wrist",
    16: "right_wrist",
    13: "left_elbow",
    14: "right_elbow",
}

# Midline landmarks (visible from both views - best for triangulation)
MIDLINE_LANDMARKS = {11, 12, 23, 24}  # shoulders and hips

SKELETON_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    ("mid_shoulder", "mid_hip"),  # spine line
]


def compute_mid_spine(points_3d):
    """Add mid-shoulder and mid-hip points for spine analysis."""
    if 11 in points_3d and 12 in points_3d:
        points_3d["mid_shoulder"] = (points_3d[11] + points_3d[12]) / 2
    if 23 in points_3d and 24 in points_3d:
        points_3d["mid_hip"] = (points_3d[23] + points_3d[24]) / 2
    return points_3d


def triangulate_frame(landmarks0, landmarks1, P0, P1, w0, h0, w1, h1):
    """
    Triangulate 3D points using a lower visibility threshold
    and smarter joint selection.
    """
    points_3d = {}
    
    for idx in LANDMARK_NAMES.keys():
        lm0 = landmarks0[idx]
        lm1 = landmarks1[idx]

        # Use lower threshold - MediaPipe still gives reasonable estimates
        # even for partially occluded joints
        vis0 = lm0[3]
        vis1 = lm1[3]
        
        # At least one camera must see it well, and the other reasonably
        if max(vis0, vis1) < 0.5 or min(vis0, vis1) < 0.1:
            continue

        # Convert normalised coords to pixels
        pt0 = np.array([lm0[0] * w0, lm0[1] * h0], dtype=np.float64)
        pt1 = np.array([lm1[0] * w1, lm1[1] * h1], dtype=np.float64)

        # Triangulate
        pts_4d = cv2.triangulatePoints(
            P0, P1,
            pt0.reshape(2, 1),
            pt1.reshape(2, 1)
        )
        
        # Convert from homogeneous coordinates
        if abs(pts_4d[3]) < 1e-6:
            continue
            
        pt3d = (pts_4d[:3] / pts_4d[3]).flatten()
        
        # Sanity check - reject points that are unreasonably far away
        # (more than 10 metres from origin)
        if np.any(np.abs(pt3d) > 10000):
            continue
            
        points_3d[idx] = pt3d

    # Add mid-spine points
    points_3d = compute_mid_spine(points_3d)
    
    return points_3d


def filter_and_smooth(all_3d_frames, window=5):
    """
    Smooth 3D trajectories over time to reduce jitter.
    Uses a simple moving average.
    """
    smoothed = []
    num_frames = len(all_3d_frames)
    
    for i in range(num_frames):
        frame_smooth = {}
        start = max(0, i - window // 2)
        end = min(num_frames, i + window // 2 + 1)
        
        # For each landmark, average its position over the window
        all_keys = set()
        for j in range(start, end):
            all_keys.update(all_3d_frames[j].keys())
        
        for key in all_keys:
            positions = []
            for j in range(start, end):
                if key in all_3d_frames[j]:
                    positions.append(all_3d_frames[j][key])
            
            if len(positions) >= window // 2:  # Need at least half the window
                frame_smooth[key] = np.mean(positions, axis=0)
        
        smoothed.append(frame_smooth)
    
    return smoothed


def reconstruct_3d(session_dir):
    """Full 3D reconstruction pipeline."""
    processed_dir = os.path.join(session_dir, "processed")

    # Load calibration
    calib = np.load("data/stereo_calib.npz")
    P0 = calib['P0']
    P1 = calib['P1']

    # Load 2D landmarks
    data0 = np.load(os.path.join(processed_dir, "cam0_landmarks.npz"))
    data1 = np.load(os.path.join(processed_dir, "cam1_landmarks.npz"))

    landmarks0 = data0['landmarks']
    landmarks1 = data1['landmarks']
    w0, h0 = float(data0['width']), float(data0['height'])
    w1, h1 = float(data1['width']), float(data1['height'])
    fps = float(data0['fps'])

    min_frames = min(len(landmarks0), len(landmarks1))
    landmarks0 = landmarks0[:min_frames]
    landmarks1 = landmarks1[:min_frames]

    print(f"Reconstructing 3D from {min_frames} frames...")

    # Triangulate all frames
    all_3d_frames = []
    for i in range(min_frames):
        points_3d = triangulate_frame(
            landmarks0[i], landmarks1[i], P0, P1, w0, h0, w1, h1
        )
        all_3d_frames.append(points_3d)

    # Count valid frames (need at least shoulders and hips)
    valid = sum(1 for f in all_3d_frames 
                if "mid_shoulder" in f and "mid_hip" in f)
    print(f"Frames with spine data: {valid}/{min_frames} ({100*valid/min_frames:.1f}%)")
    
    valid_full = sum(1 for f in all_3d_frames if len(f) >= 8)
    print(f"Frames with 8+ joints: {valid_full}/{min_frames} ({100*valid_full/min_frames:.1f}%)")

    # Smooth the trajectories
    print("Smoothing trajectories...")
    all_3d_frames = filter_and_smooth(all_3d_frames)

    # Save
    np.save(os.path.join(processed_dir, "landmarks_3d.npy"),
            all_3d_frames, allow_pickle=True)
    print(f"3D landmarks saved.")

    # Also save the 2D landmarks for hybrid analysis (fallback)
    # The side view 2D data is extremely useful for deadlift analysis
    # even without 3D, because spine angle is a sagittal plane measurement
    print("\nNote: Side-view 2D landmarks are also saved for hybrid analysis.")
    print("The rule engine will use 3D where available and 2D side-view as fallback.")

    return all_3d_frames, fps, landmarks0, landmarks1, w0, h0, w1, h1


def visualise_frame(points_3d, frame_idx=0, title="3D Skeleton"):
    """Plot a single frame's 3D skeleton."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot joints
    for idx, pos in points_3d.items():
        if isinstance(idx, str):
            # Mid-spine points
            ax.scatter(*pos, c='red', s=80, marker='^')
            ax.text(pos[0], pos[1], pos[2], idx, fontsize=8, color='red')
        else:
            ax.scatter(*pos, c='green', s=50)
            if idx in LANDMARK_NAMES:
                ax.text(pos[0], pos[1], pos[2], LANDMARK_NAMES[idx], fontsize=7)

    # Plot bones
    for (start, end) in SKELETON_CONNECTIONS:
        if start in points_3d and end in points_3d:
            xs = [points_3d[start][0], points_3d[end][0]]
            ys = [points_3d[start][1], points_3d[end][1]]
            zs = [points_3d[start][2], points_3d[end][2]]
            ax.plot(xs, ys, zs, 'b-', linewidth=2)

    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(f"{title} - Frame {frame_idx}")

    # Equal aspect ratio
    all_pts = np.array([v for v in points_3d.values() if isinstance(v, np.ndarray)])
    if len(all_pts) > 0:
        mid = all_pts.mean(axis=0)
        max_range = max((all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2, 100)
        ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
        ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
        ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()
    plt.savefig("output/skeleton_3d_sample.png", dpi=150)
    print(f"Saved skeleton image to output/skeleton_3d_sample.png")
    plt.show()


if __name__ == "__main__":
    from utils import get_latest_session, get_processed_dir
    
    session_dir = get_latest_session()
    
    all_3d_frames, fps, lm0, lm1, w0, h0, w1, h1 = reconstruct_3d(session_dir)
    
    best_idx = 0
    best_count = 0
    for i, frame in enumerate(all_3d_frames):
        if len(frame) > best_count:
            best_count = len(frame)
            best_idx = i
    
    print(f"\nBest frame: {best_idx} with {best_count} joints")
    if best_count > 0:
        visualise_frame(all_3d_frames[best_idx], frame_idx=best_idx)
    else:
        print("No valid 3D frames. Will rely on 2D side-view analysis.")