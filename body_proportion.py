import cv2
import mediapipe as mp
import numpy as np
import argparse
import sys

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# ══════════════════════════════════════════════════════════════
#  LANDMARK GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════

def seg_dist(lm, a, b, w, h, min_vis=0.35):
    """Pixel distance between landmarks a and b. None if low visibility."""
    la, lb = lm[a], lm[b]
    if la[3] < min_vis or lb[3] < min_vis:
        return None
    return np.sqrt(((la[0] - lb[0]) * w) ** 2 + ((la[1] - lb[1]) * h) ** 2)


def hip_angle(lm, w, h):
    """Hip angle in degrees — used to detect standing frame (≈180°)."""
    si = 11 if lm[11][3] > lm[12][3] else 12
    hi = 23 if lm[23][3] > lm[24][3] else 24
    ki = 25 if lm[25][3] > lm[26][3] else 26
    if min(lm[si][3], lm[hi][3], lm[ki][3]) < 0.3:
        return 0
    s = np.array([lm[si][0] * w, lm[si][1] * h])
    h_ = np.array([lm[hi][0] * w, lm[hi][1] * h])
    k = np.array([lm[ki][0] * w, lm[ki][1] * h])
    ba, bc = s - h_, k - h_
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def is_standing(lm, w, h, threshold=155):
    return hip_angle(lm, w, h) > threshold


def extract_segments(lm, w, h):
    """
    Extract all body segment lengths from one landmark frame.
    Returns dict of segment lengths in pixels, or None values if occluded.
    Averages left and right sides where both are visible.
    """
    def avg_sides(a_left, b_left, a_right, b_right):
        l = seg_dist(lm, a_left, b_left, w, h)
        r = seg_dist(lm, a_right, b_right, w, h)
        vals = [v for v in [l, r] if v is not None]
        return float(np.mean(vals)) if vals else None

    return {
        "femur": avg_sides(23, 25, 24, 26),        # hip → knee
        "tibia": avg_sides(25, 27, 26, 28),        # knee → ankle
        "torso": avg_sides(11, 23, 12, 24),        # shoulder → hip
        "upper_arm": avg_sides(11, 13, 12, 14),    # shoulder → elbow
        "forearm": avg_sides(13, 15, 14, 16),      # elbow → wrist
    }


def aggregate_segments(frames_data):
    """
    Median of each segment across multiple standing frames.
    Returns dict with medians, or None for missing segments.
    """
    keys = ["femur", "tibia", "torso", "upper_arm", "forearm"]
    result = {}
    for k in keys:
        vals = [f[k] for f in frames_data if f.get(k) is not None]
        result[k] = float(np.median(vals)) if len(vals) >= 3 else None
    return result


# ══════════════════════════════════════════════════════════════
#  RATIO COMPUTATION AND CLASSIFICATION
# ══════════════════════════════════════════════════════════════

def compute_ratios(segments):
    """
    Compute proportion ratios from segment lengths.

    FTR  (femur / torso)     — primary driver of required forward lean
    ATR  (arm reach / torso) — affects how far down the bar hangs
    LLR  (tibia / femur)     — lower/upper leg balance
    """
    f = segments.get("femur")
    t = segments.get("tibia")
    s = segments.get("torso")
    ua = segments.get("upper_arm")
    fa = segments.get("forearm")

    ftr = round(f / s, 3) if (f and s and s > 0) else None
    llr = round(t / f, 3) if (t and f and f > 0) else None
    arm = (ua + fa) if (ua and fa) else None
    atr = round(arm / s, 3) if (arm and s and s > 0) else None

    return {"ftr": ftr, "llr": llr, "atr": atr, "arm_px": arm}


def classify_proportions(ratios):
    """
    Determine body proportion category and droop ratio threshold scale.
    """
    ftr = ratios.get("ftr")
    atr = ratios.get("atr")

    if ftr is None:
        return "unknown", 1.0, "Could not measure — standard thresholds applied"

    if ftr < 0.70:
        base_scale = 1.0
        ftr_label = "short femur"
    elif ftr < 0.85:
        base_scale = 1.0
        ftr_label = "standard proportions"
    elif ftr < 1.00:
        base_scale = 1.2
        ftr_label = "long femur"
    else:
        base_scale = 1.4
        ftr_label = "very long femur"

    arm_note = ""
    if atr is not None:
        if atr > 1.4:
            base_scale = min(base_scale + 0.10, 1.5)
            arm_note = ", long arms"
        elif atr < 0.9:
            arm_note = ", short arms"

    scale = round(base_scale, 2)
    category = ftr_label + arm_note

    if scale == 1.0:
        desc = f"{category} (FTR={ftr:.2f}) — standard thresholds applied"
    else:
        pct = int((scale - 1.0) * 100)
        desc = f"{category} (FTR={ftr:.2f}) — rounding threshold relaxed {pct}%"

    return category, scale, desc


def build_profile(segments, ratios, category, scale, description):
    """Package everything into a JSON-serialisable profile dict."""
    return {
        "femur_px": round(segments.get("femur") or 0, 1),
        "tibia_px": round(segments.get("tibia") or 0, 1),
        "torso_px": round(segments.get("torso") or 0, 1),
        "upper_arm_px": round(segments.get("upper_arm") or 0, 1),
        "forearm_px": round(segments.get("forearm") or 0, 1),
        "ftr": ratios.get("ftr"),
        "llr": ratios.get("llr"),
        "atr": ratios.get("atr"),
        "category": category,
        "threshold_scale": scale,
        "description": description,
    }


def print_profile(profile):
    print("\n" + "=" * 55)
    print(" BODY PROPORTION PROFILE")
    print("=" * 55)
    print(f"  Femur:       {profile['femur_px']:.0f} px")
    print(f"  Tibia:       {profile['tibia_px']:.0f} px")
    print(f"  Torso:       {profile['torso_px']:.0f} px")
    print(f"  Upper arm:   {profile['upper_arm_px']:.0f} px")
    print(f"  Forearm:     {profile['forearm_px']:.0f} px")
    print(f"  FTR:         {profile['ftr']} (femur/torso)")
    print(f"  LLR:         {profile['llr']} (tibia/femur)")
    print(f"  ATR:         {profile['atr']} (arm reach/torso)")
    print(f"  Category:    {profile['category']}")
    print(f"  Scale:       ×{profile['threshold_scale']}")
    print(f"  Summary:     {profile['description']}")
    print("=" * 55)


# ══════════════════════════════════════════════════════════════
#  MODE A — LIVE CAMERA
# ══════════════════════════════════════════════════════════════

def measure_from_camera(cam_index=1, collect_seconds=4, show_preview=True):
    """
    Open side camera, collect standing frames, return proportion profile.
    User stands side-on for collect_seconds.
    Returns profile dict or None if measurement failed.
    """
    import time

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"Error: cannot open camera {cam_index}")
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    standing_frames = []
    start_time = None
    phase = "WAITING"

    print("\nBody proportion measurement — live camera")
    print("Stand side-on to the camera, upright, arms relaxed.")
    print(f"Hold still for {collect_seconds}s. Press Q to skip.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        lm_data = None
        if results.pose_landmarks:
            lm_data = np.array([
                [lm.x, lm.y, lm.z, lm.visibility]
                for lm in results.pose_landmarks.landmark
            ])

        if lm_data is not None and is_standing(lm_data, w, h):
            if phase == "WAITING":
                phase = "COLLECTING"
                start_time = time.time()
                print("Standing detected — collecting...")
            elif phase == "COLLECTING":
                segs = extract_segments(lm_data, w, h)
                standing_frames.append(segs)
                elapsed = time.time() - start_time
                if elapsed >= collect_seconds:
                    phase = "DONE"
        else:
            if phase == "COLLECTING":
                phase, standing_frames, start_time = "WAITING", [], None

        if phase == "DONE":
            break

        if show_preview:
            overlay = frame.copy()
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    overlay,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 200, 255), thickness=2)
                )

            cv2.rectangle(overlay, (0, 0), (w, 55), (30, 30, 30), -1)

            if phase == "WAITING":
                msg, col = "Stand side-on, upright", (60, 60, 255)
            else:
                elapsed = time.time() - start_time
                remain = max(0, collect_seconds - elapsed)
                msg, col = f"Measuring... {remain:.1f}s", (60, 220, 60)
                bar_w = int(elapsed / collect_seconds * (w - 40))
                cv2.rectangle(overlay, (20, h - 35), (20 + bar_w, h - 15), (60, 220, 60), -1)
                cv2.rectangle(overlay, (20, h - 35), (w - 20, h - 15), (150, 150, 150), 2)

            cv2.putText(
                overlay,
                "BODY PROPORTION MEASUREMENT",
                (15, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            cv2.putText(
                overlay,
                msg,
                (15, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                col,
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Body Proportion (Q=skip)", overlay)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            print("Skipped — standard thresholds applied.")
            pose.close()
            cap.release()
            cv2.destroyAllWindows()
            return None

    pose.close()
    cap.release()
    cv2.destroyAllWindows()

    if len(standing_frames) < 5:
        print("Not enough standing frames collected.")
        return None

    return _finalise(standing_frames)


# ══════════════════════════════════════════════════════════════
#  MODE B — EXISTING VIDEO FILE
# ══════════════════════════════════════════════════════════════

def measure_from_video(video_path, show_preview=False):
    """
    Extract body proportions from an existing cam1.mp4 recording.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: cannot open {video_path}")
        return None

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    standing_frames = []
    frame_idx = 0

    print(f"Scanning {video_path} for standing frames...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 2 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if results.pose_landmarks:
                lm_data = np.array([
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in results.pose_landmarks.landmark
                ])
                if is_standing(lm_data, w, h):
                    segs = extract_segments(lm_data, w, h)
                    standing_frames.append(segs)

        frame_idx += 1
        if frame_idx % 60 == 0 and total > 0:
            pct = frame_idx / total * 100
            print(f"  Scanning: {frame_idx}/{total} ({pct:.0f}%)", end='\r')

    pose.close()
    cap.release()

    print(f"\nFound {len(standing_frames)} standing frames")

    if len(standing_frames) < 3:
        print("Insufficient standing frames in video.")
        print("The person may have been lifting throughout — try the live camera mode.")
        return None

    return _finalise(standing_frames)


# ══════════════════════════════════════════════════════════════
#  SHARED FINALISATION
# ══════════════════════════════════════════════════════════════

def _finalise(standing_frames):
    """Aggregate frames → ratios → classify → return profile."""
    segments = aggregate_segments(standing_frames)
    ratios = compute_ratios(segments)
    cat, scale, desc = classify_proportions(ratios)
    profile = build_profile(segments, ratios, cat, scale, desc)
    print_profile(profile)
    return profile


# ══════════════════════════════════════════════════════════════
#  ADAPTIVE SCORING  (imported by app.py)
# ══════════════════════════════════════════════════════════════

def calc_rounding_score_adaptive(lm, w, h, threshold_scale=1.0):
    """
    Droop ratio rounding score with proportion-aware threshold scaling.
    """
    l_sh, r_sh = lm[11], lm[12]
    l_hp, r_hp = lm[23], lm[24]

    if min(l_sh[3], r_sh[3], l_hp[3], r_hp[3]) < 0.3:
        return 0

    mid_sh_x = (l_sh[0] + r_sh[0]) / 2 * w
    mid_sh_y = (l_sh[1] + r_sh[1]) / 2 * h
    mid_hp_x = (l_hp[0] + r_hp[0]) / 2 * w
    mid_hp_y = (l_hp[1] + r_hp[1]) / 2 * h

    dx = abs(mid_sh_x - mid_hp_x)
    dy = abs(mid_sh_y - mid_hp_y)

    if dy < 20:
        return 60

    droop = dx / dy

    if mid_sh_y > mid_hp_y:
        return min(80, 45 + droop * 15)

    caution_t = 1.2 * threshold_scale
    rounding_t = 1.8 * threshold_scale

    if droop < 1.2:
        return max(0, droop * 8)
    elif droop < caution_t:
        return max(0, droop * 8)
    elif droop < rounding_t:
        zone_w = rounding_t - caution_t
        return 10 + (droop - caution_t) / zone_w * 24
    else:
        return min(80, 34 + (droop - rounding_t) * 70)


def get_proportion_description(profile):
    if profile is None:
        return "Body proportions not measured — standard thresholds applied"
    return profile.get("description", "Unknown proportions")


# ══════════════════════════════════════════════════════════════
#  SUBPROCESS WRAPPER FOR macOS / QTHREAD SAFETY
# ══════════════════════════════════════════════════════════════

def _run_in_subprocess(cam_index, collect_seconds, result_queue):
    """Runs measure_from_camera in a subprocess (own main thread → cv2.imshow works)."""
    result = measure_from_camera(cam_index=cam_index, collect_seconds=collect_seconds)
    result_queue.put(result)


def measure_from_camera_subprocess(cam_index=1, collect_seconds=4):
    """
    Spawns a child process to run measure_from_camera so that cv2.imshow
    works on macOS regardless of which thread the caller is on.
    Returns the same dict that measure_from_camera returns, or None on failure.
    """
    import multiprocessing
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_run_in_subprocess, args=(cam_index, collect_seconds, q))
    p.start()
    p.join()
    if not q.empty():
        return q.get()
    return None


# ══════════════════════════════════════════════════════════════
#  COMMAND-LINE ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import json

    parser = argparse.ArgumentParser(description="Body Proportion Detector")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--camera",
        action="store_true",
        help="Use live side camera (index 1)"
    )
    group.add_argument(
        "--video",
        type=str,
        metavar="PATH",
        help="Analyse existing cam1.mp4 file"
    )
    parser.add_argument(
        "--cam-index",
        type=int,
        default=1,
        help="Camera index for live mode (default: 1)"
    )
    args = parser.parse_args()

    if args.camera:
        profile = measure_from_camera(cam_index=args.cam_index)
    else:
        profile = measure_from_video(args.video)

    if profile:
        print("\nJSON profile:")
        print(json.dumps(profile, indent=2))
    else:
        print("\nMeasurement failed — standard thresholds will be used.")