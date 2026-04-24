import cv2
import numpy as np
import os

# === BODY CONNECTIONS ===
# (start_landmark, end_landmark, region_name)
BONES = [
    (11, 13, "left_upper_arm"), (13, 15, "left_forearm"),
    (12, 14, "right_upper_arm"), (14, 16, "right_forearm"),
    (23, 25, "left_thigh"), (25, 27, "left_shin"),
    (24, 26, "right_thigh"), (26, 28, "right_shin"),
    (11, 23, "left_torso"), (12, 24, "right_torso"),
    (11, 12, "shoulders"), (23, 24, "hips"),
]

JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

SEGMENT_THICKNESS = {
    "left_thigh": 22, "right_thigh": 22,
    "left_shin": 18, "right_shin": 18,
    "left_upper_arm": 14, "right_upper_arm": 14,
    "left_forearm": 12, "right_forearm": 12,
    "left_torso": 10, "right_torso": 10,
    "shoulders": 10, "hips": 10,
}

COLOUR_GOOD = (80, 230, 80)
COLOUR_CAUTION = (50, 190, 255)
COLOUR_BAD = (70, 70, 255)
COLOUR_NEUTRAL = (220, 185, 60)
COLOUR_BONE = (200, 200, 200)
BG_COLOUR = (35, 30, 30)
PANEL_BG = (50, 45, 42)


def get_pt(landmarks, idx, width, height):
    lm = landmarks[idx]
    if lm[3] < 0.15:
        return None
    return np.array([lm[0] * width, lm[1] * height])


def smooth_landmarks(landmarks, window=7):
    smoothed = np.copy(landmarks)
    n = len(landmarks)
    for i in range(n):
        s = max(0, i - window // 2)
        e = min(n, i + window // 2 + 1)
        for j in range(33):
            vis = landmarks[s:e, j, 3]
            if np.max(vis) > 0.15:
                w = vis + 0.01
                smoothed[i, j, :3] = np.average(landmarks[s:e, j, :3], axis=0, weights=w)
                smoothed[i, j, 3] = np.max(vis)
    return smoothed


def calc_spine_angle(lm, w, h):
    """
    Calculate forward lean angle from vertical.
    Returns 0 when standing upright, increases as you lean forward.
    """
    l_sh, r_sh = lm[11], lm[12]
    l_hp, r_hp = lm[23], lm[24]

    if (l_sh[3] + r_sh[3]) / 2 < 0.3 or (l_hp[3] + r_hp[3]) / 2 < 0.3:
        return 0

    mid_sh = np.array([(l_sh[0] + r_sh[0]) / 2 * w,
                       (l_sh[1] + r_sh[1]) / 2 * h])
    mid_hp = np.array([(l_hp[0] + r_hp[0]) / 2 * w,
                       (l_hp[1] + r_hp[1]) / 2 * h])

    spine_vec = mid_sh - mid_hp
    vertical = np.array([0, -1])

    cos_a = np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    return min(angle, 90)


def calc_knee_angle(lm, w, h):
    """Knee angle: hip-knee-ankle. 180 = straight, lower = more bent."""
    if lm[23][3] > lm[24][3]:
        hip = np.array([lm[23][0] * w, lm[23][1] * h])
        knee = np.array([lm[25][0] * w, lm[25][1] * h])
        ankle = np.array([lm[27][0] * w, lm[27][1] * h])
    else:
        hip = np.array([lm[24][0] * w, lm[24][1] * h])
        knee = np.array([lm[26][0] * w, lm[26][1] * h])
        ankle = np.array([lm[28][0] * w, lm[28][1] * h])

    ba = hip - knee
    bc = ankle - knee
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def calc_hip_angle(lm, w, h):
    """Hip angle: shoulder-hip-knee. Smaller = more bent forward."""
    if lm[11][3] > lm[12][3]:
        sh = np.array([lm[11][0] * w, lm[11][1] * h])
        hp = np.array([lm[23][0] * w, lm[23][1] * h])
        kn = np.array([lm[25][0] * w, lm[25][1] * h])
    else:
        sh = np.array([lm[12][0] * w, lm[12][1] * h])
        hp = np.array([lm[24][0] * w, lm[24][1] * h])
        kn = np.array([lm[26][0] * w, lm[26][1] * h])

    ba = sh - hp
    bc = kn - hp
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def detect_rounding(lm, w, h):
    """
    Detect actual back rounding vs normal forward lean.

    Returns a rounding score 0-100:
      0-20 = neutral spine
      20-40 = slight rounding
      40+ = significant rounding
    """
    l_sh, r_sh = lm[11], lm[12]
    l_hp, r_hp = lm[23], lm[24]
    l_kn, r_kn = lm[25], lm[26]

    if min(l_sh[3], r_sh[3], l_hp[3], r_hp[3]) < 0.3:
        return 0

    mid_sh = np.array([(l_sh[0] + r_sh[0]) / 2 * w,
                       (l_sh[1] + r_sh[1]) / 2 * h])
    mid_hp = np.array([(l_hp[0] + r_hp[0]) / 2 * w,
                       (l_hp[1] + r_hp[1]) / 2 * h])

    spine_vec = mid_sh - mid_hp
    spine_len = np.linalg.norm(spine_vec)
    if spine_len < 10:
        return 0

    dx = abs(mid_sh[0] - mid_hp[0])
    dy = abs(mid_sh[1] - mid_hp[1])

    if dy < 20:
        return 60

    droop = dx / dy
    shoulder_below_hip = mid_sh[1] > mid_hp[1]

    if shoulder_below_hip:
        return min(80, 45 + droop * 15)

    if droop < 1.2:
        return max(0, droop * 8)
    elif droop < 1.8:
        return 10 + (droop - 1.2) * 40
    else:
        return min(80, 34 + (droop - 1.8) * 70)


def calc_lateral_tilt(lm, w, h):
    """Lateral tilt — only flag significant asymmetry."""
    l_sh, r_sh = lm[11], lm[12]

    if min(l_sh[3], r_sh[3]) < 0.4:
        return 0

    l_sh_y = l_sh[1] * h
    r_sh_y = r_sh[1] * h

    mid_hp_y = ((lm[23][1] + lm[24][1]) / 2) * h
    mid_sh_y = (l_sh_y + r_sh_y) / 2
    torso_height = abs(mid_hp_y - mid_sh_y)

    if torso_height < 30:
        return 0

    tilt_pct = (l_sh_y - r_sh_y) / torso_height * 100

    if abs(tilt_pct) < 5:
        return 0

    return tilt_pct


def get_region_colour(region, spine_angle, knee_angle, rounding_score=0):
    """
    Colour body regions based on actual faults, not just lean angle.
    """
    if "torso" in region or region == "shoulders" or region == "hips":
        if rounding_score > 40:
            return COLOUR_BAD
        elif rounding_score > 20:
            return COLOUR_CAUTION
        else:
            return COLOUR_GOOD

    if "thigh" in region or "shin" in region:
        if knee_angle < 80:
            return COLOUR_CAUTION
        return COLOUR_NEUTRAL

    return COLOUR_NEUTRAL


def draw_torso_filled(canvas, pts_screen, colour, alpha=0.25):
    """Draw filled torso polygon with transparency."""
    if len(pts_screen) < 4:
        return
    polygon = np.array(pts_screen, dtype=np.int32)
    overlay = canvas.copy()
    cv2.fillPoly(overlay, [polygon], colour)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)
    cv2.polylines(canvas, [polygon], True, colour, 2, cv2.LINE_AA)


def draw_head(canvas, mid_shoulder, shoulder_width, colour):
    """Draw a head with neck."""
    head_r = max(int(shoulder_width * 0.35), 16)
    head_center = (int(mid_shoulder[0]), int(mid_shoulder[1] - head_r * 1.8))

    neck_top = (head_center[0], head_center[1] + head_r)
    neck_bot = (int(mid_shoulder[0]), int(mid_shoulder[1]))
    cv2.line(canvas, neck_top, neck_bot, colour, max(head_r // 2, 8), cv2.LINE_AA)

    cv2.circle(canvas, head_center, head_r, colour, -1, cv2.LINE_AA)
    cv2.circle(canvas, head_center, head_r + 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(canvas, head_center, head_r - 3,
               tuple(min(c + 25, 255) for c in colour), 2, cv2.LINE_AA)


def draw_hand(canvas, wrist_pt, colour):
    """Draw a small hand circle at wrist."""
    cv2.circle(canvas, tuple(wrist_pt.astype(int)), 8, colour, -1, cv2.LINE_AA)
    cv2.circle(canvas, tuple(wrist_pt.astype(int)), 9, (200, 200, 200), 1, cv2.LINE_AA)


def draw_capsule(canvas, pt1, pt2, colour, thickness):
    """Draw a body segment as a clean thick rounded line."""
    p1 = tuple(pt1.astype(int))
    p2 = tuple(pt2.astype(int))
    cv2.line(canvas, p1, p2, colour, thickness, cv2.LINE_AA)
    cv2.circle(canvas, p1, thickness // 2, colour, -1, cv2.LINE_AA)
    cv2.circle(canvas, p2, thickness // 2, colour, -1, cv2.LINE_AA)


def render_body_view(canvas, landmarks, width, height, spine_angle, knee_angle,
                     hip_angle, x_offset, y_offset, scale, view_label,
                     rounding_score=0):
    """
    Render one view (side or front) of the body on the canvas.
    x_offset, y_offset: where to place this view on the canvas
    scale: scaling factor
    """

    def to_screen(pt):
        return np.array([
            int(pt[0] * scale + x_offset),
            int(pt[1] * scale + y_offset)
        ])

    joint_pts = {}
    for idx in JOINTS:
        pt = get_pt(landmarks, idx, width, height)
        if pt is not None:
            joint_pts[idx] = to_screen(pt)

    # TORSO
    torso_corners = [11, 12, 24, 23]
    torso_screen = [joint_pts[i] for i in torso_corners if i in joint_pts]
    if len(torso_screen) == 4:
        torso_col = get_region_colour("left_torso", spine_angle, knee_angle, rounding_score)
        draw_torso_filled(canvas, torso_screen, torso_col, alpha=0.2)

    # BONES
    for (start_idx, end_idx, region) in BONES:
        if start_idx not in joint_pts or end_idx not in joint_pts:
            continue

        colour = get_region_colour(region, spine_angle, knee_angle, rounding_score)
        thickness = SEGMENT_THICKNESS.get(region, 10)
        thickness = max(int(thickness * scale * 0.6), 4)

        draw_capsule(canvas, joint_pts[start_idx], joint_pts[end_idx],
                     colour, thickness)

    # HEAD
    if 11 in joint_pts and 12 in joint_pts:
        mid_sh = (joint_pts[11] + joint_pts[12]) / 2
        sh_width = np.linalg.norm(joint_pts[11] - joint_pts[12])
        head_col = get_region_colour("left_torso", spine_angle, knee_angle, rounding_score)
        draw_head(canvas, mid_sh, sh_width, head_col)

    # JOINTS
    for idx in JOINTS:
        if idx not in joint_pts:
            continue
        pt = joint_pts[idx]

        if idx in [11, 12, 23, 24]:
            col = get_region_colour("left_torso", spine_angle, knee_angle, rounding_score)
            r = int(11 * scale * 0.5)
        elif idx in [25, 26]:
            col = get_region_colour("left_thigh", spine_angle, knee_angle, rounding_score)
            r = int(10 * scale * 0.5)
        else:
            col = COLOUR_NEUTRAL
            r = int(8 * scale * 0.5)

        r = max(r, 5)
        cv2.circle(canvas, tuple(pt.astype(int)), r, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pt.astype(int)), r + 1, (220, 220, 220), 1, cv2.LINE_AA)

    # HANDS
    for idx in [15, 16]:
        if idx in joint_pts:
            draw_hand(canvas, joint_pts[idx], COLOUR_NEUTRAL)

    # SPINE LINE
    if 11 in joint_pts and 12 in joint_pts and 23 in joint_pts and 24 in joint_pts:
        mid_sh = (joint_pts[11] + joint_pts[12]) / 2
        mid_hp = (joint_pts[23] + joint_pts[24]) / 2
        spine_col = get_region_colour("left_torso", spine_angle, knee_angle, rounding_score)

        if rounding_score > 30:
            num_points = 20
            curve_amount = rounding_score / 80.0 * 35

            spine_dir = mid_hp - mid_sh
            spine_len = np.linalg.norm(spine_dir)
            if spine_len > 0:
                perp = np.array([-spine_dir[1], spine_dir[0]]) / spine_len

                curve_points = []
                for i in range(num_points + 1):
                    t = i / num_points
                    base_pt = mid_sh * (1 - t) + mid_hp * t
                    bow = curve_amount * 4 * t * (1 - t)
                    curved_pt = base_pt + perp * bow
                    curve_points.append(curved_pt.astype(int))

                for i in range(len(curve_points) - 1):
                    cv2.line(canvas,
                             tuple(curve_points[i]),
                             tuple(curve_points[i + 1]),
                             spine_col, 4, cv2.LINE_AA)
        else:
            length = np.linalg.norm(mid_sh - mid_hp)
            if length > 0:
                d = (mid_hp - mid_sh) / length
                pos = 0
                while pos < length:
                    p1 = mid_sh + d * pos
                    p2 = mid_sh + d * min(pos + 10, length)
                    cv2.line(canvas, tuple(p1.astype(int)), tuple(p2.astype(int)),
                             spine_col, 3, cv2.LINE_AA)
                    pos += 16

    # FAULT LABELS
    if rounding_score > 30 and 23 in joint_pts and 11 in joint_pts:
        mid_spine = ((joint_pts[11] + joint_pts[23]) / 2).astype(int)
        for torso_idx in [11, 12, 23, 24]:
            if torso_idx in joint_pts:
                pt = tuple(joint_pts[torso_idx].astype(int))
                cv2.circle(canvas, pt, 20, COLOUR_BAD, 2, cv2.LINE_AA)

        label_x = int(mid_spine[0] + 40 * scale)
        label_y = int(mid_spine[1])
        cv2.arrowedLine(canvas, (label_x, label_y),
                        (int(mid_spine[0]) + 5, int(mid_spine[1])),
                        COLOUR_BAD, 2, cv2.LINE_AA, tipLength=0.3)
        cv2.putText(canvas, "BACK", (label_x + 5, label_y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOUR_BAD, 1, cv2.LINE_AA)
        cv2.putText(canvas, "ROUNDING", (label_x + 5, label_y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOUR_BAD, 1, cv2.LINE_AA)
    elif rounding_score > 15 and 23 in joint_pts and 11 in joint_pts:
        mid_spine = ((joint_pts[11] + joint_pts[23]) / 2).astype(int)
        label_x = int(mid_spine[0] + 35 * scale)
        cv2.putText(canvas, "Watch back", (label_x, int(mid_spine[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOUR_CAUTION, 1, cv2.LINE_AA)

    # FRONT VIEW TILT WARNING
    if "FRONT" in view_label:
        lateral = calc_lateral_tilt(landmarks, width, height)
        if abs(lateral) > 20 and 11 in joint_pts and 12 in joint_pts:
            mid_sh = (joint_pts[11] + joint_pts[12]) / 2
            direction = "LEFT" if lateral > 0 else "RIGHT"
            head_y = int(mid_sh[1]) - 80
            if head_y < 130:
                head_y = 130
            cv2.putText(canvas, f"LEANING {direction}",
                        (int(mid_sh[0]) - 55, head_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOUR_CAUTION, 2, cv2.LINE_AA)

    # VIEW LABEL
    if "SIDE" in view_label:
        lx, ly = 15, 118
    elif "FRONT" in view_label:
        lx, ly = 525, 118
    else:
        lx, ly = int(x_offset) + 10, 118

    cv2.rectangle(canvas, (lx - 3, ly - 16), (lx + 110, ly + 6),
                  (55, 50, 48), -1)
    cv2.putText(canvas, view_label, (lx, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)


def draw_dashboard(canvas, frame_w, frame_h, spine_angle, knee_angle,
                   hip_angle, frame_idx, num_frames, fps,
                   lateral_tilt=0, rounding_score=0):
    """Draw the HUD/dashboard elements."""

    # TOP BAR
    cv2.rectangle(canvas, (0, 0), (frame_w, 55), (45, 40, 38), -1)
    cv2.putText(canvas, "DEADLIFT FORM ANALYSIS",
                (frame_w // 2 - 160, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.line(canvas, (0, 55), (frame_w, 55), (80, 75, 70), 2)

    # STATUS
    if rounding_score > 30:
        status, s_col = "ROUNDING DETECTED", COLOUR_BAD
    elif rounding_score > 15:
        status, s_col = "CAUTION", COLOUR_CAUTION
    elif spine_angle > 15:
        status, s_col = "GOOD FORM", COLOUR_GOOD
    else:
        status, s_col = "STANDING", (150, 150, 150)

    tw, _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    badge_x = frame_w // 2 - tw[0] // 2 - 12
    cv2.rectangle(canvas, (badge_x, 62), (badge_x + tw[0] + 24, 90),
                  s_col, -1, cv2.LINE_AA)
    cv2.putText(canvas, status, (badge_x + 12, 84),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # BOTTOM PANEL
    panel_y = frame_h - 100
    cv2.rectangle(canvas, (0, panel_y), (frame_w, frame_h), (45, 40, 38), -1)
    cv2.line(canvas, (0, panel_y), (frame_w, panel_y), (80, 75, 70), 2)

    gauge_y = panel_y + 25

    # Rounding
    cv2.putText(canvas, "ROUNDING", (20, gauge_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
    r_col = COLOUR_BAD if rounding_score > 40 else COLOUR_CAUTION if rounding_score > 20 else COLOUR_GOOD
    cv2.putText(canvas, f"{rounding_score:.0f}", (20, gauge_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, r_col, 2, cv2.LINE_AA)

    bar_x = 20
    bar_w = 120
    bar_y = gauge_y + 35
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8),
                  (60, 55, 52), -1)
    fill_w = min(int(rounding_score / 80 * bar_w), bar_w)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + 8),
                  r_col, -1)

    # Hip
    cv2.putText(canvas, "HIP", (180, gauge_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{hip_angle:.0f} deg", (180, gauge_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOUR_NEUTRAL, 2, cv2.LINE_AA)

    # Knee
    cv2.putText(canvas, "KNEE", (320, gauge_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
    knee_col = COLOUR_CAUTION if knee_angle < 80 else COLOUR_NEUTRAL
    cv2.putText(canvas, f"{knee_angle:.0f} deg", (320, gauge_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, knee_col, 2, cv2.LINE_AA)

    # Tilt
    cv2.putText(canvas, "TILT", (460, gauge_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (140, 140, 140), 1, cv2.LINE_AA)
    tilt_col = COLOUR_CAUTION if abs(lateral_tilt) > 15 else COLOUR_GOOD
    tilt_dir = "L" if lateral_tilt > 0 else "R" if lateral_tilt < 0 else ""
    cv2.putText(canvas, f"{abs(lateral_tilt):.0f} {tilt_dir}", (460, gauge_y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, tilt_col, 2, cv2.LINE_AA)

    # Time
    time_sec = frame_idx / fps
    cv2.putText(canvas, f"Time: {time_sec:.1f}s",
                (frame_w - 150, gauge_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Frame: {frame_idx}/{num_frames}",
                (frame_w - 150, gauge_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)

    # Progress
    prog_y = frame_h - 8
    prog_w = int((frame_idx / max(num_frames - 1, 1)) * frame_w)
    cv2.rectangle(canvas, (0, prog_y), (frame_w, frame_h), (60, 55, 52), -1)
    cv2.rectangle(canvas, (0, prog_y), (prog_w, frame_h), COLOUR_GOOD, -1)


def generate_skeleton_animation(session_dir):
    """Generate the full dual-view skeleton animation."""
    processed_dir = os.path.join(session_dir, "processed")

    # cam0 = C920 Logitech = FRONT VIEW
    # cam1 = iPhone = SIDE VIEW
    data_side = np.load(os.path.join(processed_dir, "cam1_landmarks.npz"))
    data_front = np.load(os.path.join(processed_dir, "cam0_landmarks.npz"))

    lm_side = data_side['landmarks']
    lm_front = data_front['landmarks']
    input_fps = float(data_side['fps'])
    output_fps = 12.0
    w_side, h_side = float(data_side['width']), float(data_side['height'])
    w_front, h_front = float(data_front['width']), float(data_front['height'])

    num_frames = min(len(lm_side), len(lm_front))
    lm_side = lm_side[:num_frames]
    lm_front = lm_front[:num_frames]

    print(f"Generating dual-view animation: {num_frames} frames")
    print(f"Input FPS: {input_fps}")
    print(f"Output FPS: {output_fps}")
    print("Smoothing landmarks...")
    lm_side = smooth_landmarks(lm_side, window=7)
    lm_front = smooth_landmarks(lm_front, window=7)

    spine_angles, knee_angles, hip_angles, rounding_scores = [], [], [], []
    for i in range(num_frames):
        spine_angles.append(calc_spine_angle(lm_side[i], w_side, h_side))
        knee_angles.append(calc_knee_angle(lm_side[i], w_side, h_side))
        hip_angles.append(calc_hip_angle(lm_side[i], w_side, h_side))
        rounding_scores.append(detect_rounding(lm_side[i], w_side, h_side))

    rounding_scores = np.array(rounding_scores, dtype=np.float32)
    kernel = np.ones(7, dtype=np.float32) / 7.0
    rounding_scores = np.convolve(rounding_scores, kernel, mode='same')

    lateral_tilts = []
    for i in range(num_frames):
        lateral_tilts.append(calc_lateral_tilt(lm_front[i], w_front, h_front))

    def get_bounds(landmarks, w, h):
        all_x, all_y = [], []
        for frame_lm in landmarks:
            for idx in JOINTS:
                pt = get_pt(frame_lm, idx, w, h)
                if pt is not None:
                    all_x.append(pt[0])
                    all_y.append(pt[1])

            l_sh = get_pt(frame_lm, 11, w, h)
            r_sh = get_pt(frame_lm, 12, w, h)
            if l_sh is not None and r_sh is not None:
                mid = (l_sh + r_sh) / 2
                sw = np.linalg.norm(l_sh - r_sh)
                all_y.append(mid[1] - sw * 0.9)

        margin = 80
        return (min(all_x) - margin, max(all_x) + margin,
                min(all_y) - margin, max(all_y) + margin)

    sx_min, sx_max, sy_min, sy_max = get_bounds(lm_side, w_side, h_side)
    fx_min, fx_max, fy_min, fy_max = get_bounds(lm_front, w_front, h_front)

    panel_w = 500
    panel_h = 700
    frame_w = panel_w * 2 + 20
    frame_h = panel_h + 160

    s_x_range = sx_max - sx_min
    s_y_range = sy_max - sy_min
    s_scale = min(panel_w / s_x_range, (panel_h - 40) / s_y_range) * 0.85

    f_x_range = fx_max - fx_min
    f_y_range = fy_max - fy_min
    f_scale = min(panel_w / f_x_range, (panel_h - 40) / f_y_range) * 0.85

    s_x_off = (panel_w - s_x_range * s_scale) / 2 - sx_min * s_scale
    s_y_off = 100 + (panel_h - s_y_range * s_scale) / 2 - sy_min * s_scale

    f_x_off = panel_w + 20 + (panel_w - f_x_range * f_scale) / 2 - fx_min * f_scale
    f_y_off = 100 + (panel_h - f_y_range * f_scale) / 2 - fy_min * f_scale

    output_path = os.path.join(processed_dir, "skeleton_animation.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (frame_w, frame_h))

    print(f"Output resolution: {frame_w}x{frame_h}")
    print("Rendering frames...")

    for frame_idx in range(num_frames):
        canvas = np.full((frame_h, frame_w, 3), BG_COLOUR, dtype=np.uint8)

        cv2.rectangle(canvas, (5, 100), (panel_w - 5, 100 + panel_h - 10),
                      PANEL_BG, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (panel_w + 15, 100),
                      (frame_w - 5, 100 + panel_h - 10),
                      PANEL_BG, -1, cv2.LINE_AA)

        cv2.rectangle(canvas, (5, 100), (panel_w - 5, 100 + panel_h - 10),
                      (70, 65, 60), 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (panel_w + 15, 100),
                      (frame_w - 5, 100 + panel_h - 10),
                      (70, 65, 60), 1, cv2.LINE_AA)

        spine_a = spine_angles[frame_idx]
        knee_a = knee_angles[frame_idx]
        hip_a = hip_angles[frame_idx]
        rounding = float(rounding_scores[frame_idx])
        lateral_tilt = lateral_tilts[frame_idx]

        # LEFT PANEL: SIDE VIEW
        render_body_view(canvas, lm_side[frame_idx], w_side, h_side,
                         spine_a, knee_a, hip_a,
                         s_x_off, s_y_off, s_scale,
                         "SIDE VIEW", rounding)

        # RIGHT PANEL: FRONT VIEW
        render_body_view(canvas, lm_front[frame_idx], w_front, h_front,
                         spine_a, knee_a, hip_a,
                         f_x_off, f_y_off, f_scale,
                         "FRONT VIEW", rounding)

        draw_dashboard(canvas, frame_w, frame_h, spine_a, knee_a, hip_a,
                       frame_idx, num_frames, input_fps,
                       lateral_tilt=lateral_tilt, rounding_score=rounding)

        out.write(canvas)

        if frame_idx % 30 == 0:
            pct = 100 * frame_idx / num_frames
            print(f"  Rendering: {frame_idx}/{num_frames} ({pct:.0f}%)", end='\r')

    out.release()
    print(f"\nAnimation saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    from utils import get_latest_session

    session_dir = get_latest_session()
    output = generate_skeleton_animation(session_dir)
    if output:
        print(f"\nDone! Open: {output}")