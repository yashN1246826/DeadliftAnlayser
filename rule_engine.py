import numpy as np
import os


def calculate_angle(point_a, point_b, point_c):
    """
    Calculate the angle at point_b formed by the vectors BA and BC.
    Works with both 2D and 3D points.
    Returns angle in degrees.
    """
    ba = np.array(point_a) - np.array(point_b)
    bc = np.array(point_c) - np.array(point_b)

    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return angle


def calculate_spine_angle_2d(landmarks, width, height):
    """Spine angle from vertical using side-view landmarks."""
    l_shoulder = landmarks[11]
    r_shoulder = landmarks[12]
    l_hip = landmarks[23]
    r_hip = landmarks[24]

    if l_shoulder[3] > r_shoulder[3]:
        shoulder = np.array([l_shoulder[0] * width, l_shoulder[1] * height])
        hip = np.array([l_hip[0] * width, l_hip[1] * height])
    else:
        shoulder = np.array([r_shoulder[0] * width, r_shoulder[1] * height])
        hip = np.array([r_hip[0] * width, r_hip[1] * height])

    spine_vec = shoulder - hip
    vertical = np.array([0, -1])

    cosine = np.dot(spine_vec, vertical) / (np.linalg.norm(spine_vec) + 1e-6)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return min(angle, 90)


def calculate_hip_angle_2d(landmarks, width, height):
    """
    Calculate hip hinge angle from 2D landmarks.
    Angle at hip formed by shoulder-hip-knee.
    """
    if landmarks[11][3] > landmarks[12][3]:
        shoulder = np.array([landmarks[11][0] * width, landmarks[11][1] * height])
        hip = np.array([landmarks[23][0] * width, landmarks[23][1] * height])
        knee = np.array([landmarks[25][0] * width, landmarks[25][1] * height])
    else:
        shoulder = np.array([landmarks[12][0] * width, landmarks[12][1] * height])
        hip = np.array([landmarks[24][0] * width, landmarks[24][1] * height])
        knee = np.array([landmarks[26][0] * width, landmarks[26][1] * height])

    return calculate_angle(shoulder, hip, knee)


def calculate_knee_angle_2d(landmarks, width, height):
    """Knee angle from hip-knee-ankle."""
    if landmarks[23][3] > landmarks[24][3]:
        hip = np.array([landmarks[23][0] * width, landmarks[23][1] * height])
        knee = np.array([landmarks[25][0] * width, landmarks[25][1] * height])
        ankle = np.array([landmarks[27][0] * width, landmarks[27][1] * height])
    else:
        hip = np.array([landmarks[24][0] * width, landmarks[24][1] * height])
        knee = np.array([landmarks[26][0] * width, landmarks[26][1] * height])
        ankle = np.array([landmarks[28][0] * width, landmarks[28][1] * height])

    return calculate_angle(hip, knee, ankle)


def detect_rep_phases(spine_angles, fps):
    """
    Detect individual reps from spine angle data.
    A rep = spine goes from upright → bent → upright.
    Returns list of (start_frame, end_frame) tuples.
    """
    if len(spine_angles) == 0:
        return []

    # Smooth the angles
    kernel_size = int(fps * 0.3)  # 0.3 second window
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size < 3:
        kernel_size = 3

    kernel = np.ones(kernel_size) / kernel_size
    smoothed = np.convolve(spine_angles, kernel, mode='same')

    # Find peaks (maximum forward lean = bottom of deadlift)
    threshold = 30  # degrees - minimum forward lean to count as a rep
    in_rep = False
    reps = []
    rep_start = 0

    for i in range(len(smoothed)):
        if not in_rep and smoothed[i] > threshold:
            in_rep = True
            # Walk back to find when the lean started
            rep_start = i
            for j in range(i, max(0, i - int(fps * 2)), -1):
                if smoothed[j] < 15:
                    rep_start = j
                    break

        elif in_rep and smoothed[i] < 15:
            in_rep = False
            reps.append((rep_start, i))

    # Handle case where recording ends mid-rep
    if in_rep:
        reps.append((rep_start, len(smoothed) - 1))

    return reps


def analyse_rep(spine_angles, hip_angles, knee_angles, start, end, fps):
    """
    Analyse a single rep and return faults + score.
    """
    rep_spine = spine_angles[start:end]
    rep_hip = hip_angles[start:end]
    rep_knee = knee_angles[start:end]

    faults = []
    score = 100

    # === FAULT 1: BACK ROUNDING ===
    if len(rep_spine) > 10:
        peak_idx = np.argmax(rep_spine)
        ascent_spine = rep_spine[peak_idx:]

        if len(ascent_spine) > 5:
            spine_variance = np.std(ascent_spine)
            max_spine = np.max(rep_spine)

            if max_spine > 60:
                faults.append({
                    'type': 'back_rounding',
                    'severity': 'high',
                    'message': f'Excessive forward lean detected ({max_spine:.0f}°). Keep chest up.',
                    'suggestion': 'Brace your core before initiating the pull. Think "chest up, lats tight."'
                })
                score -= 25
            elif max_spine > 45:
                faults.append({
                    'type': 'back_rounding',
                    'severity': 'medium',
                    'message': f'Moderate forward lean ({max_spine:.0f}°). Monitor your back position.',
                    'suggestion': 'Engage your lats by pulling the bar into your body.'
                })
                score -= 15

    # === FAULT 2: HIP RISE (hips shooting up before shoulders) ===
    if len(rep_hip) > 10:
        peak_idx = np.argmax(rep_spine)

        if peak_idx < len(rep_hip) - 5:
            ascent_hip = rep_hip[peak_idx:]
            ascent_spine_phase = rep_spine[peak_idx:]

            if len(ascent_hip) > 5 and len(ascent_spine_phase) > 5:
                hip_change_rate = (ascent_hip[-1] - ascent_hip[0]) / len(ascent_hip)
                spine_change_rate = (ascent_spine_phase[0] - ascent_spine_phase[-1]) / len(ascent_spine_phase)

                if abs(hip_change_rate) > 0 and abs(spine_change_rate) > 0:
                    ratio = abs(hip_change_rate / spine_change_rate)
                    if ratio > 2.5:
                        faults.append({
                            'type': 'hip_rise',
                            'severity': 'high',
                            'message': 'Hips rising faster than shoulders. This shifts load to lower back.',
                            'suggestion': 'Drive through your legs first. Push the floor away.'
                        })
                        score -= 20
                    elif ratio > 1.8:
                        faults.append({
                            'type': 'hip_rise',
                            'severity': 'medium',
                            'message': 'Slight early hip rise detected.',
                            'suggestion': 'Focus on keeping your chest and hips rising at the same rate.'
                        })
                        score -= 10

    # === FAULT 3: KNEE ANGLE ===
    if len(rep_knee) > 5:
        min_knee = np.min(rep_knee)
        if min_knee < 90:
            faults.append({
                'type': 'knee_position',
                'severity': 'medium',
                'message': f'Knees bending too much ({min_knee:.0f}°). This is a deadlift, not a squat.',
                'suggestion': 'Start with hips higher. Slight knee bend only.'
            })
            score -= 10

    # Bonus for good form
    if len(faults) == 0:
        faults.append({
            'type': 'good_form',
            'severity': 'none',
            'message': 'Good rep! Form looks solid.',
            'suggestion': 'Keep it up. Maintain this technique as weight increases.'
        })

    score = max(score, 0)

    return {
        'score': score,
        'faults': faults,
        'max_spine_angle': float(np.max(rep_spine)) if len(rep_spine) > 0 else 0,
        'min_hip_angle': float(np.min(rep_hip)) if len(rep_hip) > 0 else 0,
        'min_knee_angle': float(np.min(rep_knee)) if len(rep_knee) > 0 else 0,
        'start_frame': start,
        'end_frame': end,
    }


def analyse_recording(session_dir):
    """
    Full analysis pipeline: load landmarks, compute angles, detect reps, score.
    Uses 2D side-view landmarks (most reliable for deadlift sagittal analysis).
    """
    processed_dir = os.path.join(session_dir, "processed")

    # Determine which camera is the side view
    side_cam = "cam0"

    data = np.load(os.path.join(processed_dir, f"{side_cam}_landmarks.npz"))
    landmarks = data['landmarks']
    fps = float(data['fps'])
    width = float(data['width'])
    height = float(data['height'])

    print("=" * 50)
    print("DEADLIFT FORM ANALYSIS")
    print("=" * 50)
    print(f"Analysing {len(landmarks)} frames from {side_cam} (side view)")
    print(f"FPS: {fps}")

    # Calculate angles for every frame
    spine_angles = []
    hip_angles = []
    knee_angles = []

    for i in range(len(landmarks)):
        frame_lm = landmarks[i]

        # Check if we have pose data
        if np.sum(frame_lm[:, 3]) < 3:  # Less than 3 visible landmarks
            spine_angles.append(0)
            hip_angles.append(180)
            knee_angles.append(180)
            continue

        spine_angles.append(calculate_spine_angle_2d(frame_lm, width, height))
        hip_angles.append(calculate_hip_angle_2d(frame_lm, width, height))
        knee_angles.append(calculate_knee_angle_2d(frame_lm, width, height))

    spine_angles = np.array(spine_angles)
    hip_angles = np.array(hip_angles)
    knee_angles = np.array(knee_angles)

    # Detect reps
    reps = detect_rep_phases(spine_angles, fps)
    print(f"\nDetected {len(reps)} reps")

    if len(reps) == 0:
        print("No reps detected! Make sure the recording contains deadlift movements.")
        print(f"Max spine angle seen: {np.max(spine_angles):.1f}° (needs >30° to detect a rep)")

        # Save angle data for debugging
        np.savez(os.path.join(processed_dir, "angle_data.npz"),
                 spine=spine_angles, hip=hip_angles, knee=knee_angles, fps=fps)
        print("Angle data saved for debugging.")
        return None

    # Analyse each rep
    results = {
        'reps': [],
        'overall_score': 0,
        'total_reps': len(reps),
        'fps': fps,
    }

    total_score = 0

    for rep_num, (start, end) in enumerate(reps, 1):
        rep_result = analyse_rep(spine_angles, hip_angles, knee_angles, start, end, fps)
        rep_result['rep_number'] = rep_num
        results['reps'].append(rep_result)
        total_score += rep_result['score']

        # Print rep summary
        print(f"\n--- Rep {rep_num} (frames {start}-{end}) ---")
        print(f"  Score: {rep_result['score']}%")
        print(f"  Max spine angle: {rep_result['max_spine_angle']:.1f}°")
        print(f"  Min hip angle: {rep_result['min_hip_angle']:.1f}°")
        print(f"  Min knee angle: {rep_result['min_knee_angle']:.1f}°")
        for fault in rep_result['faults']:
            icon = "✓" if fault['severity'] == 'none' else "⚠" if fault['severity'] == 'medium' else "✗"
            print(f"  {icon} {fault['message']}")
            print(f"    → {fault['suggestion']}")

    results['overall_score'] = total_score / len(reps)

    print(f"\n{'=' * 50}")
    print(f"OVERALL SET SCORE: {results['overall_score']:.0f}%")
    print(f"{'=' * 50}")

    # Save results
    np.save(os.path.join(processed_dir, "analysis_results.npy"),
            results, allow_pickle=True)
    np.savez(os.path.join(processed_dir, "angle_data.npz"),
             spine=spine_angles, hip=hip_angles, knee=knee_angles, fps=fps)

    print(f"\nResults saved to {processed_dir}/")

    return results


if __name__ == "__main__":
    from utils import get_latest_session

    session_dir = get_latest_session()
    results = analyse_recording(session_dir)