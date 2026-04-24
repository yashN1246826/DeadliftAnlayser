# 🏋️ DeadliftAnalyser

> A real-time dual-camera AI system for deadlift form analysis, fault detection, and longitudinal progress tracking.

**Author:** Yash Kumar (N1246826)  
**Degree:** BSc Computer Science with Artificial Intelligence — Nottingham Trent University  
**Module:** COMP30151/2 — Final Year Project (2025/26)  
**Supervisor:** Steven Lambert

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Hardware Requirements](#hardware-requirements)
- [Software Requirements](#software-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Technical Design](#technical-design)
  - [Droop Ratio](#droop-ratio--back-rounding-detection)
  - [Hip-Angle-Conditional Thresholds](#hip-angle-conditional-thresholds)
  - [Femur-to-Torso Ratio (FTR)](#femur-to-torso-ratio-ftr)
  - [Stereo Calibration](#stereo-calibration)
  - [Scoring Algorithm](#scoring-algorithm)
- [Evaluation Results](#evaluation-results)
- [Pipeline Details](#pipeline-details)
- [Known Issues & Limitations](#known-issues--limitations)
- [Future Work](#future-work)
- [References](#references)

---

## Overview

DeadliftAnalyser is a desktop application that uses two synchronised cameras — an **iPhone (side view)** and a **Logitech C920 webcam (front view)** — to capture, analyse, and score deadlift repetitions in real time. It applies **MediaPipe BlazePose** for 2D pose estimation on both views simultaneously, then reconstructs full **3D skeletal landmarks** via stereo triangulation to detect form faults that are geometrically invisible from a single camera angle.

The system addresses a two-layered gap in existing tools:
1. **Single-camera blindness** — lateral tilt is undetectable from a side view alone; prior research tools were all monocular.
2. **Stateless feedback** — commercial tools (Tempo, Kaia, FormCheck) give per-session scores with no longitudinal tracking; this system persists every session as a JSON summary and charts progress over time.

---

## Key Features

| Feature | Detail |
|---|---|
| **Dual-camera sync** | iPhone (side) + Logitech C920 (front), recorded simultaneously |
| **Parallel 2D pose estimation** | MediaPipe BlazePose on both cameras via Python `multiprocessing` |
| **3D reconstruction** | Stereo triangulation from calibrated camera pair |
| **Back rounding detection** | Novel droop ratio metric (dx/dy of shoulder–hip midpoints) |
| **Lateral tilt detection** | 3D shoulder angle — only possible with dual-camera setup |
| **Hip-angle-conditional thresholds** | Droop thresholds relax automatically for deep hip hinge; eliminates false positives for tall lifters |
| **Body proportion profiling** | Femur-to-Torso Ratio (FTR) measured live from side camera |
| **Rep detection & scoring** | Automatic rep segmentation, setup rep skipped, per-rep scores 0–100 |
| **Progress Tracker** | Charts form score, rounding, and weight lifted across all historical sessions |
| **Two-phase pipeline** | Scores shown at ~60s; skeleton animation renders in background thread |
| **PyQt5 desktop GUI** | One-click pipeline, video player, session review, export ready |

---

## System Architecture

```
User (Lifter)
     │
     ▼
┌─────────────────────────┐
│  Input Layer            │
│  cam0: Logitech C920    │  ← Front view
│  cam1: iPhone (side)    │  ← Side view
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  record_dual.py         │  ← Synchronised dual recording
│  Session folder created │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  pose_detector_2d.py    │  ← BlazePose on both videos (parallel)
│  2D landmark extraction │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  reconstruction_3d.py   │  ← Stereo triangulation → 3D landmarks
└───────────┬─────────────┘
            │
       ┌────┴────┐
       ▼         ▼
  Rule Engine   body_proportion.py
  Rounding,     FTR + hip-angle-
  tilt, scoring conditional adapt.
       │
       ▼
┌─────────────────────────┐
│  skeleton_visualiser.py │  ← Animated dual-view skeleton (background)
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  app.py (PyQt5 GUI)     │  ← Session review, rep breakdown, controls
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  session_summary.json   │  ← Persisted per session
│  Progress Tracker tab   │  ← Charts across all sessions
└─────────────────────────┘
```

---

## Hardware Requirements

| Component | Specification | Role |
|---|---|---|
| **Computer** | MacBook Air M4 (Apple Silicon) | Processing host |
| **Camera 0** | Logitech C920 HD Webcam | Front view (lateral tilt, shoulder asymmetry) |
| **Camera 1** | iPhone via Continuity Camera | Side view (back rounding, hip angle, knee angle) |
| **Calibration board** | 9×6 chessboard, 42mm squares | Stereo calibration |

> **Note:** The system was developed and tested on macOS. Windows/Linux compatibility is not guaranteed due to `cv2.imshow` threading constraints on macOS and Continuity Camera (iPhone) availability.

---

## Software Requirements

- Python 3.12.6
- macOS 14+ (for iPhone Continuity Camera support)

### Python Dependencies

```
mediapipe>=0.10
opencv-python>=4.9
numpy>=1.26
PyQt5>=5.15
matplotlib>=3.8
pandas>=2.2
openpyxl>=3.1
scipy>=1.13
```

Install all dependencies via:

```bash
pip install -r requirements.txt
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/deadlift-analyser.git
cd deadlift-analyser

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run stereo calibration (first time only)
#    Print a 9x6 chessboard with 42mm squares
#    Position both cameras, then run:
python calibrate_stereo.py

# 5. Launch the application
python app.py
```

---

## Project Structure

```
deadlift_trainer/
│
├── app.py                      # Main PyQt5 GUI application
├── record_dual.py              # Synchronised dual-camera recording
├── pose_detector_2d.py         # MediaPipe BlazePose (parallel, both cameras)
├── reconstruction_3d.py        # Stereo triangulation → 3D landmarks
├── skeleton_visualiser.py      # Animated skeleton with fault colour-coding
├── body_proportion.py          # FTR measurement + hip-angle-conditional thresholds
├── run_evaluation.py           # Evaluation framework (reads ground_truth.xlsx)
│
├── calibration/
│   ├── calibrate_stereo.py     # Stereo calibration script
│   └── stereo_params.npz       # Saved calibration matrices (generated)
│
├── sessions/                   # Auto-created; one folder per session
│   └── 20260421_220743/
│       ├── cam0.mp4            # Front view video
│       ├── cam1.mp4            # Side view video
│       ├── landmarks_2d_cam0.npy
│       ├── landmarks_2d_cam1.npy
│       ├── landmarks_3d.npy
│       ├── skeleton_animation.mp4
│       └── session_summary.json
│
├── ground_truth.xlsx           # Manual annotation for evaluation
├── requirements.txt
└── README.md
```

---

## Usage

### 1. First Launch — Body Proportion Measurement

On launch, the app prompts you to stand **side-on** to the iPhone camera, upright with arms relaxed. Hold still for **4 seconds**. The system measures your **Femur-to-Torso Ratio (FTR)** and stores it for threshold adaptation.

Press **Q** to skip and use default thresholds.

> **macOS note:** The preview window runs in a subprocess to avoid `cv2.imshow` threading restrictions. If it doesn't appear, the system will collect silently and still compute FTR.

### 2. Recording a Session

1. Enter your weight (kg) in the top bar
2. Click **Record & Analyse**
3. Perform your deadlift set — the system records both cameras simultaneously
4. Press **Q** to stop recording when your set is complete

### 3. Analysis

The pipeline runs automatically after recording:

- **~60 seconds** — 2D pose extraction + 3D reconstruction + rep scoring. Scores appear immediately.
- **2–3 minutes** — Skeleton animation renders in the background. Loads automatically when done.

### 4. Session Review

- **Rep Breakdown table** — per-rep score, rounding value, tilt angle, faults flagged
- **Skeleton animation player** — dual-view animated skeleton with colour-coded fault overlays (green = good, amber = caution, red = fault)
- **Suggestions panel** — personalised feedback based on dominant fault type

### 5. Re-analysing a Session

Click **Re-analyse Session** to reprocess an existing session folder without re-recording. Useful for testing different threshold values.

### 6. Progress Tracker

Switch to the **Progress Tracker** tab to view charts across all saved sessions:
- Average form score trend
- Back rounding trend
- Weight lifted over time

---

## Technical Design

### Droop Ratio — Back Rounding Detection

The system replaces the traditional **spine-angle-from-vertical** metric, which produces false positives during the hip hinge phase of the deadlift (when a forward lean is biomechanically correct).

The **droop ratio** measures the horizontal-to-vertical displacement between shoulder and hip midpoints:

```
droop = Δx / Δy

where:
  Δx = horizontal distance (shoulder midpoint → hip midpoint)
  Δy = vertical distance (shoulder midpoint → hip midpoint)
```

| Droop Value | Classification |
|---|---|
| < 1.2 | ✅ Neutral — good form |
| 1.2 – 1.8 | ⚠️ Caution — moderate rounding |
| > 1.8 | ❌ Fault — significant rounding |

All thresholds are computed from the **side camera (cam1/iPhone)** only.

---

### Hip-Angle-Conditional Thresholds

Tall lifters with long femurs reach a greater forward lean at the bottom of the lift, causing elevated droop ratios even with correct form. A fixed threshold would always flag them as rounding when they aren't.

The system relaxes thresholds dynamically based on the **minimum hip angle** observed:

| Hip Angle at Bottom | Threshold Multiplier |
|---|---|
| ≥ 35° | × 1.00 (standard) |
| < 35° | × 1.25 |
| < 20° | × 1.70 |

**Real-world validation:** A 190cm participant with hip angle 25.6° at bottom and FTR=0.77 was triggering false rounding faults. With hip-angle-conditional thresholds applied, their session score improved from 60% to ~75–78% and rounding flags for reps 1–2 were correctly cleared.

---

### Femur-to-Torso Ratio (FTR)

FTR is measured from the side camera using MediaPipe landmarks:

```
FTR = femur_length / torso_length

where:
  femur_length  = hip → knee distance
  torso_length  = shoulder → hip distance
```

**Key finding (negative result):** FTR was expected to correlate with body height and explain false positive rates. Investigation across participants (158cm and 190cm) showed **FTR ≈ 0.77 for both** — the ratio normalises for body size and cannot distinguish tall from short lifters with the same proportions. The hip-angle-conditional threshold mechanism proved to be the operative fix, not FTR. FTR is retained as a supplementary body characterisation metric only.

---

### Stereo Calibration

- **Board:** 9×6 chessboard, 42mm square size
- **Image pairs:** 20 stereo pairs
- **Reprojection error:** **0.92 pixels** (target < 1.5px ✅)
- **Outputs:** Camera matrices, distortion coefficients, rotation matrix (R), translation vector (T), essential matrix (E), fundamental matrix (F) — saved to `calibration/stereo_params.npz`

---

### Scoring Algorithm

Each rep is scored from 100 (perfect) with deductions applied:

```
score = 100
      − rounding_deduction   (from hip-angle-conditional droop value)
      − tilt_deduction        (from 3D shoulder angle)
      − knee_deduction        (optional; from side camera)
```

**Rounding deductions** (applied to hip-angle-conditional droop):

| Condition | Deduction |
|---|---|
| Droop > 40 (normalised) | −40 points |
| Droop > 20 | −20 points |
| Droop > 10 | −10 points |

**Tilt deductions:**

| Shoulder tilt angle | Deduction |
|---|---|
| > 15° | −20 points |
| > 8° | −10 points |

> **Note:** The deduction bands themselves (>40, >20, >10) are fixed. Only the droop value fed into them is hip-angle-conditional.

---

## Evaluation Results

### Back Rounding Detection (n = 5 sessions)

| Metric | Value |
|---|---|
| Precision | 67% |
| Recall | **100%** |
| F1 Score | 80% |
| Accuracy | 80% |
| TP / FP / TN / FN | 2 / 1 / 2 / 0 |

100% recall means the system **never missed a real rounding fault**. The single false positive was the 190cm tall participant session — corrected by hip-angle-conditional thresholds.

---

### Lateral Tilt Detection (n = 9 sessions)

| Metric | Value |
|---|---|
| Precision | 75% |
| Recall | 75% |
| F1 Score | 75% |
| Accuracy | 78% |
| TP / FP / TN / FN | 3 / 1 / 4 / 1 |

---

### 2D vs 3D Comparison — Key Finding

| System | Rounding F1 | Tilt F1 |
|---|---|---|
| 2D only (single camera) | 80% | **0%** |
| 3D dual-camera (this system) | 80% | **75%** |

Lateral tilt is **geometrically invisible** from a single side camera — the 0% → 75% F1 jump proves the necessity of the dual-camera architecture.

---

### Fatigue Analysis (n = 21 sessions)

- Form score trend: **+0.3% per rep** (slight improvement, not degradation — counterintuitive)
- Rep 6 dip observed: score at 64.2% vs Rep 1 average 79.8% (final-rep effect)
- Sample sizes: n=21 at Rep 1, declining to n=2 at Rep 11

---

## Pipeline Details

| Stage | Script | Approx. Time |
|---|---|---|
| Dual-camera recording | `record_dual.py` | User-controlled |
| 2D pose extraction (parallel) | `pose_detector_2d.py` | ~30–40s |
| 3D reconstruction | `reconstruction_3d.py` | ~10–15s |
| Rule engine + scoring | (inside `app.py`) | ~5s |
| **Scores visible to user** | — | **~60s total** |
| Skeleton animation render | `skeleton_visualiser.py` | 2–3 min (background) |

---

## Known Issues & Limitations

| Issue | Detail |
|---|---|
| `cv2.imshow` crash on macOS | Calling from a QThread triggers `Unknown C++ exception`. Fixed by running body proportion measurement in a subprocess. |
| iPhone camera index varies | Continuity Camera may appear as index 1 or 2 depending on connection order. Try both if the wrong feed appears. |
| Setup rep skipped | The first detected rep is dropped when 2+ reps are found (it captures the bar-grip setup, not a lift). Sessions with only 1 rep will show 0 scored reps. |
| Small evaluation dataset | n=5 for rounding, n=9 for tilt. Results are indicative; larger datasets needed for statistical significance. |
| Single participant type | All evaluation sessions used recreational lifters. Performance on powerlifting competition form (intentional forward lean) has not been tested. |
| macOS only | Continuity Camera (iPhone as webcam) requires macOS 14+. Windows users would need to substitute a different side-view camera at index 1. |

---

## Future Work

1. **Wider camera support** — Android phone as side camera; generalise beyond Apple Continuity Camera
2. **Larger evaluation dataset** — recruit 10+ participants across height ranges (155cm–200cm) to improve statistical power
3. **Knee cave detection** — add valgus collapse detection from front camera (currently out of scope)
4. **Mobile companion app** — stream results to phone for gym use without a laptop
5. **Multi-exercise support** — extend pipeline to squat, Romanian deadlift, hip hinge variants

---

## References

- Bazarevsky, V. et al. (2020) *BlazePose: On-device Real-time Body Pose Tracking.* arXiv:2006.10204
- Bradski, G. and Kaehler, A. (2008) *Learning OpenCV.* O'Reilly Media
- Cao, Z. et al. (2019) 'OpenPose: Realtime Multi-Person 2D Pose Estimation', *IEEE TPAMI*, 43(1), pp. 172–186
- Escamilla, R.F. et al. (2001) 'A three-dimensional biomechanical analysis of the deadlift', *Medicine & Science in Sports & Exercise*, 33(8), pp. 1260–1268
- Hartley, R. and Zisserman, A. (2004) *Multiple View Geometry in Computer Vision.* 2nd edn. Cambridge University Press
- McGill, S. (2016) *Low Back Disorders: Evidence-Based Prevention and Rehabilitation.* 3rd edn. Human Kinetics
- Nguyen, T. et al. (2021) 'Dual-camera rehabilitation exercise assessment', *Journal of Biomedical Informatics*, 118, p. 103798
- Zhang, Z. (2000) 'A flexible new technique for camera calibration', *IEEE TPAMI*, 22(11), pp. 1330–1334

---

## Licence

This project was developed as an academic Final Year Project at Nottingham Trent University. All rights reserved. Not licensed for commercial use.

---

*Built with MediaPipe · OpenCV · PyQt5 · Python 3.12*
