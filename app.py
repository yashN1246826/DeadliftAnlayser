import sys
import os
import json
import subprocess
import numpy as np
import cv2
from datetime import datetime

try:
    from body_proportion import (
        measure_from_camera,
        measure_from_video,
        get_proportion_description,
    )
    BP_AVAILABLE = True
except ImportError:
    BP_AVAILABLE = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QSplitter, QTableWidget,
    QTableWidgetItem, QTextEdit, QFileDialog, QProgressBar,
    QFrame, QHeaderView, QTabWidget, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# ══════════════════════════════════════════════════════════════
#  ANALYSIS FUNCTIONS  (match skeleton_visualiser.py exactly)
# ══════════════════════════════════════════════════════════════

def _droop_thresholds_for_hip(hip_deg):
    """Droop ratio thresholds that relax as hip angle decreases.
    Deep forward lean is biomechanically required at low hip angles — not rounding."""
    if hip_deg >= 50:   return 1.2, 1.8   # standard
    elif hip_deg >= 35: return 1.50, 2.25  # deep lean — relax 25%
    elif hip_deg >= 20: return 1.80, 2.70  # very deep  — relax 50%
    else:               return 2.04, 3.06  # near horiz — relax 70%


def calc_rounding_score(lm, w, h, hip_angle_deg=90):
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
    caution_t, rounding_t = _droop_thresholds_for_hip(hip_angle_deg)
    if droop < 1.2:
        return max(0, droop * 8)
    elif droop < caution_t:
        return max(0, droop * 8)
    elif droop < rounding_t:
        zone_w = rounding_t - caution_t
        return 10 + (droop - caution_t) / zone_w * 24
    else:
        return min(80, 34 + (droop - rounding_t) * 70)


def calc_hip_angle(lm, w, h):
    if lm[11][3] > lm[12][3]:
        sh = np.array([lm[11][0] * w, lm[11][1] * h])
        hp = np.array([lm[23][0] * w, lm[23][1] * h])
        kn = np.array([lm[25][0] * w, lm[25][1] * h])
    else:
        sh = np.array([lm[12][0] * w, lm[12][1] * h])
        hp = np.array([lm[24][0] * w, lm[24][1] * h])
        kn = np.array([lm[26][0] * w, lm[26][1] * h])
    ba, bc = sh - hp, kn - hp
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def calc_knee_angle(lm, w, h):
    if lm[23][3] > lm[24][3]:
        hip   = np.array([lm[23][0] * w, lm[23][1] * h])
        knee  = np.array([lm[25][0] * w, lm[25][1] * h])
        ankle = np.array([lm[27][0] * w, lm[27][1] * h])
    else:
        hip   = np.array([lm[24][0] * w, lm[24][1] * h])
        knee  = np.array([lm[26][0] * w, lm[26][1] * h])
        ankle = np.array([lm[28][0] * w, lm[28][1] * h])
    ba, bc = hip - knee, ankle - knee
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def calc_lateral_tilt(lm, w, h):
    l_sh, r_sh = lm[11], lm[12]
    if min(l_sh[3], r_sh[3]) < 0.4:
        return 0
    l_sh_y, r_sh_y = l_sh[1] * h, r_sh[1] * h
    mid_hp_y = ((lm[23][1] + lm[24][1]) / 2) * h
    mid_sh_y = (l_sh_y + r_sh_y) / 2
    torso_h = abs(mid_hp_y - mid_sh_y)
    if torso_h < 30:
        return 0
    tilt_pct = (l_sh_y - r_sh_y) / torso_h * 100
    return 0 if abs(tilt_pct) < 5 else tilt_pct


def detect_reps(hip_angles, threshold=130, min_frames=10):
    reps, in_rep, rep_start = [], False, 0
    for i, angle in enumerate(hip_angles):
        if not in_rep and angle < threshold:
            in_rep, rep_start = True, i
        elif in_rep and angle >= threshold:
            if i - rep_start >= min_frames:
                reps.append((rep_start, i))
            in_rep = False
    return reps


def score_rep(rounding_scores, tilts, start, end):
    max_r = max(rounding_scores[start:end], default=0)
    max_t = max((abs(t) for t in tilts[start:end]), default=0)
    score = 100
    if max_r > 40:   score -= 35
    elif max_r > 20: score -= 15
    elif max_r > 10: score -= 5
    if max_t > 25:   score -= 20
    elif max_t > 15: score -= 10
    elif max_t > 8:  score -= 5
    return max(0, min(100, score)), max_r, max_t


def generate_suggestions(rep_num, max_rounding, max_tilt):
    lines = []
    if max_rounding > 40:
        lines.append(
            f"Rep {rep_num}: Significant back rounding detected "
            f"(score {max_rounding:.0f}). Keep your chest up and engage "
            f"your lats before pulling — think 'proud chest'."
        )
    elif max_rounding > 20:
        lines.append(
            f"Rep {rep_num}: Mild rounding (score {max_rounding:.0f}). "
            f"Brace your core harder — deep breath before each rep."
        )
    if max_tilt > 25:
        lines.append(
            f"Rep {rep_num}: Lateral tilt of {max_tilt:.0f}% detected. "
            f"Check for uneven grip or strength imbalance."
        )
    elif max_tilt > 15:
        lines.append(
            f"Rep {rep_num}: Minor lateral lean ({max_tilt:.0f}%). "
            f"Ensure equal foot pressure and grip width."
        )
    if not lines:
        lines.append(f"Rep {rep_num}: Good form — maintain this technique.")
    return lines


# ══════════════════════════════════════════════════════════════
#  SESSION PERSISTENCE
# ══════════════════════════════════════════════════════════════

def _get_recorded_timestamp(session_dir):
    """
    Use the session folder name as the true recorded timestamp.
    Example folder: 20260423_230351
    """
    session_id = os.path.basename(session_dir)
    try:
        dt = datetime.strptime(session_id, "%Y%m%d_%H%M%S")
        return dt.isoformat()
    except Exception:
        return datetime.now().isoformat()


def save_session_summary(session_dir, rep_results, rounding_scores,
                         hip_angles, knee_angles, tilts, fps,
                         weight_kg=None):
    """
    Save a JSON summary for long-term progress tracking.
    Written to: session_dir/session_summary.json
    """
    faults = set()
    for r in rep_results:
        if r['rounding'] > 20:
            faults.add("back_rounding")
        if abs(r['tilt']) > 15:
            faults.add("lateral_tilt")

    rep_roundings = [r['rounding'] for r in rep_results]
    rep_tilts = [abs(r['tilt']) for r in rep_results]

    summary = {
        "timestamp":       _get_recorded_timestamp(session_dir),
        "session_id":      os.path.basename(session_dir),
        "weight_kg":       weight_kg,
        "num_reps":        len(rep_results),
        "avg_score":       round(
            sum(r['score'] for r in rep_results) / len(rep_results), 1
        ) if rep_results else 0,
        "rep_scores":      [r['score'] for r in rep_results],
        "max_rounding":    round(float(max(rounding_scores)), 1) if len(rounding_scores) else 0.0,
        "avg_rounding":    round(float(np.mean(rounding_scores)), 1) if len(rounding_scores) else 0.0,
        "max_rep_rounding": round(float(max(rep_roundings)), 1) if rep_roundings else 0.0,
        "avg_rep_rounding": round(float(np.mean(rep_roundings)), 1) if rep_roundings else 0.0,
        "max_tilt":        round(float(max(abs(t) for t in tilts)), 1) if len(tilts) else 0.0,
        "avg_rep_tilt":    round(float(np.mean(rep_tilts)), 1) if rep_tilts else 0.0,
        "avg_hip_angle":   round(float(np.mean(hip_angles)), 1) if len(hip_angles) else 0.0,
        "avg_knee_angle":  round(float(np.mean(knee_angles)), 1) if len(knee_angles) else 0.0,
        "faults_detected": list(faults),
    }

    out_path = os.path.join(session_dir, "session_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return out_path


def load_all_sessions(project_dir):
    """
    Scan data/recordings/ for session_summary.json files.
    Returns list of dicts sorted by recorded timestamp (oldest first).
    """
    recordings_dir = os.path.join(project_dir, "data", "recordings")
    if not os.path.isdir(recordings_dir):
        return []
    sessions = []
    for d in os.listdir(recordings_dir):
        p = os.path.join(recordings_dir, d, "session_summary.json")
        if os.path.exists(p):
            try:
                with open(p) as f:
                    s = json.load(f)

                # Repair old files that saved load-time instead of record-time
                if not s.get("timestamp"):
                    s["timestamp"] = _get_recorded_timestamp(os.path.join(recordings_dir, d))
                else:
                    try:
                        datetime.fromisoformat(s["timestamp"])
                    except Exception:
                        s["timestamp"] = _get_recorded_timestamp(os.path.join(recordings_dir, d))

                # Backfill newer summary fields for older sessions
                if "avg_rep_rounding" not in s:
                    s["avg_rep_rounding"] = s.get("avg_rounding", s.get("max_rounding", 0))
                if "max_rep_rounding" not in s:
                    s["max_rep_rounding"] = s.get("max_rounding", 0)

                sessions.append(s)
            except Exception:
                pass
    sessions.sort(key=lambda s: s.get("timestamp", ""))
    return sessions


# ══════════════════════════════════════════════════════════════
#  PIPELINE WORKER
# ══════════════════════════════════════════════════════════════

class PipelineWorker(QThread):
    progress = pyqtSignal(str, int)
    finished = pyqtSignal(str)
    error    = pyqtSignal(str)

    def __init__(self, project_dir, skip_recording=False, session_dir=None):
        super().__init__()
        self.project_dir    = project_dir
        self.skip_recording = skip_recording
        self.session_dir    = session_dir  # if set, emit this folder not latest

    def run(self):
        scripts = []
        if not self.skip_recording:
            scripts.append(("record_dual.py",
                             "Recording — press 'q' to stop...", None))
        scripts += [
            ("pose_detector_2d.py",   "Running MediaPipe pose estimation...", 600),
            ("reconstruction_3d.py",  "Triangulating 3D landmarks...",        300),
            ("skeleton_visualiser.py","Generating skeleton animation...",      600),
        ]
        total = len(scripts)
        for i, (script, msg, timeout) in enumerate(scripts):
            self.progress.emit(msg, int(i / total * 100))
            path = os.path.join(self.project_dir, script)
            if not os.path.exists(path):
                self.error.emit(f"Script not found: {path}")
                return
            try:
                res = subprocess.run(
                    [sys.executable, path],
                    cwd=self.project_dir,
                    capture_output=True, text=True, timeout=timeout
                )
                if res.returncode != 0:
                    self.error.emit(
                        f"{script} failed:\n{res.stderr[-500:]}"
                    )
                    return
            except subprocess.TimeoutExpired:
                self.error.emit(f"{script} timed out after {timeout}s")
                return
            except Exception as e:
                self.error.emit(f"{script} error: {e}")
                return

        self.progress.emit("Complete!", 100)
        try:
            # If a specific session was requested, emit that directly
            if self.session_dir and os.path.isdir(self.session_dir):
                self.finished.emit(self.session_dir)
                return
            rec_dir = os.path.join(self.project_dir, "data", "recordings")
            sessions = sorted([
                os.path.join(rec_dir, d)
                for d in os.listdir(rec_dir)
                if os.path.isdir(os.path.join(rec_dir, d))
            ])
            if sessions:
                self.finished.emit(sessions[-1])
            else:
                self.error.emit("No recording sessions found.")
        except Exception as e:
            self.error.emit(f"Error finding session: {e}")


# ══════════════════════════════════════════════════════════════
#  VIDEO PLAYER
# ══════════════════════════════════════════════════════════════

class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self._next_frame)
        self.playing = False
        self.total_frames = self.current_frame = 0
        self.fps = 30

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.display = QLabel("Load a session to view the skeleton animation")
        self.display.setAlignment(Qt.AlignCenter)
        self.display.setMinimumSize(520, 380)
        self.display.setStyleSheet(
            "background-color: #1a1a1a; border-radius: 8px; "
            "color: #666666; font-size: 13px;"
        )
        layout.addWidget(self.display)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(8)
        self.play_btn = QPushButton("▶  Play")
        self.play_btn.setFixedWidth(100)
        self.play_btn.setFixedHeight(32)
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setEnabled(False)
        ctrl.addWidget(self.play_btn)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setEnabled(False)
        self.slider.sliderPressed.connect(self._pause)
        self.slider.sliderReleased.connect(self._seek)
        ctrl.addWidget(self.slider)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setFixedWidth(110)
        self.time_label.setStyleSheet("color: #888888; font-size: 12px;")
        ctrl.addWidget(self.time_label)
        layout.addLayout(ctrl)

    def load_video(self, path):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            self.display.setText("Failed to open video")
            return
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.slider.setMaximum(max(self.total_frames - 1, 0))
        self.slider.setEnabled(True)
        self.play_btn.setEnabled(True)
        self.current_frame = 0
        self._show_frame(0)

    def _show_frame(self, idx):
        if not self.cap:
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        dw, dh = self.display.width() - 4, self.display.height() - 4
        if dw > 0 and dh > 0:
            scale = min(dw / w, dh / h, 1.0)
            if scale < 1.0:
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)),
                                   interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
        img = QImage(frame.data, w, h, w*ch, QImage.Format_RGB888)
        self.display.setPixmap(QPixmap.fromImage(img))
        self.current_frame = idx
        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        cur_s = idx / self.fps
        tot_s = self.total_frames / self.fps
        self.time_label.setText(
            f"{int(cur_s//60)}:{int(cur_s%60):02d} / "
            f"{int(tot_s//60)}:{int(tot_s%60):02d}"
        )

    def _next_frame(self):
        if self.current_frame < self.total_frames - 1:
            self._show_frame(self.current_frame + 1)
        else:
            self._pause()
            self._show_frame(0)

    def _toggle_play(self):
        self._pause() if self.playing else self._play()

    def _play(self):
        self.playing = True
        self.play_btn.setText("⏸  Pause")
        self.timer.start(int(1000 / self.fps))

    def _pause(self):
        self.playing = False
        self.play_btn.setText("▶  Play")
        self.timer.stop()

    def _seek(self):
        self._show_frame(self.slider.value())


# ══════════════════════════════════════════════════════════════
#  SESSION CHART  (rounding over time, single session)
# ══════════════════════════════════════════════════════════════

class SessionChart(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(5, 2.2), facecolor='#2b2b2b')
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.10, right=0.96, top=0.88, bottom=0.20)
        self.setMinimumHeight(160)

    def plot(self, fps, rounding_scores, hip_angles, reps):
        self.ax.clear()
        self.ax.set_facecolor('#2b2b2b')
        t = np.arange(len(rounding_scores)) / fps
        self.ax.plot(t, rounding_scores, color='#ff5555',
                     lw=1.5, label='Rounding', alpha=0.9)
        scaled = [a / 2 for a in hip_angles]
        self.ax.plot(t, scaled, color='#55aaff',
                     lw=1, label='Hip ÷2', alpha=0.6)
        y_top = max(max(rounding_scores, default=0),
                    max(scaled, default=0)) * 1.1 or 50
        for i, (s, e) in enumerate(reps):
            self.ax.axvspan(s/fps, e/fps, alpha=0.07, color='white')
            self.ax.text((s+e)/2/fps, y_top*0.92, f'R{i+1}',
                         ha='center', va='top', color='#cccccc',
                         fontsize=8, fontweight='bold')
        self.ax.axhline(40, color='#ff5555', ls='--', alpha=0.3, lw=0.8)
        self.ax.axhline(20, color='#ffaa00', ls='--', alpha=0.3, lw=0.8)
        self.ax.set_xlabel('Time (s)', color='#aaaaaa', fontsize=9)
        self.ax.set_ylabel('Score',    color='#aaaaaa', fontsize=9)
        self.ax.set_title('Rounding score over time',
                          color='white', fontsize=10, pad=6)
        self.ax.set_ylim(bottom=0, top=y_top)
        self.ax.tick_params(colors='#888888', labelsize=8)
        self.ax.legend(loc='upper right', fontsize=7,
                       facecolor='#333333', edgecolor='#555555',
                       labelcolor='white')
        for sp in ['top', 'right']:
            self.ax.spines[sp].set_visible(False)
        for sp in ['bottom', 'left']:
            self.ax.spines[sp].set_color('#555555')
        self.draw()


# ══════════════════════════════════════════════════════════════
#  PROGRESS CHART  (multi-session, three panels)
# ══════════════════════════════════════════════════════════════

class ProgressChart(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 5), facecolor='#2b2b2b')
        super().__init__(self.fig)
        self.setMinimumHeight(320)

    def plot(self, sessions):
        self.fig.clear()
        self.fig.subplots_adjust(
            left=0.12, right=0.96, top=0.93,
            bottom=0.08, hspace=0.55
        )

        if not sessions:
            ax = self.fig.add_subplot(111)
            ax.set_facecolor('#2b2b2b')
            ax.text(0.5, 0.5,
                    'No sessions recorded yet.\nComplete a session to see your progress.',
                    ha='center', va='center', color='#666666',
                    fontsize=11, transform=ax.transAxes)
            ax.axis('off')
            self.draw()
            return

        labels = []
        for s in sessions:
            try:
                dt = datetime.fromisoformat(s['timestamp'])
                labels.append(dt.strftime('%d %b'))
            except Exception:
                labels.append(s.get('session_id', '?')[:8])

        scores   = [s.get('avg_score', 0) for s in sessions]
        rounding = [s.get('avg_rep_rounding', s.get('avg_rounding', s.get('max_rounding', 0))) for s in sessions]
        weights  = [s.get('weight_kg') for s in sessions]
        x        = list(range(len(sessions)))

        # Panel 1: Form score trend
        ax1 = self.fig.add_subplot(311)
        ax1.set_facecolor('#2b2b2b')
        ax1.plot(x, scores, color='#50e880', lw=2, marker='o',
                 markersize=5, markerfacecolor='white')
        ax1.fill_between(x, scores, alpha=0.15, color='#50e880')
        ax1.axhline(80, color='#50e880', ls='--', alpha=0.3, lw=0.8)
        ax1.set_ylim(0, 105)
        ax1.set_ylabel('Score %', color='#aaaaaa', fontsize=8)
        ax1.set_title('Avg form score per session',
                      color='white', fontsize=9, pad=4)
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, fontsize=7, color='#888888')
        ax1.tick_params(colors='#888888', labelsize=7)
        if len(scores) > 1:
            diff = scores[-1] - scores[0]
            ax1.annotate(
                f"{'+'if diff>=0 else ''}{diff:.0f}%",
                xy=(x[-1], scores[-1]),
                xytext=(x[-1]-0.3, min(scores[-1]+8, 100)),
                color='#50e880', fontsize=8, fontweight='bold'
            )
        for sp in ['top','right']: ax1.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax1.spines[sp].set_color('#555555')

        # Panel 2: Avg rep rounding per session
        ax2 = self.fig.add_subplot(312)
        ax2.set_facecolor('#2b2b2b')
        r_cols = ['#ff5555' if r > 40 else '#ffaa00' if r > 20
                  else '#50e880' for r in rounding]
        ax2.bar(x, rounding, color=r_cols, alpha=0.8, width=0.5)
        ax2.axhline(40, color='#ff5555', ls='--', alpha=0.3, lw=0.8)
        ax2.axhline(20, color='#ffaa00', ls='--', alpha=0.3, lw=0.8)
        ax2.set_ylabel('Score', color='#aaaaaa', fontsize=8)
        ax2.set_title('Avg rep back rounding per session  (lower = better)',
                      color='white', fontsize=9, pad=4)
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, fontsize=7, color='#888888')
        ax2.tick_params(colors='#888888', labelsize=7)
        for sp in ['top','right']: ax2.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax2.spines[sp].set_color('#555555')

        # Panel 3: Weight lifted
        ax3 = self.fig.add_subplot(313)
        ax3.set_facecolor('#2b2b2b')
        has_w = any(w is not None for w in weights)
        if has_w:
            w_vals = [w if w is not None else 0 for w in weights]
            ax3.plot(x, w_vals, color='#55aaff', lw=2, marker='o',
                     markersize=5, markerfacecolor='white')
            ax3.fill_between(x, w_vals, alpha=0.15, color='#55aaff')
            ax3.set_ylabel('kg', color='#aaaaaa', fontsize=8)
            if len(w_vals) > 1:
                g = w_vals[-1] - w_vals[0]
                ax3.annotate(
                    f"{'+'if g>=0 else ''}{g:.0f} kg",
                    xy=(x[-1], w_vals[-1]),
                    xytext=(x[-1]-0.3, w_vals[-1]+1),
                    color='#55aaff', fontsize=8, fontweight='bold'
                )
        else:
            ax3.text(0.5, 0.5,
                     'Enter weight before recording\nto track load progression',
                     ha='center', va='center', color='#555555',
                     fontsize=9, transform=ax3.transAxes)
        ax3.set_title('Weight lifted per session',
                      color='white', fontsize=9, pad=4)
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, fontsize=7, color='#888888')
        ax3.tick_params(colors='#888888', labelsize=7)
        for sp in ['top','right']: ax3.spines[sp].set_visible(False)
        for sp in ['bottom','left']: ax3.spines[sp].set_color('#555555')

        self.draw()


# ══════════════════════════════════════════════════════════════
#  STYLESHEET
# ══════════════════════════════════════════════════════════════

DARK_STYLE = """
QMainWindow { background-color: #1e1e1e; }
QWidget     { background-color: #1e1e1e; color: #dddddd; }
QTabWidget::pane { border: 1px solid #3a3a3a; border-radius: 6px; }
QTabBar::tab {
    background: #2a2a2a; color: #aaaaaa;
    border: 1px solid #3a3a3a; padding: 6px 18px;
    border-bottom: none; border-radius: 4px 4px 0 0;
}
QTabBar::tab:selected { background: #333333; color: #ffffff; }
QPushButton {
    background-color: #333333; color: white;
    border: 1px solid #555555; border-radius: 6px;
    padding: 6px 16px; font-size: 13px;
}
QPushButton:hover    { background-color: #444444; }
QPushButton:pressed  { background-color: #555555; }
QPushButton:disabled { background-color: #2a2a2a; color: #555555; border-color: #3a3a3a; }
QPushButton#primary  { background-color: #2d5a27; border-color: #3a7a32; }
QPushButton#primary:hover { background-color: #3a7a32; }
QSlider::groove:horizontal { height: 6px; background: #444444; border-radius: 3px; }
QSlider::handle:horizontal { background: #aaaaaa; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }
QSlider::sub-page:horizontal { background: #50c878; border-radius: 3px; }
QProgressBar { background-color: #333333; border: none; border-radius: 4px; text-align: center; color: white; font-size: 11px; }
QProgressBar::chunk { background-color: #50c878; border-radius: 4px; }
QTableWidget { background-color: #2b2b2b; gridline-color: #3a3a3a; border: 1px solid #3a3a3a; border-radius: 4px; font-size: 12px; }
QTableWidget::item { padding: 4px 8px; }
QHeaderView::section { background-color: #333333; color: #aaaaaa; border: 1px solid #3a3a3a; padding: 5px 8px; font-size: 11px; font-weight: bold; }
QTextEdit { background-color: #2b2b2b; border: 1px solid #3a3a3a; border-radius: 4px; padding: 8px; font-size: 12px; }
QDoubleSpinBox { background-color: #2b2b2b; border: 1px solid #555555; border-radius: 4px; padding: 4px 8px; color: white; font-size: 12px; }
QStatusBar { background-color: #252525; color: #888888; font-size: 11px; }
QSplitter::handle { background-color: #3a3a3a; width: 2px; }
"""


# ══════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deadlift Form Analyser — Yash Kumar (N1246826)")
        self.setMinimumSize(1100, 740)
        self.resize(1220, 800)
        self.project_dir = self._find_project_dir()
        self.session_dir        = None
        self.worker             = None
        self.proportion_profile = None
        self._last_rep_results = []
        self._last_rounding    = []
        self._last_hip_angles  = []
        self._last_knee_angles = []
        self._last_tilts       = []
        self._last_fps         = 30.0

        self._build_ui()
        self.setStyleSheet(DARK_STYLE)

    def _find_project_dir(self):
        candidate = os.path.join(os.path.expanduser("~"), "deadlift_trainer")
        if os.path.isdir(candidate):
            return candidate
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(script_dir, "record_dual.py")):
            return script_dir
        return candidate

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12, 10, 12, 6)
        root.setSpacing(8)

        # Header
        header = QHBoxLayout()
        header.setSpacing(10)
        title = QLabel("DEADLIFT FORM ANALYSER")
        title.setFont(QFont("Helvetica", 15, QFont.Bold))
        title.setStyleSheet("color: #ffffff; letter-spacing: 1px;")
        header.addWidget(title)
        header.addStretch()

        wt_lbl = QLabel("Weight (kg):")
        wt_lbl.setStyleSheet("color: #888888; font-size: 12px;")
        header.addWidget(wt_lbl)

        self.weight_input = QDoubleSpinBox()
        self.weight_input.setRange(0, 500)
        self.weight_input.setValue(0)
        self.weight_input.setSuffix(" kg")
        self.weight_input.setDecimals(1)
        self.weight_input.setFixedWidth(100)
        self.weight_input.setFixedHeight(32)
        self.weight_input.setToolTip(
            "Enter the weight you are lifting — saved for progress tracking"
        )
        header.addWidget(self.weight_input)

        self.record_btn = QPushButton("⏺  Record && Analyse")
        self.record_btn.setObjectName("primary")
        self.record_btn.setFixedHeight(34)
        self.record_btn.clicked.connect(self._start_full_pipeline)
        header.addWidget(self.record_btn)

        self.analyse_btn = QPushButton("⚡  Re-analyse Session")
        self.analyse_btn.setFixedHeight(34)
        self.analyse_btn.clicked.connect(self._analyse_latest)
        header.addWidget(self.analyse_btn)

        self.load_btn = QPushButton("📂  Load Session")
        self.load_btn.setFixedHeight(34)
        self.load_btn.clicked.connect(self._load_session_dialog)
        header.addWidget(self.load_btn)

        self.detect_bp_btn = QPushButton("📐  Detect Body Proportions")
        self.detect_bp_btn.setFixedHeight(34)
        self.detect_bp_btn.setToolTip(
            "Detect body proportions from loaded session video.\n"
            "Then press Re-analyse Session to apply adapted thresholds."
        )
        self.detect_bp_btn.clicked.connect(self._detect_bp_from_session)
        self.detect_bp_btn.setEnabled(False)
        header.addWidget(self.detect_bp_btn)

        root.addLayout(header)

        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(18)
        self.progress_bar.setVisible(False)
        root.addWidget(self.progress_bar)

        self.bp_status_label = QLabel("Body proportions: not measured — standard thresholds active")
        self.bp_status_label.setStyleSheet("color: #666666; font-size: 11px; padding: 0 4px;")
        root.addWidget(self.bp_status_label)

        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #888888; font-size: 12px;")
        self.progress_label.setVisible(False)
        root.addWidget(self.progress_label)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_session_tab(),  "  Session Review  ")
        self.tabs.addTab(self._build_progress_tab(), "  Progress Tracker  ")
        root.addWidget(self.tabs)

        self.statusBar().showMessage(f"Ready — Project: {self.project_dir}")

    def _build_session_tab(self):
        tab = QWidget()
        splitter = QSplitter(Qt.Horizontal)

        self.video_player = VideoPlayer()
        splitter.addWidget(self.video_player)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(10, 0, 0, 0)
        rl.setSpacing(6)

        self.score_label = QLabel("—")
        self.score_label.setAlignment(Qt.AlignCenter)
        self.score_label.setFont(QFont("Helvetica", 36, QFont.Bold))
        self.score_label.setStyleSheet("color: #555555;")
        self.score_label.setFixedHeight(52)
        rl.addWidget(self.score_label)

        cap = QLabel("OVERALL SCORE")
        cap.setAlignment(Qt.AlignCenter)
        cap.setStyleSheet("color: #777777; font-size: 10px; letter-spacing: 3px;")
        rl.addWidget(cap)

        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("color: #3a3a3a;")
        rl.addWidget(div)

        rl.addWidget(self._small_label("REP BREAKDOWN"))

        self.rep_table = QTableWidget()
        self.rep_table.setColumnCount(4)
        self.rep_table.setHorizontalHeaderLabels(["Rep","Score","Rounding","Tilt"])
        self.rep_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.rep_table.verticalHeader().setVisible(False)
        self.rep_table.setSelectionMode(QTableWidget.NoSelection)
        self.rep_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.rep_table.setMaximumHeight(160)
        rl.addWidget(self.rep_table)

        rl.addWidget(self._small_label("SUGGESTIONS"))

        self.suggestions = QTextEdit()
        self.suggestions.setReadOnly(True)
        self.suggestions.setMaximumHeight(110)
        self.suggestions.setPlaceholderText(
            "Analyse a session to see personalised feedback..."
        )
        rl.addWidget(self.suggestions)

        self.session_chart = SessionChart()
        rl.addWidget(self.session_chart)

        splitter.addWidget(right)
        splitter.setSizes([560, 440])

        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.addWidget(splitter)
        return tab

    def _build_progress_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        hdr = QHBoxLayout()
        desc = QLabel(
            "Track your deadlift form quality and strength load across sessions"
        )
        desc.setStyleSheet("color: #888888; font-size: 12px;")
        hdr.addWidget(desc)
        hdr.addStretch()
        refresh_btn = QPushButton("↻  Refresh")
        refresh_btn.setFixedHeight(30)
        refresh_btn.clicked.connect(self._refresh_progress)
        hdr.addWidget(refresh_btn)
        layout.addLayout(hdr)

        splitter = QSplitter(Qt.Horizontal)

        self.progress_chart = ProgressChart()
        splitter.addWidget(self.progress_chart)

        hist = QWidget()
        hl = QVBoxLayout(hist)
        hl.setContentsMargins(10, 0, 0, 0)
        hl.setSpacing(6)

        hl.addWidget(self._small_label("SESSION HISTORY"))

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(
            ["Reps","Avg Score","Rounding","Weight"]
        )
        self.history_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setSelectionMode(QTableWidget.NoSelection)
        self.history_table.setEditTriggers(QTableWidget.NoEditTriggers)
        hl.addWidget(self.history_table)

        self.progress_summary = QLabel("Complete a session to see progress")
        self.progress_summary.setStyleSheet(
            "color: #666666; font-size: 12px; padding: 6px 0;"
        )
        self.progress_summary.setWordWrap(True)
        hl.addWidget(self.progress_summary)

        splitter.addWidget(hist)
        splitter.setSizes([520, 380])
        layout.addWidget(splitter)
        return tab

    def _small_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet(
            "color: #999999; font-size: 10px; letter-spacing: 2px;"
        )
        return lbl

    # ── ACTIONS ──

    def _set_buttons_enabled(self, en):
        self.record_btn.setEnabled(en)
        self.analyse_btn.setEnabled(en)
        self.load_btn.setEnabled(en)

    def _detect_bp_from_session(self):
        if not self.session_dir:
            self.bp_status_label.setText("No session loaded")
            return
        cam1_path = os.path.join(self.session_dir, "cam1.mp4")
        if not os.path.exists(cam1_path):
            self.bp_status_label.setText("cam1.mp4 not found in session folder")
            return
        if not BP_AVAILABLE:
            self.bp_status_label.setText("body_proportion.py not in project folder")
            return
        self._set_buttons_enabled(False)
        self.detect_bp_btn.setEnabled(False)
        self.bp_status_label.setText("Scanning video for standing frames...")
        self.bp_status_label.setStyleSheet("color: #ffaa00; font-size: 11px; padding: 0 4px;")

        class BPWorker(QThread):
            done  = pyqtSignal(object)
            error = pyqtSignal(str)
            def __init__(self, path):
                super().__init__()
                self.path = path
            def run(self):
                try:
                    from body_proportion import measure_from_video
                    profile = measure_from_video(self.path, show_preview=False)
                    self.done.emit(profile)
                except Exception as e:
                    self.error.emit(str(e))

        self._bp_worker = BPWorker(cam1_path)

        def on_done(profile):
            self.proportion_profile = profile
            desc = get_proportion_description(profile) if BP_AVAILABLE else "—"
            self.bp_status_label.setText(f"Proportions detected: {desc}")
            col = "#ffaa00" if (profile and profile.get("threshold_scale",1.0)!=1.0) else "#50e880"
            self.bp_status_label.setStyleSheet(
                f"color: {col}; font-size: 11px; padding: 0 4px;")
            self._set_buttons_enabled(True)
            self.detect_bp_btn.setEnabled(True)
            self.statusBar().showMessage(
                "Proportions detected — press Re-analyse Session to apply")

        def on_err(msg):
            self.bp_status_label.setText(f"Detection failed: {msg[:80]}")
            self._set_buttons_enabled(True)
            self.detect_bp_btn.setEnabled(True)

        self._bp_worker.done.connect(on_done)
        self._bp_worker.error.connect(on_err)
        self._bp_worker.start()

    def _start_full_pipeline(self):
        self._set_buttons_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.worker = PipelineWorker(self.project_dir, skip_recording=False)
        self._connect_worker()
        self.worker.start()

    def _analyse_latest(self):
        """Re-run the analysis pipeline on the currently loaded session."""
        if not self.session_dir or not os.path.isdir(self.session_dir):
            self.progress_label.setText(
                "No session loaded — use Load Session or Record & Analyse first"
            )
            self.progress_label.setStyleSheet("color: #ffaa00; font-size: 12px;")
            self.progress_label.setVisible(True)
            return
        self._set_buttons_enabled(False)
        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.worker = PipelineWorker(
            self.project_dir,
            skip_recording=True,
            session_dir=self.session_dir
        )
        self._connect_worker()
        self.worker.start()

    def _connect_worker(self):
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_pipeline_done)
        self.worker.error.connect(self._on_pipeline_error)

    def _load_session_dialog(self):
        default = os.path.join(self.project_dir, "data", "recordings")
        if not os.path.isdir(default):
            default = self.project_dir
        folder = QFileDialog.getExistingDirectory(
            self, "Select Session Folder", default
        )
        if folder:
            self._load_results(folder)

    def _refresh_progress(self):
        self._update_progress_tab(load_all_sessions(self.project_dir))

    # ── CALLBACKS ──

    def _on_progress(self, msg, pct):
        self.progress_bar.setValue(pct)
        self.progress_label.setText(msg)
        self.progress_label.setVisible(True)
        self.statusBar().showMessage(msg)

    def _on_pipeline_done(self, session_dir):
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self._set_buttons_enabled(True)
        self._load_results(session_dir)

    def _on_pipeline_error(self, err):
        self.progress_bar.setVisible(False)
        self.progress_label.setText(f"Error: {err}")
        self.progress_label.setStyleSheet("color: #ff5555; font-size: 12px;")
        self.progress_label.setVisible(True)
        self.statusBar().showMessage("Pipeline failed")
        self._set_buttons_enabled(True)

    # ── LOAD RESULTS ──

    def _load_results(self, session_dir):
        self.session_dir = session_dir
        processed    = os.path.join(session_dir, "processed")
        session_name = os.path.basename(session_dir)
        self.detect_bp_btn.setEnabled(True)

        video_path = os.path.join(processed, "skeleton_animation.mp4")
        if os.path.exists(video_path):
            self.video_player.load_video(video_path)
        else:
            self.statusBar().showMessage(
                f"No skeleton_animation.mp4 in {session_name}"
            )
            return

        side_p  = os.path.join(processed, "cam1_landmarks.npz")
        front_p = os.path.join(processed, "cam0_landmarks.npz")
        if not os.path.exists(side_p) or not os.path.exists(front_p):
            self.suggestions.setPlainText(
                "Landmark files not found. Run pose_detector_2d.py first."
            )
            return

        sd = np.load(side_p)
        fd = np.load(front_p)
        lm_s = sd['landmarks']
        lm_f = fd['landmarks']
        fps  = float(sd['fps'])
        ws, hs = float(sd['width']), float(sd['height'])
        wf, hf = float(fd['width']), float(fd['height'])

        n = min(len(lm_s), len(lm_f))
        lm_s, lm_f = lm_s[:n], lm_f[:n]

        rounding, hips, knees, tilts = [], [], [], []
        for i in range(n):
            hip_a = calc_hip_angle(lm_s[i], ws, hs)
            hips.append(hip_a)
            rounding.append(calc_rounding_score(lm_s[i], ws, hs, hip_a))
            knees.append(calc_knee_angle(lm_s[i], ws, hs))
            tilts.append(calc_lateral_tilt(lm_f[i], wf, hf))

        reps = detect_reps(hips)

        if not reps:
            self.score_label.setText("—")
            self.score_label.setStyleSheet("color: #888888;")
            self.rep_table.setRowCount(0)
            self.suggestions.setPlainText(
                "No reps detected.\n\n"
                "• Hip angle did not drop below 130° threshold\n"
                "• Try a full range-of-motion deadlift"
            )
            self.session_chart.plot(fps, rounding, hips, reps)
            self.statusBar().showMessage(
                f"Session: {session_name} — no reps detected"
            )
            return

        rep_results = []
        for s, e in reps:
            sc, max_r, max_t = score_rep(rounding, tilts, s, e)
            rep_results.append({
                'score':       sc,
                'rounding':    max_r,
                'tilt':        max_t,
                'suggestions': generate_suggestions(
                    len(rep_results)+1, max_r, max_t
                ),
            })

        avg = sum(r['score'] for r in rep_results) / len(rep_results)
        col = ("#50e880" if avg >= 80 else
               "#ffaa00" if avg >= 60 else "#ff5555")
        self.score_label.setText(f"{avg:.0f}%")
        self.score_label.setStyleSheet(
            f"color: {col}; font-size: 36px; font-weight: bold;"
        )

        self.rep_table.setRowCount(len(rep_results))
        for i, r in enumerate(rep_results):
            self.rep_table.setItem(i, 0, QTableWidgetItem(f"  Rep {i+1}"))
            sc_item = QTableWidgetItem(f"  {r['score']}%")
            sc_c = ("#50e880" if r['score'] >= 80 else
                    "#ffaa00" if r['score'] >= 60 else "#ff5555")
            sc_item.setForeground(QColor(sc_c))
            sc_item.setFont(QFont("Helvetica", 12, QFont.Bold))
            self.rep_table.setItem(i, 1, sc_item)

            ri = QTableWidgetItem(f"  {r['rounding']:.0f}")
            if r['rounding'] > 40:
                ri.setForeground(QColor("#ff5555"))
            elif r['rounding'] > 20:
                ri.setForeground(QColor("#ffaa00"))
            self.rep_table.setItem(i, 2, ri)

            td = "L" if r['tilt'] > 0 else "R" if r['tilt'] < 0 else ""
            ti = QTableWidgetItem(f"  {abs(r['tilt']):.0f}° {td}")
            if abs(r['tilt']) > 15:
                ti.setForeground(QColor("#ffaa00"))
            self.rep_table.setItem(i, 3, ti)

        self.suggestions.setPlainText(
            "\n\n".join(s for r in rep_results for s in r['suggestions'])
        )
        self.session_chart.plot(fps, rounding, hips, reps)

        weight_kg = self.weight_input.value() or None
        save_session_summary(
            session_dir, rep_results, rounding,
            hips, knees, tilts, fps, weight_kg
        )
        self._update_progress_tab(load_all_sessions(self.project_dir))

        self.statusBar().showMessage(
            f"Session: {session_name}  •  {len(reps)} reps  •  "
            f"Avg: {avg:.0f}%  •  Progress saved"
        )
        self.tabs.setCurrentIndex(0)

    # ── PROGRESS TAB ──

    def _update_progress_tab(self, sessions):
        self.progress_chart.plot(sessions)

        self.history_table.setRowCount(len(sessions))
        for i, s in enumerate(sessions):
            self.history_table.setItem(
                i, 0, QTableWidgetItem(f"  {s.get('num_reps', 0)}")
            )

            avg_s = s.get('avg_score', 0)
            si = QTableWidgetItem(f"  {avg_s:.0f}%")
            si.setForeground(QColor(
                "#50e880" if avg_s >= 80 else
                "#ffaa00" if avg_s >= 60 else "#ff5555"
            ))
            self.history_table.setItem(i, 1, si)

            rv = s.get('avg_rep_rounding', s.get('avg_rounding', s.get('max_rounding', 0)))
            ri = QTableWidgetItem(f"  {rv:.0f}")
            if rv > 40:
                ri.setForeground(QColor("#ff5555"))
            elif rv > 20:
                ri.setForeground(QColor("#ffaa00"))
            else:
                ri.setForeground(QColor("#50e880"))
            self.history_table.setItem(i, 2, ri)

            w = s.get('weight_kg')
            self.history_table.setItem(
                i, 3,
                QTableWidgetItem(f"  {w:.1f} kg" if w else "  —")
            )

        n = len(sessions)
        if n >= 2:
            f_sc = sessions[0].get('avg_score', 0)
            l_sc = sessions[-1].get('avg_score', 0)
            diff = l_sc - f_sc

            w0 = sessions[0].get('weight_kg')
            wn = sessions[-1].get('weight_kg')

            try:
                first_dt = datetime.fromisoformat(sessions[0]['timestamp']).strftime('%d %b %Y')
                last_dt = datetime.fromisoformat(sessions[-1]['timestamp']).strftime('%d %b %Y')
                date_part = f"Recorded sessions: {first_dt} → {last_dt}."
            except Exception:
                date_part = f"{n} sessions recorded."

            parts = [date_part]
            parts.append(
                f"Form: {f_sc:.0f}% → {l_sc:.0f}% "
                f"({'↑' if diff >= 0 else '↓'}{abs(diff):.0f}%)."
            )

            if w0 and wn:
                wd = wn - w0
                parts.append(
                    f"Load: {w0:.0f} kg → {wn:.0f} kg "
                    f"({'↑' if wd >= 0 else '↓'}{abs(wd):.0f} kg)."
                )

            self.progress_summary.setText("  ".join(parts))
        elif n == 1:
            try:
                dt = datetime.fromisoformat(sessions[0]['timestamp']).strftime('%d %b %Y')
                self.progress_summary.setText(
                    f"1 session recorded ({dt}). Keep training to see your trends."
                )
            except Exception:
                self.progress_summary.setText(
                    "1 session recorded. Keep training to see your trends."
                )
        else:
            self.progress_summary.setText("No sessions recorded yet.")


# ══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Helvetica", 12))
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())