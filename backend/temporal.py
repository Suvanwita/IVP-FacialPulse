"""
FacialPulse Temporal Aggregation Engine
- Temporal sliding window analysis for depression screening
- Morphological baseline calibration (removes resting-face bias)
- Dwell time computation for affective indicators
- Expression variance and flat affect (psychomotor blunting) index
- EMA smoothing for real-time risk scores
"""

from collections import deque
import numpy as np
from typing import Dict, List, Optional, Any
import time


class BaselineCalibrator:
    """Accumulates resting frames to establish a user's morphological baseline."""
    def __init__(self, target_frames: int = 10):
        self.target_frames = target_frames
        self.samples: List[Dict[str, float]] = []

    def add_sample(self, sample: Dict[str, float]) -> bool:
        """Add a frame during calibration. Returns True when calibration is complete."""
        self.samples.append(sample)
        return len(self.samples) >= self.target_frames

    @property
    def progress(self) -> float:
        return min(1.0, len(self.samples) / max(1, self.target_frames))

    def compute_baseline(self) -> Dict[str, float]:
        if not self.samples:
            return {}

        keys = ["AU01", "AU04", "AU12", "AU15", "smile", "pitch", "yaw", "roll", "eye_openness"]
        baseline: Dict[str, float] = {}

        for k in keys:
            vals = [s.get(k, 0.0) for s in self.samples if k in s]
            if vals:
                baseline[k] = float(np.median(vals))
            else:
                baseline[k] = 0.0

        return baseline

    def reset(self):
        self.samples = []


class TemporalAggregator:
    """Sliding-window temporal aggregator for continuous facial behavior analysis."""
    def __init__(self, window_size: int = 30, ema_alpha: float = 0.3):
        self.window_size = window_size
        self.ema_alpha = ema_alpha
        self.history: deque = deque(maxlen=window_size)
        self.baseline: Optional[Dict[str, float]] = None
        self.smoothed_risk_score: float = 0.0

    def set_baseline(self, baseline: Dict[str, float]):
        self.baseline = baseline

    def clear_baseline(self):
        self.baseline = None

    def add_frame(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Record a frame and return updated temporal analysis.
        frame_data: {
            'action_units': {'AU01': 0.1, 'AU04': 0.3, ...},
            'expressions': {'smile_probability': 0.05},
            'head_pose': {'pitch': -12.0, 'yaw': 1.0, 'roll': 0.0},
            'eye_analysis': {'average_openness': 0.65},
            'timestamp': float (optional)
        }
        """
        ts = frame_data.get("timestamp", time.time())
        aus = frame_data.get("action_units") or {}
        expressions = frame_data.get("expressions") or {}
        head_pose = frame_data.get("head_pose") or {}
        eye = frame_data.get("eye_analysis") or {}

        # Helper to extract AU values by prefix
        def get_au(key: str) -> float:
            for k, v in aus.items():
                if key in k:
                    return float(v)
            return 0.0

        raw_au01 = get_au("AU01")
        raw_au04 = get_au("AU04")
        raw_au12 = get_au("AU12")
        raw_au15 = get_au("AU15")
        smile = float(expressions.get("smile_probability", 0.0))
        pitch = float(head_pose.get("pitch", 0.0)) if head_pose else 0.0
        yaw = float(head_pose.get("yaw", 0.0)) if head_pose else 0.0
        roll = float(head_pose.get("roll", 0.0)) if head_pose else 0.0
        eye_openness = float(eye.get("average_openness", 0.5)) if eye else 0.5

        # Baseline-subtracted offsets (if baseline calibrated)
        if self.baseline:
            adj_au01 = max(0.0, raw_au01 - self.baseline.get("AU01", 0.0))
            adj_au04 = max(0.0, raw_au04 - self.baseline.get("AU04", 0.0))
            adj_au12 = max(0.0, raw_au12 - self.baseline.get("AU12", 0.0))
            adj_au15 = max(0.0, raw_au15 - self.baseline.get("AU15", 0.0))
            adj_pitch = pitch - self.baseline.get("pitch", 0.0)
        else:
            adj_au01 = raw_au01
            adj_au04 = raw_au04
            adj_au12 = raw_au12
            adj_au15 = raw_au15
            adj_pitch = pitch

        record = {
            "timestamp": ts,
            "raw_au01": raw_au01,
            "raw_au04": raw_au04,
            "raw_au12": raw_au12,
            "raw_au15": raw_au15,
            "adj_au01": adj_au01,
            "adj_au04": adj_au04,
            "adj_au12": adj_au12,
            "adj_au15": adj_au15,
            "smile": smile,
            "pitch": pitch,
            "adj_pitch": adj_pitch,
            "yaw": yaw,
            "roll": roll,
            "eye_openness": eye_openness
        }

        self.history.append(record)
        return self.compute_metrics()

    def compute_metrics(self) -> Dict[str, Any]:
        n = len(self.history)
        if n == 0:
            return {
                "ready": False,
                "frames_count": 0,
                "score": 0,
                "smoothed_score": 0,
                "baseline_active": self.baseline is not None,
                "factors": [],
                "metrics": {},
                "timeline": []
            }

        # Arrays across window
        au04_arr = np.array([f["adj_au04"] for f in self.history])
        au15_arr = np.array([f["adj_au15"] for f in self.history])
        au01_arr = np.array([f["adj_au01"] for f in self.history])
        au12_arr = np.array([f["adj_au12"] for f in self.history])
        smile_arr = np.array([f["smile"] for f in self.history])
        pitch_arr = np.array([f["pitch"] for f in self.history])
        adj_pitch_arr = np.array([f["adj_pitch"] for f in self.history])
        eye_arr = np.array([f["eye_openness"] for f in self.history])

        # 1. Dwell Times (Percentage of observation window holding affective states)
        # Thresholds: if baseline-adjusted, >0.20 indicates active elevation; if raw, >0.35
        thresh_au = 0.20 if self.baseline else 0.35
        au04_dwell = float(np.mean(au04_arr > thresh_au) * 100)
        au15_dwell = float(np.mean(au15_arr > thresh_au) * 100)
        au01_dwell = float(np.mean(au01_arr > 0.25) * 100)

        # Head down dwell (absolute pitch < -10° or relative drop < -8°)
        if self.baseline:
            head_down_dwell = float(np.mean(adj_pitch_arr < -8.0) * 100)
        else:
            head_down_dwell = float(np.mean(pitch_arr < -10.0) * 100)

        low_eye_dwell = float(np.mean(eye_arr < 0.48) * 100)

        # 2. Dynamic Range and Flat Affect (Psychomotor Immobility)
        smile_max = float(np.max(smile_arr))
        smile_var = float(np.var(smile_arr))
        au_variance = float((np.var(au04_arr) + np.var(au15_arr) + np.var(au12_arr)) / 3.0)

        # Affect Flatness Index (0.0 to 1.0):
        # High value means immobility / very little expressive variance across observation
        variability = smile_var * 20.0 + au_variance * 10.0
        affect_flatness = float(np.clip(1.0 - variability, 0.0, 1.0))

        # 3. Kinematics (Psychomotor movement velocity)
        if n >= 2:
            au04_velocity = float(np.mean(np.abs(np.diff(au04_arr))))
            au15_velocity = float(np.mean(np.abs(np.diff(au15_arr))))
            smile_velocity = float(np.mean(np.abs(np.diff(smile_arr))))
        else:
            au04_velocity = au15_velocity = smile_velocity = 0.0

        # 4. Composite Temporal Risk Score Calculation
        raw_score = 0.0
        factors: List[str] = []

        # AU04 Brow Lowering (Worry / Tension)
        if au04_dwell >= 40.0:
            raw_score += 25.0
            factors.append(f"Persistent brow tension ({au04_dwell:.0f}% of window)")
        elif au04_dwell >= 20.0:
            raw_score += 15.0
            factors.append(f"Frequent brow furrowing ({au04_dwell:.0f}% of window)")

        # AU15 Lip Corner Depressor (Sadness)
        if au15_dwell >= 40.0:
            raw_score += 25.0
            factors.append(f"Persistent lip corner depression ({au15_dwell:.0f}% of window)")
        elif au15_dwell >= 20.0:
            raw_score += 15.0
            factors.append(f"Intermittent sadness expression ({au15_dwell:.0f}% of window)")

        # Smile Deficiency & Flat Affect
        if smile_max < 0.15:
            raw_score += 15.0
            factors.append("Absence of positive smile peaks")
        if affect_flatness >= 0.70 and n >= 5:
            raw_score += 15.0
            factors.append(f"Blunted facial affect (flatness index: {affect_flatness:.2f})")

        # Head Downward Gaze Dwell
        if head_down_dwell >= 45.0:
            raw_score += 15.0
            factors.append(f"Prolonged downward head posture ({head_down_dwell:.0f}% of window)")
        elif head_down_dwell >= 25.0:
            raw_score += 8.0
            factors.append(f"Frequent downward gaze ({head_down_dwell:.0f}% of window)")

        # Inner Brow Raise (Distress)
        if au01_dwell >= 35.0:
            raw_score += 10.0
            factors.append(f"Inner brow distress holding ({au01_dwell:.0f}% of window)")

        # Reduced Eye Openness
        if low_eye_dwell >= 50.0:
            raw_score += 10.0
            factors.append(f"Reduced eye alertness ({low_eye_dwell:.0f}% of window)")

        raw_score = float(np.clip(raw_score, 0.0, 100.0))

        # Apply Exponential Moving Average (EMA) smoothing for stable real-time display
        if n == 1:
            self.smoothed_risk_score = raw_score
        else:
            self.smoothed_risk_score = (
                self.ema_alpha * raw_score + (1.0 - self.ema_alpha) * self.smoothed_risk_score
            )

        # Timeline data points for frontend sparkline (last 20 points)
        timeline = []
        for f in list(self.history)[-20:]:
            timeline.append({
                "time": f["timestamp"],
                "au04": round(f["adj_au04"], 2),
                "au15": round(f["adj_au15"], 2),
                "smile": round(f["smile"], 2),
                "pitch": round(f["pitch"], 1)
            })

        return {
            "ready": n >= 3,
            "frames_count": n,
            "score": int(round(raw_score)),
            "smoothed_score": int(round(self.smoothed_risk_score)),
            "baseline_active": self.baseline is not None,
            "factors": factors,
            "metrics": {
                "au04_dwell_pct": round(au04_dwell, 1),
                "au15_dwell_pct": round(au15_dwell, 1),
                "au01_dwell_pct": round(au01_dwell, 1),
                "head_down_pct": round(head_down_dwell, 1),
                "low_eye_pct": round(low_eye_dwell, 1),
                "smile_max": round(smile_max, 2),
                "smile_variance": round(smile_var, 4),
                "affect_flatness_index": round(affect_flatness, 2),
                "psychomotor_velocity": round((au04_velocity + au15_velocity + smile_velocity) / 3.0, 3)
            },
            "timeline": timeline
        }

    def reset(self):
        self.history.clear()
        self.smoothed_risk_score = 0.0
