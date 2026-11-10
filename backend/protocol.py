"""
FacialPulse Phase 2: Guided Interactive Protocol Engine
Implements a 4-Phase Affect Elicitation Screening Protocol:
1. Resting Baseline (8s) - Natural morphology & pose
2. Positive Affect / Anhedonia Elicitation (12s) - Duchenne smile latency, peak, & duration
3. Cognitive Load / Reading Task (15s) - Downward gaze dwell, AU04 brow tension, psychomotor speed
4. Affective Recovery (8s) - Return rate to baseline and negative affect carryover
"""

import time
import numpy as np
from typing import Dict, List, Optional, Any


PHASE_CONFIG = {
    "baseline": {
        "title": "Phase 1: Resting Baseline",
        "instruction": "Look naturally into the camera with a relaxed, neutral expression.",
        "duration_sec": 8,
        "prompt_type": "neutral"
    },
    "positive": {
        "title": "Phase 2: Positive Affect Elicitation",
        "instruction": "Recall a recent moment that made you genuinely laugh or smile. Feel free to smile!",
        "duration_sec": 12,
        "prompt_type": "stimulus"
    },
    "cognitive": {
        "title": "Phase 3: Standardized Reading Task",
        "instruction": "Read the following passage aloud or quietly to yourself:",
        "reading_passage": "When sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.",
        "duration_sec": 15,
        "prompt_type": "cognitive"
    },
    "recovery": {
        "title": "Phase 4: Affective Recovery",
        "instruction": "Take a gentle breath, relax your facial muscles, and rest your eyes.",
        "duration_sec": 8,
        "prompt_type": "recovery"
    }
}


class ProtocolSessionManager:
    """Manages multi-phase affective screening sessions and computes cross-phase biomarkers."""
    def __init__(self):
        self.active = False
        self.session_id: Optional[str] = None
        self.current_phase_idx = 0
        self.phases = ["baseline", "positive", "cognitive", "recovery"]
        self.phase_start_time: float = 0.0
        self.session_start_time: float = 0.0

        # Phase recorded frame buffers
        self.phase_frames: Dict[str, List[Dict[str, Any]]] = {p: [] for p in self.phases}
        self.baseline_values: Dict[str, float] = {}

    def start_session(self) -> Dict[str, Any]:
        """Start a new 4-phase guided assessment session."""
        self.active = True
        self.session_id = f"fp_{int(time.time())}"
        self.current_phase_idx = 0
        self.phase_start_time = time.time()
        self.session_start_time = self.phase_start_time
        self.phase_frames = {p: [] for p in self.phases}
        self.baseline_values = {}

        return self.get_current_phase_info()

    def get_current_phase_info(self) -> Dict[str, Any]:
        if not self.active or self.current_phase_idx >= len(self.phases):
            return {
                "active": False,
                "completed": True,
                "current_phase": None,
                "phase_index": self.current_phase_idx,
                "total_phases": len(self.phases)
            }

        phase_key = self.phases[self.current_phase_idx]
        cfg = PHASE_CONFIG[phase_key]
        elapsed = time.time() - self.phase_start_time
        remaining = max(0, int(cfg["duration_sec"] - elapsed))

        return {
            "active": True,
            "completed": False,
            "session_id": self.session_id,
            "phase_key": phase_key,
            "phase_index": self.current_phase_idx + 1,
            "total_phases": len(self.phases),
            "title": cfg["title"],
            "instruction": cfg["instruction"],
            "reading_passage": cfg.get("reading_passage"),
            "duration_sec": cfg["duration_sec"],
            "time_remaining_sec": remaining,
            "elapsed_sec": round(elapsed, 1)
        }

    def feed_frame(self, frame_features: Dict[str, Any]) -> Dict[str, Any]:
        """Record a frame into the current active phase and advance phase if duration elapsed."""
        if not self.active:
            return {"active": False, "message": "No active protocol session"}

        phase_key = self.phases[self.current_phase_idx]
        cfg = PHASE_CONFIG[phase_key]

        # Extract normalized numerical features
        aus = frame_features.get("action_units") or {}
        def get_au(key: str) -> float:
            for k, v in aus.items():
                if key in k:
                    return float(v)
            return 0.0

        head_pose = frame_features.get("head_pose") or {}
        expressions = frame_features.get("expressions") or {}
        eye = frame_features.get("eye_analysis") or {}

        record = {
            "timestamp": time.time(),
            "time_in_phase": time.time() - self.phase_start_time,
            "au01": get_au("AU01"),
            "au04": get_au("AU04"),
            "au06": get_au("AU06"),
            "au12": get_au("AU12"),
            "au15": get_au("AU15"),
            "smile": float(expressions.get("smile_probability", 0.0)),
            "pitch": float(head_pose.get("pitch", 0.0)),
            "yaw": float(head_pose.get("yaw", 0.0)),
            "eye_openness": float(eye.get("average_openness", 0.5))
        }

        self.phase_frames[phase_key].append(record)

        # Check if current phase duration has elapsed
        phase_elapsed = time.time() - self.phase_start_time
        advanced = False

        if phase_elapsed >= cfg["duration_sec"]:
            advanced = self.advance_phase()

        info = self.get_current_phase_info()
        info["phase_advanced"] = advanced
        return info

    def advance_phase(self) -> bool:
        """Advance to the next phase or finalize session."""
        phase_key = self.phases[self.current_phase_idx]

        # If concluding baseline phase, compute baseline reference values
        if phase_key == "baseline" and self.phase_frames["baseline"]:
            bf = self.phase_frames["baseline"]
            self.baseline_values = {
                "au01": float(np.median([f["au01"] for f in bf])),
                "au04": float(np.median([f["au04"] for f in bf])),
                "au06": float(np.median([f["au06"] for f in bf])),
                "au12": float(np.median([f["au12"] for f in bf])),
                "au15": float(np.median([f["au15"] for f in bf])),
                "pitch": float(np.median([f["pitch"] for f in bf])),
                "smile": float(np.median([f["smile"] for f in bf])),
                "eye_openness": float(np.median([f["eye_openness"] for f in bf]))
            }

        self.current_phase_idx += 1
        self.phase_start_time = time.time()

        if self.current_phase_idx >= len(self.phases):
            self.active = False
            return False

        return True

    def compute_assessment_report(self) -> Dict[str, Any]:
        """Compute the multi-phase clinical affect report across all 4 phases."""
        baseline_frames = self.phase_frames.get("baseline", [])
        positive_frames = self.phase_frames.get("positive", [])
        cognitive_frames = self.phase_frames.get("cognitive", [])
        recovery_frames = self.phase_frames.get("recovery", [])

        # Fallback baseline medians if baseline phase had frames
        if not self.baseline_values and baseline_frames:
            self.baseline_values = {
                "au01": float(np.median([f["au01"] for f in baseline_frames])),
                "au04": float(np.median([f["au04"] for f in baseline_frames])),
                "au06": float(np.median([f["au06"] for f in baseline_frames])),
                "au12": float(np.median([f["au12"] for f in baseline_frames])),
                "au15": float(np.median([f["au15"] for f in baseline_frames])),
                "pitch": float(np.median([f["pitch"] for f in baseline_frames])),
                "smile": float(np.median([f["smile"] for f in baseline_frames])),
                "eye_openness": float(np.median([f["eye_openness"] for f in baseline_frames]))
            }

        base = self.baseline_values or {
            "au01": 0.1, "au04": 0.1, "au06": 0.05, "au12": 0.05, "au15": 0.1, "pitch": 0.0, "smile": 0.0, "eye_openness": 0.5
        }

        # -------------------------------------------------------------
        # 1. Positive Affect & Anhedonia Analysis (Phase 2)
        # -------------------------------------------------------------
        pos_smiles = [f["smile"] for f in positive_frames] if positive_frames else [0.0]
        pos_au12 = [f["au12"] for f in positive_frames] if positive_frames else [0.0]
        pos_au06 = [f["au06"] for f in positive_frames] if positive_frames else [0.0]

        smile_peak = float(np.max(pos_smiles))
        smile_mean = float(np.mean(pos_smiles))

        # Smile latency: time to first smile >= 0.25
        smile_latency = None
        for f in positive_frames:
            if f["smile"] >= 0.25 or f["au12"] >= 0.25:
                smile_latency = round(f["time_in_phase"], 1)
                break

        # Duchenne marker (AU06 Cheek Raiser + AU12 Lip Puller)
        duchenne_hits = 0
        for f in positive_frames:
            if f["au06"] > base["au06"] + 0.15 and f["au12"] > base["au12"] + 0.15:
                duchenne_hits += 1
        duchenne_pct = round((duchenne_hits / max(1, len(positive_frames))) * 100, 1)

        # Anhedonia / Blunted Affect Score (0 = healthy responsive smile, 100 = completely blunted/anhedonic)
        anhedonia_score = 0.0
        if smile_peak < 0.15:
            anhedonia_score += 45.0
        elif smile_peak < 0.35:
            anhedonia_score += 25.0

        if duchenne_pct < 10.0:
            anhedonia_score += 35.0
        elif duchenne_pct < 25.0:
            anhedonia_score += 20.0

        if smile_latency is None or smile_latency > 6.0:
            anhedonia_score += 20.0

        anhedonia_score = min(100.0, anhedonia_score)

        # -------------------------------------------------------------
        # 2. Cognitive Task & Psychomotor Retardation Analysis (Phase 3)
        # -------------------------------------------------------------
        cog_au04 = [f["au04"] for f in cognitive_frames] if cognitive_frames else [0.0]
        cog_pitch = [f["pitch"] for f in cognitive_frames] if cognitive_frames else [0.0]

        # Downward head dwell (pitch drop relative to baseline)
        down_hits = sum(1 for p in cog_pitch if p < base["pitch"] - 8.0)
        head_down_pct = round((down_hits / max(1, len(cognitive_frames))) * 100, 1)

        # Corrugator (AU04 Brow Furrow) tension holding during reading
        brow_tension_hits = sum(1 for b in cog_au04 if b > base["au04"] + 0.20)
        brow_tension_pct = round((brow_tension_hits / max(1, len(cognitive_frames))) * 100, 1)

        # Psychomotor movement velocity (mean derivative of pitch and AUs)
        if len(cognitive_frames) >= 2:
            velocities = []
            for i in range(1, len(cognitive_frames)):
                dt = max(0.01, cognitive_frames[i]["timestamp"] - cognitive_frames[i-1]["timestamp"])
                d_au = abs(cognitive_frames[i]["au04"] - cognitive_frames[i-1]["au04"]) + \
                       abs(cognitive_frames[i]["au15"] - cognitive_frames[i-1]["au15"]) + \
                       abs(cognitive_frames[i]["au12"] - cognitive_frames[i-1]["au12"])
                velocities.append(d_au / dt)
            psychomotor_speed = float(np.mean(velocities))
        else:
            psychomotor_speed = 0.0

        # Cognitive Psychomotor & Tension Score (0 to 100)
        cognitive_burden_score = 0.0
        if head_down_pct > 50.0:
            cognitive_burden_score += 35.0
        elif head_down_pct > 25.0:
            cognitive_burden_score += 20.0

        if brow_tension_pct > 40.0:
            cognitive_burden_score += 35.0
        elif brow_tension_pct > 20.0:
            cognitive_burden_score += 20.0

        if psychomotor_speed < 0.05 and len(cognitive_frames) >= 5:
            cognitive_burden_score += 30.0  # Hypomimia / psychomotor slowing

        cognitive_burden_score = min(100.0, cognitive_burden_score)

        # -------------------------------------------------------------
        # 3. Affective Recovery & Negative Carryover (Phase 4)
        # -------------------------------------------------------------
        rec_au04 = [f["au04"] for f in recovery_frames] if recovery_frames else [0.0]
        rec_au15 = [f["au15"] for f in recovery_frames] if recovery_frames else [0.0]

        end_brow_diff = float(np.mean(rec_au04[-4:] if len(rec_au04) >= 4 else rec_au04)) - base["au04"]
        end_sadness_diff = float(np.mean(rec_au15[-4:] if len(rec_au15) >= 4 else rec_au15)) - base["au15"]

        # Recovery impairment score (0 to 100)
        recovery_impairment = 0.0
        if end_brow_diff > 0.20:
            recovery_impairment += 50.0
        elif end_brow_diff > 0.10:
            recovery_impairment += 25.0

        if end_sadness_diff > 0.20:
            recovery_impairment += 50.0
        elif end_sadness_diff > 0.10:
            recovery_impairment += 25.0

        recovery_impairment = min(100.0, recovery_impairment)

        # -------------------------------------------------------------
        # 4. Multi-Phase Composite Depression Risk Score (0 to 100)
        # -------------------------------------------------------------
        # Weights: Anhedonia (40%) + Cognitive Tension/Retardation (35%) + Recovery Impairment (25%)
        composite_score = int(round(
            anhedonia_score * 0.40 +
            cognitive_burden_score * 0.35 +
            recovery_impairment * 0.25
        ))
        composite_score = max(0, min(100, composite_score))

        # Severity Classification
        if composite_score >= 65:
            severity = "Elevated Risk"
            severity_code = "high"
            summary_text = "Signs of significant affective blunting, reduced positive emotional reactivity, and prolonged cognitive tension observed."
        elif composite_score >= 40:
            severity = "Moderate Concern"
            severity_code = "moderate"
            summary_text = "Moderate markers of positive affect deficit or persistent brow tension noted across tasks."
        elif composite_score >= 20:
            severity = "Mild Concern"
            severity_code = "mild"
            summary_text = "Minor affective holding or delayed smile onset, with generally preserved expressive agility."
        else:
            severity = "Minimal / Normal"
            severity_code = "minimal"
            summary_text = "Expressions demonstrated dynamic responsiveness, clear Duchenne activation, and prompt return to baseline."

        # Actionable Clinical Findings
        findings: List[str] = []
        if anhedonia_score >= 40:
            findings.append(f"Blunted positive reactivity: peak smile {smile_peak*100:.0f}% with low Duchenne activation ({duchenne_pct:.0f}%).")
        if head_down_pct >= 40:
            findings.append(f"Frequent downward head gaze during cognitive load ({head_down_pct:.0f}% of reading task).")
        if brow_tension_pct >= 35:
            findings.append(f"Persistent corrugator brow tension under cognitive load ({brow_tension_pct:.0f}% of task).")
        if recovery_impairment >= 40:
            findings.append(f"Delayed affective recovery with persistent facial tension after task completion.")
        if not findings:
            findings.append("Positive smile reactivity and baseline affective recovery were fully preserved.")

        # Aggregate timeline across all phases with phase markers
        full_timeline = []
        for phase_name in self.phases:
            for f in self.phase_frames[phase_name]:
                full_timeline.append({
                    "phase": phase_name,
                    "time": round(f["timestamp"] - self.session_start_time, 2),
                    "au04": round(f["au04"], 2),
                    "au15": round(f["au15"], 2),
                    "smile": round(f["smile"], 2),
                    "pitch": round(f["pitch"], 1)
                })

        return {
            "session_id": self.session_id,
            "composite_score": composite_score,
            "severity": severity,
            "severity_code": severity_code,
            "summary": summary_text,
            "clinical_findings": findings,
            "subscores": {
                "anhedonia_score": int(round(anhedonia_score)),
                "cognitive_burden_score": int(round(cognitive_burden_score)),
                "recovery_impairment_score": int(round(recovery_impairment))
            },
            "phase_metrics": {
                "baseline": {
                    "frames": len(baseline_frames),
                    "baseline_values": base
                },
                "positive": {
                    "frames": len(positive_frames),
                    "smile_peak": round(smile_peak, 2),
                    "smile_mean": round(smile_mean, 2),
                    "smile_latency_sec": smile_latency,
                    "duchenne_pct": duchenne_pct
                },
                "cognitive": {
                    "frames": len(cognitive_frames),
                    "head_down_pct": head_down_pct,
                    "brow_tension_pct": brow_tension_pct,
                    "psychomotor_speed": round(psychomotor_speed, 3)
                },
                "recovery": {
                    "frames": len(recovery_frames),
                    "residual_brow_diff": round(end_brow_diff, 2),
                    "residual_sadness_diff": round(end_sadness_diff, 2)
                }
            },
            "timeline": full_timeline
        }

    def cancel_session(self):
        """Cancel current session without generating report."""
        self.active = False
        self.current_phase_idx = 0
        self.phase_frames = {p: [] for p in self.phases}
