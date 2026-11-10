"""
FacialPulse Automated Verification Test Suite
Tests Phase 1 through Phase 5 engines end-to-end:
1. Temporal Engine & Morphological Baseline Calibration
2. Guided 4-Phase Elicitation Protocol
3. Multimodal Audio-Visual Fusion & Acoustic Biomarkers
4. Machine Learning DAIC-WOZ Calibration & PHQ-8 Regressor
5. Longitudinal Psychiatric Tracking, HIPAA De-identification & GDPR Purge
"""

import os
import sys
import time
import json
import numpy as np
import io
import wave

# Ensure backend root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from temporal import TemporalAggregator, BaselineCalibrator
from protocol import ProtocolSessionManager
from audio import AcousticAnalyzer
from fusion import MultimodalFusionEngine
from ml_scoring import MLDepressionScorer
from longitudinal import LongitudinalTracker


def create_synthetic_wav(duration_sec=3.0, sr=16000, f0=140.0) -> bytes:
    """Generate a clean synthetic speech-like vowel waveform for acoustic testing."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Fundamental frequency + harmonics + pauses
    sig = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * 2 * f0 * t) + 0.15 * np.sin(2 * np.pi * 3 * f0 * t)
    # Add a realistic 0.6s pause in the middle
    pause_start = int(sr * 1.0)
    pause_end = int(sr * 1.6)
    sig[pause_start:pause_end] *= 0.01

    int_samples = (sig * 32767.0).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int_samples.tobytes())
    return bio.getvalue()


def test_phase_1_temporal():
    print("Testing Phase 1: Temporal Aggregator & Baseline Calibrator...")
    aggregator = TemporalAggregator(window_size=30, ema_alpha=0.35)
    calibrator = BaselineCalibrator(target_frames=8)

    # Feed neutral calibration frames
    complete = False
    for _ in range(8):
        sample = {
            "AU01": 0.1, "AU04": 0.12, "AU06": 0.05, "AU12": 0.05, "AU15": 0.08,
            "smile": 0.05, "eye_openness": 0.75, "pitch": -5.0, "yaw": 0.0, "roll": 0.0
        }
        complete = calibrator.add_sample(sample)

    assert complete is True, "Baseline should be calibrated after 8 frames"
    baseline = calibrator.compute_baseline()
    aggregator.set_baseline(baseline)
    assert aggregator.baseline is not None, "Aggregator baseline should be set"

    # Feed expressive frames to test normalization
    res = None
    for _ in range(10):
        frame = {
            "action_units": {"AU01": 0.4, "AU04": 0.6, "AU06": 0.3, "AU12": 0.4, "AU15": 0.3},
            "expressions": {"smile_probability": 0.5},
            "eye_analysis": {"average_openness": 0.6},
            "head_pose": {"pitch": -12.0, "yaw": 2.0, "roll": 0.0}
        }
        res = aggregator.add_frame(frame)

    assert res["ready"] is True, "Aggregator should be ready"
    assert "affect_flatness_index" in res["metrics"]
    print(f"  ✓ Temporal metrics calculated (Affect Flatness: {res['metrics']['affect_flatness_index']}%)")


def test_phase_2_protocol():
    print("Testing Phase 2: Guided Interactive Elicitation Protocol...")
    manager = ProtocolSessionManager()
    info = manager.start_session()
    assert info["active"] is True
    assert manager.active is True
    assert manager.current_phase_idx == 0

    # Phase 1: Baseline resting
    info = manager.get_current_phase_info()
    assert info["phase_key"] == "baseline"

    # Add frames
    for _ in range(5):
        manager.feed_frame({
            "action_units": {"AU01": 0.1, "AU04": 0.1, "AU06": 0.05, "AU12": 0.05, "AU15": 0.1},
            "expressions": {"smile_probability": 0.0},
            "eye_analysis": {"average_openness": 0.6},
            "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        })

    # Advance to positive phase
    manager.advance_phase()
    assert manager.get_current_phase_info()["phase_key"] == "positive"

    # Feed smile frames
    for _ in range(5):
        manager.feed_frame({
            "action_units": {"AU01": 0.1, "AU04": 0.05, "AU06": 0.4, "AU12": 0.6, "AU15": 0.05},
            "expressions": {"smile_probability": 0.8},
            "eye_analysis": {"average_openness": 0.7},
            "head_pose": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
        })

    # Advance to cognitive phase
    manager.advance_phase()
    assert manager.get_current_phase_info()["phase_key"] == "cognitive"
    assert manager.get_current_phase_info()["reading_passage"] is not None

    # Advance to recovery phase
    manager.advance_phase()
    assert manager.get_current_phase_info()["phase_key"] == "recovery"

    report = manager.compute_assessment_report()
    assert report["session_id"] is not None
    assert "composite_score" in report
    assert "clinical_findings" in report
    print(f"  ✓ Protocol report generated (Session ID: {report['session_id']}, Visual Score: {report['composite_score']}%)")


def test_phase_3_audio_and_fusion():
    print("Testing Phase 3: Acoustic Analysis & Multimodal Fusion...")
    analyzer = AcousticAnalyzer(target_sr=16000)
    wav_bytes = create_synthetic_wav(duration_sec=3.0, sr=16000, f0=150.0)
    samples, sr = analyzer.decode_wav_bytes(wav_bytes)

    assert len(samples) > 0
    assert sr == 16000
    acoustic_report = analyzer.analyze(samples, sr)

    assert acoustic_report["success"] is True
    assert acoustic_report["duration_sec"] >= 2.9
    assert acoustic_report["pitch_metrics"]["f0_mean_hz"] > 100.0
    print(f"  ✓ Acoustic analysis verified (F0 Mean: {acoustic_report['pitch_metrics']['f0_mean_hz']:.1f} Hz, Pause Ratio: {acoustic_report['pause_metrics']['pause_ratio']:.2f})")

    # Multimodal Fusion
    fusion_engine = MultimodalFusionEngine()
    mock_visual = {
        "composite_risk_score": 35.0,
        "sub_scores": {"anhedonia": 30.0, "cognitive_tension": 25.0, "recovery_impairment": 20.0},
        "severity_tier": "Mild",
        "clinical_narrative": "Mild facial affect blunting"
    }

    fused = fusion_engine.fuse(mock_visual, acoustic_report)
    assert fused["mode"] == "multimodal_fusion"
    assert fused["multimodal_composite_score"] > 0
    assert fused["concordance_rate"] is not None
    assert "symptom_domains" in fused
    print(f"  ✓ Multimodal fusion verified (Composite: {fused['multimodal_composite_score']}%, Concordance: {fused['concordance_rate']}%)")


def test_phase_4_ml_scoring():
    print("Testing Phase 4: DAIC-WOZ / AVEC Calibrated ML Scoring Engine...")
    scorer = MLDepressionScorer()

    mock_report = {
        "phase_metrics": {
            "positive": {"smile_peak": 0.65, "smile_mean": 0.35, "smile_latency_sec": 2.0, "duchenne_pct": 50.0},
            "cognitive": {"head_down_pct": 15.0, "brow_tension_pct": 12.0, "psychomotor_speed": 0.08},
            "recovery": {"residual_brow_diff": 0.02, "residual_sadness_diff": 0.01}
        },
        "subscores": {"anhedonia_score": 25, "cognitive_burden_score": 15, "recovery_impairment_score": 10}
    }

    feat = scorer.extract_feature_vector(mock_report, acoustic_report=None)
    assert feat.shape == (20,)

    pred = scorer.predict(feat, has_audio=False)
    assert 0 <= pred["phq8_score"] <= 24
    assert "severity_tier" in pred
    assert "confidence_interval_95" in pred
    assert "item_breakdown" in pred
    assert len(pred["top_contributors"]) > 0
    print(f"  ✓ Calibrated PHQ-8 Predicted: {pred['phq8_score']}/24 ({pred['severity_tier']}), 95% CI: [{pred['confidence_interval_95']['lower']}, {pred['confidence_interval_95']['upper']}]")


def test_phase_5_longitudinal_and_privacy(tmp_path="/tmp/fp_test_longitudinal"):
    print("Testing Phase 5: Privacy, Security & Longitudinal Tracking Engine...")
    import shutil
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    tracker = LongitudinalTracker(storage_dir=tmp_path)
    subj_id = tracker.generate_pseudonym_subject_id("test_patient_42")
    assert subj_id.startswith("SUBJ-")
    assert len(subj_id) == 15

    # Session 1: Baseline Moderate
    report1 = {
        "session_id": "sess_1",
        "phq8_assessment": {"phq8_score": 14, "phq8_score_continuous": 14.1, "severity_tier": "Moderate", "clinical_threshold_met": True},
        "multimodal": {"symptom_domains": {"anhedonia": 55, "cognitive_tension": 40}}
    }
    rec1 = tracker.record_session(report1, raw_subject_hint="test_patient_42")
    assert rec1["trajectory_status"] == "Baseline Session"
    assert rec1["delta_from_previous"] == 0

    # Session 2: Improvement by 6 points (Clinical response)
    report2 = {
        "session_id": "sess_2",
        "phq8_assessment": {"phq8_score": 8, "phq8_score_continuous": 8.3, "severity_tier": "Mild", "clinical_threshold_met": False},
        "multimodal": {"symptom_domains": {"anhedonia": 20, "cognitive_tension": 15}}
    }
    rec2 = tracker.record_session(report2, raw_subject_hint="test_patient_42")
    assert rec2["delta_from_previous"] == -6
    assert "Significant Clinical Improvement" in rec2["trajectory_status"]

    # History retrieval
    history = tracker.get_subject_history(subj_id)
    assert len(history) == 2

    # HIPAA Export
    export_payload = tracker.get_export_payload(subj_id)
    assert export_payload["export_metadata"]["total_sessions"] == 2
    assert "HIPAA" in export_payload["export_metadata"]["compliance_standard"]

    # GDPR Art. 17 Purge
    purged = tracker.purge_all_data()
    assert purged == 2
    assert len(tracker.get_subject_history(subj_id)) == 0

    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)
    print("  ✓ Longitudinal tracking, HIPAA export & GDPR purge verified")


if __name__ == "__main__":
    print("==================================================")
    print("      FacialPulse Comprehensive Engine Test       ")
    print("==================================================")
    test_phase_1_temporal()
    test_phase_2_protocol()
    test_phase_3_audio_and_fusion()
    test_phase_4_ml_scoring()
    test_phase_5_longitudinal_and_privacy()
    print("==================================================")
    print("   ALL TESTS PASSED: PHASES 1 - 5 100% OPERATIONAL")
    print("==================================================")
