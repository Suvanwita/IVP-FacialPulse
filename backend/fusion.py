"""
FacialPulse Multimodal Audio-Visual Fusion Engine
Phase 3: Multimodal Audio-Visual Fusion

Combines visual affect temporal biomarkers with acoustic speech biomarkers
to generate a synchronized, cross-modal clinical depression assessment:
1. Visual Biomarker Synthesis (Anhedonia, Hypomimia, Gaze Retardation, AU04 Tension)
2. Acoustic Biomarker Synthesis (Prosodic Monotonicity, Pause Dynamics, Vocal Perturbation)
3. Cross-Modal Concordance & Incongruence (Masked Depression / Affective Divergence)
4. Multi-Domain Clinical Diagnostic Tiers & Confidence Calibration
"""

from typing import Dict, Any, Optional


class MultimodalFusionEngine:
    def __init__(self):
        # Default multimodal fusion weights (visual: 0.50, acoustic: 0.40, incongruence: 0.10)
        self.w_visual = 0.50
        self.w_acoustic = 0.40
        self.w_incongruence = 0.10

    def fuse(self, visual_report: Dict[str, Any], acoustic_report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Fuse visual protocol/temporal report with acoustic speech report.
        Handles graceful degradation if acoustic data is absent or voice not detected.
        """
        # 1. Extract visual score
        vis_score = float(visual_report.get("composite_risk_score", 0.0))
        vis_subscores = visual_report.get("sub_scores", {})
        anhedonia = float(vis_subscores.get("anhedonia", 0.0))
        tension = float(vis_subscores.get("cognitive_tension", 0.0))
        recovery = float(vis_subscores.get("recovery_impairment", 0.0))

        # Check if acoustic analysis succeeded
        has_audio = (
            acoustic_report is not None and
            acoustic_report.get("success", False) and
            acoustic_report.get("duration_sec", 0.0) >= 0.5
        )

        if not has_audio:
            # Fallback to visual-only mode
            return {
                "mode": "visual_only",
                "multimodal_composite_score": round(vis_score, 1),
                "visual_score": round(vis_score, 1),
                "acoustic_score": None,
                "concordance_rate": None,
                "affective_congruence": "Visual Only (No Voice Track)",
                "symptom_domains": {
                    "affective_flatness": round((anhedonia * 0.6 + recovery * 0.4), 1),
                    "psychomotor_retardation": round(tension * 0.7, 1),
                    "anhedonia": round(anhedonia, 1),
                    "cognitive_tension": round(tension, 1)
                },
                "confidence": {
                    "level": "Moderate",
                    "reason": "Single modality (facial visual tracking only); acoustic speech data was not captured."
                },
                "clinical_narrative": visual_report.get("clinical_narrative", ""),
                "severity_tier": visual_report.get("severity_tier", "Indeterminate")
            }

        # 2. Extract acoustic score
        ac_score = float(acoustic_report.get("composite_acoustic_risk", 0.0))
        ac_pitch = acoustic_report.get("pitch_metrics", {})
        ac_pause = acoustic_report.get("pause_metrics", {})
        ac_pert = acoustic_report.get("perturbation_metrics", {})
        ac_energy = acoustic_report.get("energy_metrics", {})

        monotone_idx = float(ac_pitch.get("monotone_index", 50.0))
        pause_retardation_idx = float(ac_pause.get("psychomotor_retardation_index", 50.0))
        vocal_strain_idx = float(ac_pert.get("vocal_strain_index", 20.0))
        energy_flatness_idx = float(ac_energy.get("energy_flatness_index", 30.0))

        # 3. Cross-Modal Concordance & Incongruence
        score_diff = abs(vis_score - ac_score)
        concordance_pct = max(0.0, min(100.0, (1.0 - (score_diff / 100.0)) * 100.0))

        # Detect Masked Affect / Incongruence
        if score_diff > 35.0:
            if vis_score < ac_score:
                affective_congruence = "Affective Incongruence: Masked Facial Presentation (Speech displays pronounced psychomotor blunting despite responsive facial dynamics)"
            else:
                affective_congruence = "Affective Incongruence: Expressive Facial Blunting (Facial affect is blunted while vocal prosody retains expressive range)"
            incongruence_penalty = min(20.0, (score_diff - 35.0) * 0.5)
        else:
            affective_congruence = "Concordant Affect (Facial dynamics and vocal acoustics corroborate clinical assessment)"
            incongruence_penalty = 0.0

        # 4. Multimodal Composite Score
        raw_composite = (
            self.w_visual * vis_score +
            self.w_acoustic * ac_score +
            self.w_incongruence * incongruence_penalty
        )
        multimodal_score = max(0.0, min(100.0, raw_composite))

        # 5. Symptom Domain Synthesis (Cross-modal combinations)
        # Domain 1: Affective Flatness / Blunting (Visual Hypomimia + Acoustic Monotone Prosody)
        domain_flatness = 0.45 * anhedonia + 0.35 * monotone_idx + 0.20 * energy_flatness_idx

        # Domain 2: Psychomotor Retardation (Gaze / Head Downcast + Speech Pause Latency)
        domain_retardation = 0.45 * tension + 0.55 * pause_retardation_idx

        # Domain 3: Anhedonia & Reduced Reactivity (Blunted Smile Peak + Flat Voice Dynamic)
        domain_anhedonia = 0.60 * anhedonia + 0.40 * (100.0 - float(ac_pitch.get("f0_range_hz", 40.0)))

        # Domain 4: Stress & Affective Tension (AU04 Brow Furrow + Vocal Jitter/Shimmer)
        domain_tension = 0.60 * tension + 0.40 * vocal_strain_idx

        # Severity categorization
        if multimodal_score < 25.0:
            severity = "Minimal / No Depressive Indications"
        elif multimodal_score < 50.0:
            severity = "Mild Affective Blunting"
        elif multimodal_score < 72.0:
            severity = "Moderate Depressive Symptomatology"
        else:
            severity = "Moderately Severe to Severe Indication"

        # Confidence level
        if concordance_pct >= 75.0 and acoustic_report.get("duration_sec", 0.0) >= 3.0:
            conf_level = "High"
            conf_reason = f"Strong audio-visual cross-modal concordance ({concordance_pct:.0f}%) across synchronized behavioral tracks."
        elif concordance_pct >= 55.0:
            conf_level = "Moderate"
            conf_reason = f"Moderate audio-visual concordance ({concordance_pct:.0f}%). Visual and acoustic signals are generally aligned."
        else:
            conf_level = "Guarded"
            conf_reason = f"Cross-modal divergence detected ({score_diff:.0f} pt difference). Patient may exhibit masked affect or situational vocal tension."

        # Multimodal clinical narrative
        narrative = (
            f"Multimodal assessment combined 4-phase facial analysis (Visual Risk: {vis_score:.1f}) "
            f"with clinical acoustic speech prosody (Acoustic Risk: {ac_score:.1f}). "
            f"Overall Cross-Modal Concordance is {concordance_pct:.0f}%. "
            f"Symptom profiling indicates {severity.lower()} with "
            f"Flatness Index at {domain_flatness:.1f}/100 and Psychomotor Pause Retardation at {domain_retardation:.1f}/100."
        )

        return {
            "mode": "multimodal_fusion",
            "multimodal_composite_score": round(multimodal_score, 1),
            "visual_score": round(vis_score, 1),
            "acoustic_score": round(ac_score, 1),
            "concordance_rate": round(concordance_pct, 1),
            "affective_congruence": affective_congruence,
            "symptom_domains": {
                "affective_flatness": round(domain_flatness, 1),
                "psychomotor_retardation": round(domain_retardation, 1),
                "anhedonia": round(domain_anhedonia, 1),
                "cognitive_tension": round(domain_tension, 1)
            },
            "acoustic_breakdown": {
                "monotone_prosody_index": round(monotone_idx, 1),
                "pause_retardation_index": round(pause_retardation_idx, 1),
                "vocal_strain_index": round(vocal_strain_idx, 1),
                "energy_flatness_index": round(energy_flatness_idx, 1),
                "f0_mean_hz": ac_pitch.get("f0_mean_hz", 0.0),
                "f0_std_hz": ac_pitch.get("f0_std_hz", 0.0),
                "pause_ratio": ac_pause.get("pause_ratio", 0.0),
                "avg_pause_duration_ms": ac_pause.get("avg_pause_duration_ms", 0.0),
                "speech_pause_ratio": ac_pause.get("speech_pause_ratio", 0.0),
                "duration_sec": acoustic_report.get("duration_sec", 0.0)
            },
            "confidence": {
                "level": conf_level,
                "reason": conf_reason
            },
            "severity_tier": severity,
            "clinical_narrative": narrative
        }
