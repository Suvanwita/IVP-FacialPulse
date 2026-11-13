import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
BACKEND_DIR = REPO_ROOT / "09_Source_Code" / "backend"
INPUT_CSV = HERE / "synthetic_cases.csv"
OUTPUT_CSV = HERE / "synthetic_results.csv"
ANALYSIS_MD = HERE / "analysis.md"

if not BACKEND_DIR.is_dir():
    raise SystemExit(f"Backend folder not found: {BACKEND_DIR}")
if not INPUT_CSV.is_file():
    raise SystemExit(f"Input CSV not found: {INPUT_CSV}")

sys.path.insert(0, str(BACKEND_DIR))
from ml_scoring import MLDepressionScorer  # noqa: E402


def as_bool(value):
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def expected_band_bounds(band):
    low, high = band.split("-", 1)
    return int(low), int(high)


def make_acoustic_report(row):
    if not as_bool(row["has_audio"]):
        return None

    return {
        "success": True,
        "pitch_metrics": {
            "f0_mean_hz": float(row["f0_mean_hz"]),
            "f0_std_hz": float(row["f0_std_hz"]),
            "monotone_index": float(row["monotone_index"]),
        },
        "pause_metrics": {
            "pause_ratio": float(row["pause_ratio"]),
            "avg_pause_duration_ms": float(row["avg_pause_duration_ms"]),
            "speech_pause_ratio": float(row["speech_pause_ratio"]),
        },
        "perturbation_metrics": {
            "jitter_local_pct": float(row["jitter_local_pct"]),
            "shimmer_db": float(row["shimmer_db"]),
            "vocal_strain_index": float(row["vocal_strain_index"]),
        },
        "energy_metrics": {
            "energy_flatness_index": float(row["energy_flatness_index"]),
        },
    }


def make_protocol_report(row):
    return {
        "phase_metrics": {
            "positive": {
                "smile_peak": float(row["smile_peak"]),
                "smile_mean": float(row["smile_mean"]),
                "smile_latency_sec": float(row["smile_latency_sec"]),
                "duchenne_pct": float(row["duchenne_pct"]),
            },
            "cognitive": {
                "head_down_pct": float(row["head_down_pct"]),
                "brow_tension_pct": float(row["brow_tension_pct"]),
                "psychomotor_speed": float(row["psychomotor_speed"]),
            },
            "recovery": {
                "residual_brow_diff": float(row["residual_brow_diff"]),
                "residual_sadness_diff": float(row["residual_sadness_diff"]),
            },
        },
        "subscores": {
            "anhedonia_score": float(row["anhedonia_score"]),
            "recovery_impairment_score": float(row["recovery_impairment_score"]),
        },
    }


def severity_from_score(score):
    if score <= 4:
        return "Minimal / Normal"
    elif score <= 9:
        return "Mild Concern"
    elif score <= 14:
        return "Moderate Concern"
    else:
        return "Elevated Risk"


def valid_output(pred):
    score = pred["phq8_score"]
    ci = pred["confidence_interval_95"]
    return (
        0 <= score <= 24
        and 0 <= ci["lower"] <= ci["upper"] <= 24
        and all(0 <= item["score"] <= 3 for item in pred["item_breakdown"].values())
        and len(pred["top_contributors"]) > 0
    )


def main():
    scorer = MLDepressionScorer()
    results = []

    with INPUT_CSV.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        report = make_protocol_report(row)
        acoustic_report = make_acoustic_report(row)

        feature_vector = scorer.extract_feature_vector(
            report,
            acoustic_report=acoustic_report,
        )

        pred = scorer.predict(
            feature_vector,
            has_audio=as_bool(row["has_audio"]),
        )

        expected_low, expected_high = expected_band_bounds(row["design_band"])
        pred_score = int(pred["phq8_score"])
        pred_sev = pred["severity_tier"]

        results.append({
            "session_id": row["session_id"],
            "input_band": row["design_band"],
            "has_audio": row["has_audio"],
            "phq8_score": pred_score,
            "severity_tier": pred_sev,
            "clinical_threshold_met": pred["clinical_threshold_met"],
            "ci95_lower": pred["confidence_interval_95"]["lower"],
            "ci95_upper": pred["confidence_interval_95"]["upper"],
            "in_expected_band": expected_low <= pred_score <= expected_high,
            "output_valid": valid_output(pred),
            "top_contributors": "; ".join(item["feature"] for item in pred["top_contributors"]),
        })

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "session_id",
                "input_band",
                "has_audio",
                "phq8_score",
                "severity_tier",
                "clinical_threshold_met",
                "ci95_lower",
                "ci95_upper",
                "in_expected_band",
                "output_valid",
                "top_contributors",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    valid_count = sum(1 for r in results if r["output_valid"])
    band_matches = sum(1 for r in results if r["in_expected_band"])
    avg_score = np.mean([r["phq8_score"] for r in results])

    md_lines = [
        "# Synthetic evaluation summary",
        "",
        "This file validates the pipeline logic with synthetic input values. It does not validate against real patient data.",
        "",
        f"- Total cases: {len(results)}",
        f"- Output structure valid: {valid_count}/{len(results)}",
        f"- Predicted score inside the synthetic expected band: {band_matches}/{len(results)}",
        f"- Average PHQ-8 score: {avg_score:.2f}",
        "",
        "| Session | Input band | Audio | PHQ-8 | Severity | 95% CI | In expected band | Output valid |",
        "|---|---|---:|---:|---|---|---:|---:|",
    ]

    for r in results:
        md_lines.append(
            f"| {r['session_id']} | {r['input_band']} | {r['has_audio']} "
            f"| {r['phq8_score']} | {r['severity_tier']} | "
            f"{r['ci95_lower']} - {r['ci95_upper']} | {r['in_expected_band']} | {r['output_valid']} |"
        )

    md_lines.extend([
        "",
        "Important:",
        "- These are synthetic engineering tests only.",
        "- They are not medical or clinical validation.",
        "- They only check if the scoring pipeline produces a valid PHQ-8-like output and range.",
    ])

    ANALYSIS_MD.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Processed {len(results)} synthetic cases")
    print(f"Valid outputs: {valid_count}/{len(results)}")
    print(f"Expected-band matches: {band_matches}/{len(results)}")
    print(f"Results CSV: {OUTPUT_CSV}")
    print(f"Analysis MD: {ANALYSIS_MD}")


if __name__ == "__main__":
    main()