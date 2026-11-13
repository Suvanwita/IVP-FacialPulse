import csv
import numpy as np
from pathlib import Path

out_dir = Path(__file__).resolve().parent
output_csv = out_dir / "synthetic_cases.csv"

rng = np.random.default_rng(42)

def build_row(session_id, band):
    if band == "0-4":
        smile_peak = rng.uniform(0.68, 0.95)
        smile_mean = rng.uniform(0.40, 0.68)
        smile_latency_sec = rng.uniform(0.8, 2.8)
        duchenne_pct = rng.uniform(65, 92)
        head_down_pct = rng.uniform(4, 16)
        brow_tension_pct = rng.uniform(5, 16)
        psychomotor_speed = rng.uniform(0.10, 0.22)
        residual_brow_diff = rng.uniform(0.00, 0.05)
        residual_sadness_diff = rng.uniform(0.00, 0.05)
        anhedonia_score = rng.uniform(5, 18)
        recovery_impairment_score = rng.uniform(2, 12)
        has_audio = False
        f0_mean_hz = rng.uniform(150, 220)
        f0_std_hz = rng.uniform(28, 58)
        monotone_index = rng.uniform(10, 35)
        pause_ratio = rng.uniform(0.10, 0.28)
        avg_pause_duration_ms = rng.uniform(220, 500)
        speech_pause_ratio = rng.uniform(2.8, 6.2)
        jitter_local_pct = rng.uniform(0.28, 0.80)
        shimmer_db = rng.uniform(0.20, 0.65)
        vocal_strain_index = rng.uniform(12, 30)
        energy_flatness_index = rng.uniform(10, 30)
    elif band == "5-9":
        smile_peak = rng.uniform(0.35, 0.68)
        smile_mean = rng.uniform(0.18, 0.42)
        smile_latency_sec = rng.uniform(2.5, 5.2)
        duchenne_pct = rng.uniform(20, 50)
        head_down_pct = rng.uniform(15, 35)
        brow_tension_pct = rng.uniform(15, 35)
        psychomotor_speed = rng.uniform(0.06, 0.12)
        residual_brow_diff = rng.uniform(0.04, 0.10)
        residual_sadness_diff = rng.uniform(0.03, 0.10)
        anhedonia_score = rng.uniform(22, 42)
        recovery_impairment_score = rng.uniform(10, 28)
        has_audio = False
        f0_mean_hz = rng.uniform(120, 210)
        f0_std_hz = rng.uniform(18, 40)
        monotone_index = rng.uniform(28, 52)
        pause_ratio = rng.uniform(0.24, 0.42)
        avg_pause_duration_ms = rng.uniform(420, 700)
        speech_pause_ratio = rng.uniform(1.6, 3.2)
        jitter_local_pct = rng.uniform(0.65, 1.2)
        shimmer_db = rng.uniform(0.45, 0.95)
        vocal_strain_index = rng.uniform(24, 48)
        energy_flatness_index = rng.uniform(25, 52)
    elif band == "10-14":
        smile_peak = rng.uniform(0.18, 0.36)
        smile_mean = rng.uniform(0.08, 0.20)
        smile_latency_sec = rng.uniform(4.8, 8.5)
        duchenne_pct = rng.uniform(8, 24)
        head_down_pct = rng.uniform(30, 52)
        brow_tension_pct = rng.uniform(30, 54)
        psychomotor_speed = rng.uniform(0.03, 0.08)
        residual_brow_diff = rng.uniform(0.08, 0.18)
        residual_sadness_diff = rng.uniform(0.07, 0.17)
        anhedonia_score = rng.uniform(46, 68)
        recovery_impairment_score = rng.uniform(30, 48)
        has_audio = True
        f0_mean_hz = rng.uniform(100, 190)
        f0_std_hz = rng.uniform(12, 28)
        monotone_index = rng.uniform(52, 72)
        pause_ratio = rng.uniform(0.38, 0.58)
        avg_pause_duration_ms = rng.uniform(650, 980)
        speech_pause_ratio = rng.uniform(0.9, 1.8)
        jitter_local_pct = rng.uniform(1.0, 1.7)
        shimmer_db = rng.uniform(0.8, 1.4)
        vocal_strain_index = rng.uniform(44, 72)
        energy_flatness_index = rng.uniform(45, 78)
    else:  # 15-24
        smile_peak = rng.uniform(0.00, 0.18)
        smile_mean = rng.uniform(0.00, 0.10)
        smile_latency_sec = rng.uniform(7.5, 12.0)
        duchenne_pct = rng.uniform(0, 10)
        head_down_pct = rng.uniform(50, 88)
        brow_tension_pct = rng.uniform(48, 86)
        psychomotor_speed = rng.uniform(0.00, 0.04)
        residual_brow_diff = rng.uniform(0.18, 0.40)
        residual_sadness_diff = rng.uniform(0.18, 0.40)
        anhedonia_score = rng.uniform(70, 95)
        recovery_impairment_score = rng.uniform(60, 90)
        has_audio = True
        f0_mean_hz = rng.uniform(90, 170)
        f0_std_hz = rng.uniform(6, 20)
        monotone_index = rng.uniform(70, 96)
        pause_ratio = rng.uniform(0.52, 0.80)
        avg_pause_duration_ms = rng.uniform(900, 1600)
        speech_pause_ratio = rng.uniform(0.3, 1.0)
        jitter_local_pct = rng.uniform(1.4, 3.2)
        shimmer_db = rng.uniform(1.1, 2.5)
        vocal_strain_index = rng.uniform(65, 95)
        energy_flatness_index = rng.uniform(70, 96)

    return {
        "session_id": session_id,
        "smile_peak": round(float(smile_peak), 3),
        "smile_mean": round(float(smile_mean), 3),
        "smile_latency_sec": round(float(smile_latency_sec), 2),
        "duchenne_pct": round(float(duchenne_pct), 2),
        "head_down_pct": round(float(head_down_pct), 2),
        "brow_tension_pct": round(float(brow_tension_pct), 2),
        "psychomotor_speed": round(float(psychomotor_speed), 3),
        "residual_brow_diff": round(float(residual_brow_diff), 3),
        "residual_sadness_diff": round(float(residual_sadness_diff), 3),
        "anhedonia_score": round(float(anhedonia_score), 2),
        "recovery_impairment_score": round(float(recovery_impairment_score), 2),
        "has_audio": str(bool(has_audio)).lower(),
        "design_band": band,
        "f0_mean_hz": round(float(f0_mean_hz), 2),
        "f0_std_hz": round(float(f0_std_hz), 2),
        "monotone_index": round(float(monotone_index), 2),
        "pause_ratio": round(float(pause_ratio), 3),
        "avg_pause_duration_ms": round(float(avg_pause_duration_ms), 2),
        "speech_pause_ratio": round(float(speech_pause_ratio), 2),
        "jitter_local_pct": round(float(jitter_local_pct), 2),
        "shimmer_db": round(float(shimmer_db), 2),
        "vocal_strain_index": round(float(vocal_strain_index), 2),
        "energy_flatness_index": round(float(energy_flatness_index), 2),
    }


bands = ["0-4", "5-9", "10-14", "15-24"]
counts = [14, 12, 12, 12]
rows = []

for band, count in zip(bands, counts):
    for i in range(1, count + 1):
        session_id = f"T{len(rows) + 1:02d}"
        rows.append(build_row(session_id, band))

with output_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "session_id",
            "smile_peak",
            "smile_mean",
            "smile_latency_sec",
            "duchenne_pct",
            "head_down_pct",
            "brow_tension_pct",
            "psychomotor_speed",
            "residual_brow_diff",
            "residual_sadness_diff",
            "anhedonia_score",
            "recovery_impairment_score",
            "has_audio",
            "design_band",
            "f0_mean_hz",
            "f0_std_hz",
            "monotone_index",
            "pause_ratio",
            "avg_pause_duration_ms",
            "speech_pause_ratio",
            "jitter_local_pct",
            "shimmer_db",
            "vocal_strain_index",
            "energy_flatness_index",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Generated {len(rows)} synthetic cases at {output_csv}")