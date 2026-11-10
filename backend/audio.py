"""
FacialPulse Acoustic Speech Biomarker Extraction Engine
Phase 3: Multimodal Audio-Visual Fusion

Extracts clinical acoustic biomarkers from speech recordings (e.g. during standardized reading tasks):
1. Prosodic Flatness / Monotone Pitch (F0 Mean, F0 Std, Pitch Contour Dynamics)
2. Psychomotor Speech Retardation (Pause Ratio, Avg Pause Duration, Speech-to-Pause Ratio)
3. Vocal Dynamic Compression (RMS Energy Dynamic Range)
4. Vocal Perturbation (Jitter local %, Shimmer dB, Spectral Centroid)
5. Acoustic Depression Risk Index (Calibrated clinical composite score 0-100)
"""

import io
import base64
import wave
import numpy as np
import scipy.signal
from typing import Dict, Any, Optional, Tuple, List


class AcousticAnalyzer:
    def __init__(self, target_sr: int = 16000):
        self.target_sr = target_sr
        self.frame_len_ms = 25.0    # 25 ms analysis window
        self.frame_step_ms = 10.0   # 10 ms frame hop
        self.f0_min = 70.0          # Hz (human speech lower bound)
        self.f0_max = 400.0         # Hz (human speech upper bound)

    def decode_base64_wav(self, base64_str: str) -> Tuple[np.ndarray, int]:
        """Decode base64-encoded WAV data into a normalized float32 numpy array [-1.0, 1.0]."""
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]

        raw_bytes = base64.b64decode(base64_str)
        return self.decode_wav_bytes(raw_bytes)

    def decode_wav_bytes(self, raw_bytes: bytes) -> Tuple[np.ndarray, int]:
        """Parse raw WAV bytes into audio samples and sample rate."""
        with io.BytesIO(raw_bytes) as bio:
            with wave.open(bio, "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                frames = wf.readframes(n_frames)

        if sampwidth == 2:
            data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 1:
            data = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sampwidth == 4:
            data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            data = np.frombuffer(frames, dtype=np.float32)

        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)

        return data, framerate

    def resample_if_needed(self, audio: np.ndarray, orig_sr: int) -> np.ndarray:
        """Resample audio signal to self.target_sr (16 kHz)."""
        if orig_sr == self.target_sr or len(audio) == 0:
            return audio
        num_samples = int(len(audio) * float(self.target_sr) / orig_sr)
        return scipy.signal.resample(audio, num_samples)

    def extract_f0_autocorr(self, frame: np.ndarray, sr: int) -> Tuple[float, float, float]:
        """
        Estimate fundamental frequency F0, autocorrelation peak, and amplitude using Normalized Cross-Correlation.
        Returns: (f0_hz, peak_corr, peak_amp)
        """
        energy = np.sum(frame ** 2)
        if energy < 1e-6:
            return 0.0, 0.0, 0.0

        min_lag = int(sr / self.f0_max)
        max_lag = int(sr / self.f0_min)
        if max_lag >= len(frame):
            max_lag = len(frame) - 1

        if min_lag >= max_lag:
            return 0.0, 0.0, 0.0

        # Normalized autocorrelation
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(frame)-1:]  # lags >= 0

        norm = np.sqrt(corr[0] * np.maximum(corr[0], 1e-6))
        norm_corr = corr / (norm + 1e-7)

        search_slice = norm_corr[min_lag:max_lag]
        if len(search_slice) == 0:
            return 0.0, 0.0, 0.0

        best_lag_idx = np.argmax(search_slice)
        peak_val = float(search_slice[best_lag_idx])
        best_lag = min_lag + best_lag_idx

        # Parabolic interpolation around peak for sub-sample precision
        if 0 < best_lag_idx < len(search_slice) - 1:
            alpha = search_slice[best_lag_idx - 1]
            beta = search_slice[best_lag_idx]
            gamma = search_slice[best_lag_idx + 1]
            denom = 2 * (2 * beta - alpha - gamma)
            if abs(denom) > 1e-6:
                delta = (alpha - gamma) / denom
                best_lag = best_lag + delta

        if peak_val >= 0.32:  # Voiced threshold
            f0 = sr / float(best_lag)
            if self.f0_min <= f0 <= self.f0_max:
                amp = float(np.max(np.abs(frame)))
                return f0, peak_val, amp

        return 0.0, peak_val, 0.0

    def analyze(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        Execute comprehensive acoustic analysis on normalized mono audio array.
        Returns full clinical acoustic biomarker dictionary.
        """
        if len(audio) == 0:
            return self._empty_response("No audio samples provided")

        audio_16k = self.resample_if_needed(audio, sr)
        total_samples = len(audio_16k)
        total_duration_sec = total_samples / self.target_sr

        if total_duration_sec < 0.5:
            return self._empty_response(f"Audio duration too short ({total_duration_sec:.2f}s < 0.5s)")

        # Frame parameters
        frame_len = int(self.frame_len_ms * self.target_sr / 1000.0)   # 400 samples
        frame_step = int(self.frame_step_ms * self.target_sr / 1000.0) # 160 samples
        num_frames = (total_samples - frame_len) // frame_step + 1

        if num_frames < 5:
            return self._empty_response("Insufficient frames for analysis")

        window = np.hanning(frame_len)

        # Frame extraction buffers
        rms_list: List[float] = []
        zcr_list: List[float] = []
        f0_list: List[float] = []
        voiced_frames: List[bool] = []
        spectral_centroids: List[float] = []
        cycle_periods: List[float] = []
        cycle_amps: List[float] = []

        for i in range(num_frames):
            start = i * frame_step
            end = start + frame_len
            frame = audio_16k[start:end]

            # RMS Energy
            rms = float(np.sqrt(np.mean(frame ** 2)))
            rms_list.append(rms)

            # Zero-Crossing Rate
            zcr = float(np.mean(np.abs(np.diff(np.sign(frame) >= 0))))
            zcr_list.append(zcr)

            # Windowed frame for frequency analysis
            w_frame = frame * window

            # F0 pitch & periodicity
            f0, peak_corr, peak_amp = self.extract_f0_autocorr(w_frame, self.target_sr)
            if f0 > 0.0:
                f0_list.append(f0)
                voiced_frames.append(True)
                cycle_periods.append(1.0 / f0)
                cycle_amps.append(peak_amp)
            else:
                voiced_frames.append(False)

            # Spectral Centroid (FFT)
            fft_mag = np.abs(np.fft.rfft(w_frame))
            freqs = np.fft.rfftfreq(frame_len, 1.0 / self.target_sr)
            mag_sum = np.sum(fft_mag)
            if mag_sum > 1e-6:
                centroid = float(np.sum(freqs * fft_mag) / mag_sum)
                spectral_centroids.append(centroid)

        # -------------------------------------------------------------
        # 1. Pitch & Monotonicity Biomarkers
        # -------------------------------------------------------------
        if len(f0_list) >= 4:
            f0_arr = np.array(f0_list)
            f0_mean = float(np.mean(f0_arr))
            f0_std = float(np.std(f0_arr))
            f0_p05 = float(np.percentile(f0_arr, 5))
            f0_p95 = float(np.percentile(f0_arr, 95))
            f0_range = float(f0_p95 - f0_p05)
            # Semitone dispersion: 12 * log2(F0_std / F0_mean)
            f0_semitone_dispersion = float(12.0 * np.log2(max(1.01, (f0_mean + f0_std) / f0_mean)))
        else:
            f0_mean = 0.0
            f0_std = 0.0
            f0_range = 0.0
            f0_semitone_dispersion = 0.0

        # Monotonicity Index (0 = highly dynamic melody, 100 = completely flat/monotone)
        # Clinical normal std: 25-50 Hz. Depressed flat prosody: < 15-20 Hz.
        if len(f0_list) >= 4:
            monotone_index = max(0.0, min(100.0, (1.0 - (f0_std / 40.0)) * 100.0))
        else:
            monotone_index = 50.0

        # -------------------------------------------------------------
        # 2. Psychomotor Speech Retardation & Pause Dynamics
        # -------------------------------------------------------------
        rms_arr = np.array(rms_list)
        min_rms = float(np.min(rms_arr)) if len(rms_arr) > 0 else 0.0
        max_rms = float(np.max(rms_arr)) if len(rms_arr) > 0 else 1.0
        # Adaptive silence threshold: 10% above noise floor, at least 0.005
        silence_threshold = max(0.005, min_rms + 0.10 * (max_rms - min_rms))
        
        # A frame is speech if its energy exceeds the silence threshold
        is_speech = rms_arr > silence_threshold

        # Contiguous speech/pause runs
        frame_dur_sec = self.frame_step_ms / 1000.0
        pause_durations_sec: List[float] = []
        speech_durations_sec: List[float] = []
        current_run_len = 0
        current_run_is_speech = bool(is_speech[0]) if len(is_speech) > 0 else False

        for speech_flag in is_speech:
            flag_bool = bool(speech_flag)
            if flag_bool == current_run_is_speech:
                current_run_len += 1
            else:
                dur = current_run_len * frame_dur_sec
                if current_run_is_speech:
                    speech_durations_sec.append(dur)
                else:
                    pause_durations_sec.append(dur)
                current_run_is_speech = flag_bool
                current_run_len = 1

        if current_run_len > 0:
            dur = current_run_len * frame_dur_sec
            if current_run_is_speech:
                speech_durations_sec.append(dur)
            else:
                pause_durations_sec.append(dur)

        total_speech_sec = float(sum(speech_durations_sec))
        total_pause_sec = float(sum(pause_durations_sec))
        total_calc_sec = total_speech_sec + total_pause_sec
        pause_ratio = float(total_pause_sec / max(0.1, total_calc_sec))
        speech_pause_ratio = float(total_speech_sec / max(0.05, total_pause_sec))

        avg_pause_duration_ms = float(np.mean(pause_durations_sec) * 1000.0) if pause_durations_sec else 0.0
        pause_count = len(pause_durations_sec)

        # Psychomotor Retardation Index: elevated pause ratio (> 0.40) & long pauses (> 600ms)
        pause_score = (pause_ratio / 0.50) * 50.0
        dur_score = min(50.0, (avg_pause_duration_ms / 800.0) * 50.0)
        psychomotor_retardation_index = float(max(0.0, min(100.0, pause_score + dur_score)))

        # -------------------------------------------------------------
        # 3. Vocal Perturbation (Jitter & Shimmer)
        # -------------------------------------------------------------
        # Jitter Local (%): cycle-to-cycle F0 period variation
        if len(cycle_periods) >= 6:
            p_arr = np.array(cycle_periods)
            diffs = np.abs(np.diff(p_arr))
            jitter_local_pct = float((np.mean(diffs) / np.mean(p_arr)) * 100.0)
        else:
            jitter_local_pct = 0.0

        # Shimmer Local (dB): cycle-to-cycle peak amplitude variation
        if len(cycle_amps) >= 6:
            a_arr = np.array(cycle_amps)
            valid_pairs = []
            for j in range(len(a_arr) - 1):
                if a_arr[j] > 1e-4 and a_arr[j+1] > 1e-4:
                    valid_pairs.append(abs(20.0 * np.log10(a_arr[j+1] / a_arr[j])))
            shimmer_db = float(np.mean(valid_pairs)) if valid_pairs else 0.0
        else:
            shimmer_db = 0.0

        # Perturbation index (normal jitter < 1.0%, normal shimmer < 0.8 dB)
        jitter_component = min(50.0, (jitter_local_pct / 1.5) * 50.0)
        shimmer_component = min(50.0, (shimmer_db / 1.2) * 50.0)
        vocal_strain_index = float(max(0.0, min(100.0, jitter_component + shimmer_component)))

        # -------------------------------------------------------------
        # 4. Energy Dynamic Range & Spectral Brightness
        # -------------------------------------------------------------
        voiced_rms = rms_arr[is_speech] if np.any(is_speech) else rms_arr
        rms_db = 20.0 * np.log10(np.maximum(voiced_rms, 1e-5))
        rms_dynamic_range_db = float(np.std(rms_db)) if len(rms_db) > 1 else 0.0

        # Energy Flatness / Compression Index (lower std = more monotone volume)
        energy_flatness_index = float(max(0.0, min(100.0, (1.0 - (rms_dynamic_range_db / 12.0)) * 100.0)))

        mean_spectral_centroid = float(np.mean(spectral_centroids)) if spectral_centroids else 0.0

        # -------------------------------------------------------------
        # 5. Composite Acoustic Depression Risk Index (0 - 100)
        # -------------------------------------------------------------
        # Weighted late synthesis of acoustic biomarkers:
        # 35% Monotonicity + 35% Psychomotor Pause Retardation + 15% Energy Flatness + 15% Vocal Strain
        composite_acoustic_risk = float(
            0.35 * monotone_index +
            0.35 * psychomotor_retardation_index +
            0.15 * energy_flatness_index +
            0.15 * vocal_strain_index
        )
        composite_acoustic_risk = float(max(0.0, min(100.0, composite_acoustic_risk)))

        # Qualitative acoustic assessment
        if composite_acoustic_risk < 25.0:
            acoustic_tier = "Normal Expressive Prosody"
            acoustic_severity = "Minimal / Healthy"
        elif composite_acoustic_risk < 50.0:
            acoustic_tier = "Mild Monotone Tendency"
            acoustic_severity = "Mild"
        elif composite_acoustic_risk < 72.0:
            acoustic_tier = "Moderate Psychomotor Blunting"
            acoustic_severity = "Moderate"
        else:
            acoustic_tier = "Severe Prosodic Flatness & Pause Retardation"
            acoustic_severity = "Severe"

        return {
            "success": True,
            "duration_sec": round(total_duration_sec, 2),
            "voiced_ratio": round(float(np.mean(voiced_frames)), 3),
            "pitch_metrics": {
                "f0_mean_hz": round(f0_mean, 1),
                "f0_std_hz": round(f0_std, 1),
                "f0_range_hz": round(f0_range, 1),
                "semitone_dispersion": round(f0_semitone_dispersion, 2),
                "monotone_index": round(monotone_index, 1)
            },
            "pause_metrics": {
                "total_speech_sec": round(total_speech_sec, 2),
                "total_pause_sec": round(total_pause_sec, 2),
                "pause_ratio": round(pause_ratio, 3),
                "speech_pause_ratio": round(speech_pause_ratio, 2),
                "avg_pause_duration_ms": round(avg_pause_duration_ms, 1),
                "pause_count": pause_count,
                "psychomotor_retardation_index": round(psychomotor_retardation_index, 1)
            },
            "perturbation_metrics": {
                "jitter_local_pct": round(jitter_local_pct, 2),
                "shimmer_db": round(shimmer_db, 2),
                "spectral_centroid_hz": round(mean_spectral_centroid, 1),
                "vocal_strain_index": round(vocal_strain_index, 1)
            },
            "energy_metrics": {
                "rms_dynamic_range_db": round(rms_dynamic_range_db, 1),
                "energy_flatness_index": round(energy_flatness_index, 1)
            },
            "composite_acoustic_risk": round(composite_acoustic_risk, 1),
            "acoustic_tier": acoustic_tier,
            "acoustic_severity": acoustic_severity
        }

    def _empty_response(self, reason: str) -> Dict[str, Any]:
        return {
            "success": False,
            "reason": reason,
            "duration_sec": 0.0,
            "voiced_ratio": 0.0,
            "pitch_metrics": {
                "f0_mean_hz": 0.0,
                "f0_std_hz": 0.0,
                "f0_range_hz": 0.0,
                "semitone_dispersion": 0.0,
                "monotone_index": 0.0
            },
            "pause_metrics": {
                "total_speech_sec": 0.0,
                "total_pause_sec": 0.0,
                "pause_ratio": 0.0,
                "speech_pause_ratio": 0.0,
                "avg_pause_duration_ms": 0.0,
                "pause_count": 0,
                "psychomotor_retardation_index": 0.0
            },
            "perturbation_metrics": {
                "jitter_local_pct": 0.0,
                "shimmer_db": 0.0,
                "spectral_centroid_hz": 0.0,
                "vocal_strain_index": 0.0
            },
            "energy_metrics": {
                "rms_dynamic_range_db": 0.0,
                "energy_flatness_index": 0.0
            },
            "composite_acoustic_risk": 0.0,
            "acoustic_tier": "No Voice Detected",
            "acoustic_severity": "Indeterminate"
        }
