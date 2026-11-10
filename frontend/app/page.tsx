'use client';

import { useRef, useState, useCallback, useEffect } from 'react';

interface TemporalAnalysis {
  ready: boolean;
  frames_count: number;
  score: number;
  smoothed_score: number;
  baseline_active: boolean;
  factors: string[];
  metrics: {
    au04_dwell_pct: number;
    au15_dwell_pct: number;
    au01_dwell_pct: number;
    head_down_pct: number;
    low_eye_pct: number;
    smile_max: number;
    smile_variance: number;
    affect_flatness_index: number;
    psychomotor_velocity: number;
  };
  timeline: Array<{
    time: number;
    au04: number;
    au15: number;
    smile: number;
    pitch: number;
  }>;
}

interface ProtocolPhaseInfo {
  active: boolean;
  completed: boolean;
  session_id?: string;
  phase_key: string;
  phase_index: number;
  total_phases: number;
  title: string;
  instruction: string;
  reading_passage?: string | null;
  duration_sec: number;
  time_remaining_sec: number;
  elapsed_sec: number;
}

interface MultimodalFusionReport {
  mode: 'multimodal_fusion' | 'visual_only';
  multimodal_composite_score: number;
  visual_score: number;
  acoustic_score: number | null;
  concordance_rate: number | null;
  affective_congruence: string;
  symptom_domains: {
    affective_flatness: number;
    psychomotor_retardation: number;
    anhedonia: number;
    cognitive_tension: number;
  };
  acoustic_breakdown?: {
    monotone_prosody_index: number;
    pause_retardation_index: number;
    vocal_strain_index: number;
    energy_flatness_index: number;
    f0_mean_hz: number;
    f0_std_hz: number;
    pause_ratio: number;
    avg_pause_duration_ms: number;
    speech_pause_ratio: number;
    duration_sec: number;
  };
  confidence: {
    level: string;
    reason: string;
  };
  severity_tier: string;
  clinical_narrative: string;
}

interface ProtocolReport {
  session_id: string;
  composite_score: number;
  visual_composite_score?: number;
  severity: string;
  severity_code: 'minimal' | 'mild' | 'moderate' | 'high';
  summary: string;
  clinical_findings: string[];
  multimodal?: MultimodalFusionReport;
  subscores: {
    anhedonia_score: number;
    cognitive_burden_score: number;
    recovery_impairment_score: number;
  };
  phase_metrics: {
    baseline: {
      frames: number;
      baseline_values: Record<string, number>;
    };
    positive: {
      frames: number;
      smile_peak: number;
      smile_mean: number;
      smile_latency_sec: number | null;
      duchenne_pct: number;
    };
    cognitive: {
      frames: number;
      head_down_pct: number;
      brow_tension_pct: number;
      psychomotor_speed: number;
    };
    recovery: {
      frames: number;
      residual_brow_diff: number;
      residual_sadness_diff: number;
    };
  };
  timeline: Array<{
    phase: string;
    time: number;
    au04: number;
    au15: number;
    smile: number;
    pitch: number;
  }>;
}

function encodeWav(samples: Float32Array, sampleRate: number): string {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset: number, str: string) => {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // Mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true); // 16-bit
  writeString(36, 'data');
  view.setUint32(40, samples.length * 2, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return 'data:audio/wav;base64,' + btoa(binary);
}

interface AnalysisResult {
  success: boolean;
  message: string;
  data: {
    face_detected: boolean;
    bounding_box: { x: number; y: number; width: number; height: number };
    head_pose: { pitch: number; yaw: number; roll: number } | null;
    eye_analysis: { left_eye_openness: number; right_eye_openness: number; average_openness: number };
    expressions: { smile_probability: number };
    action_units: Record<string, number> | null;
    landmarks_count: number;
    temporal_analysis?: TemporalAnalysis;
  } | null;
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isAutoCapture, setIsAutoCapture] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<'checking' | 'connected' | 'disconnected'>('checking');
  const [currentTime, setCurrentTime] = useState(new Date());
  const [scanStatus, setScanStatus] = useState<'ready' | 'scanning' | 'success' | 'no-face'>('ready');

  // Baseline Calibration State
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [calibrationStep, setCalibrationStep] = useState(0);
  const [isBaselineCalibrated, setIsBaselineCalibrated] = useState(false);

  // Phase 2: Guided Interactive Protocol State
  const [protocolActive, setProtocolActive] = useState(false);
  const [protocolPhaseInfo, setProtocolPhaseInfo] = useState<ProtocolPhaseInfo | null>(null);
  const [protocolLiveFeatures, setProtocolLiveFeatures] = useState<any>(null);
  const [protocolReportData, setProtocolReportData] = useState<ProtocolReport | null>(null);

  // Phase 3: Multimodal Audio-Visual State
  const [micActive, setMicActive] = useState(false);
  const [micAvailable, setMicAvailable] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [voiceDetected, setVoiceDetected] = useState(false);
  const [freqBins, setFreqBins] = useState<number[]>(new Array(16).fill(0));

  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const audioChunksRef = useRef<Float32Array[]>([]);
  const recordingAudioRef = useRef(false);

  // Theme Colors
  const colors = {
    primary: '#1565C0',      // Medical Blue
    primaryDark: '#0D47A1',  // Deep Blue
    accent: '#E31937',       // Alert Red
    accentLight: '#FFEBEE',  // Light Red bg
    lightBg: '#E3F2FD',      // Light Blue bg
    text: '#1A237E',         // Dark text
    textLight: '#546E7A',
    white: '#FFFFFF',
    success: '#2E7D32',
    warning: '#F57C00',
    purple: '#7B1FA2',
    gold: '#FBC02D'
  };

  // Update time every second
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Check API status on mount
  useEffect(() => {
    fetch('http://localhost:8000/')
      .then(res => res.json())
      .then(data => {
        setApiStatus(data.model_loaded ? 'connected' : 'connected');
      })
      .catch(() => setApiStatus('disconnected'));

    fetch('http://localhost:8000/temporal/status')
      .then(res => res.json())
      .then(data => {
        if (data.baseline_active) {
          setIsBaselineCalibrated(true);
        }
      })
      .catch(() => {});
  }, []);

  // Auto capture interval for standard continuous monitor
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isAutoCapture && isStreaming && !isCalibrating && !protocolActive) {
      interval = setInterval(() => {
        captureAndAnalyze();
      }, 1200);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isAutoCapture, isStreaming, isCalibrating, protocolActive]);

  // Phase 2: Guided Interactive Protocol Frame Ingestion Loop
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (protocolActive && isStreaming) {
      interval = setInterval(async () => {
        if (!videoRef.current || !canvasRef.current) return;
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg', 0.8);

        try {
          const res = await fetch('http://localhost:8000/protocol/frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData })
          });
          const data = await res.json();
          if (data.success && data.phase_info) {
            setProtocolPhaseInfo(data.phase_info);
            if (data.features) {
              setProtocolLiveFeatures(data.features);
            }

            // Check if protocol session completed
            if (data.phase_info.completed) {
              if (recordingAudioRef.current && audioChunksRef.current.length > 0) {
                recordingAudioRef.current = false;
                const totalLen = audioChunksRef.current.reduce((acc, c) => acc + c.length, 0);
                const merged = new Float32Array(totalLen);
                let offset = 0;
                for (const chunk of audioChunksRef.current) {
                  merged.set(chunk, offset);
                  offset += chunk.length;
                }
                const sampleRate = audioContextRef.current?.sampleRate || 16000;
                const wavBase64 = encodeWav(merged, sampleRate);
                try {
                  await fetch('http://localhost:8000/protocol/audio', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ audio: wavBase64 })
                  });
                } catch (audioErr) {
                  console.error('Error uploading protocol audio', audioErr);
                }
              }
              const repRes = await fetch('http://localhost:8000/protocol/report', { method: 'POST' });
              const repData = await repRes.json();
              if (repData.success) {
                setProtocolReportData(repData.report);
                setIsBaselineCalibrated(true);
              }
              setProtocolActive(false);
            }
          }
        } catch (err) {
          console.error('Error feeding protocol frame', err);
        }
      }, 700);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [protocolActive, isStreaming]);

  // Phase 3: Real-time Audio Spectrum & Equalizer Meter
  useEffect(() => {
    let animId: number;
    const updateMeter = () => {
      if (analyserRef.current && micActive) {
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);

        const bins: number[] = [];
        const step = Math.max(1, Math.floor(dataArray.length / 16));
        let sum = 0;
        for (let i = 0; i < 16; i++) {
          let bSum = 0;
          for (let j = 0; j < step; j++) {
            bSum += dataArray[i * step + j] || 0;
          }
          const val = bSum / step;
          bins.push(Math.round((val / 255) * 100));
          sum += val;
        }
        setFreqBins(bins);

        const avg = sum / 16;
        setAudioLevel(Math.round((avg / 255) * 100));
        setVoiceDetected(avg > 18);
      }
      animId = requestAnimationFrame(updateMeter);
    };

    if (isStreaming && micActive) {
      animId = requestAnimationFrame(updateMeter);
    }
    return () => {
      if (animId) cancelAnimationFrame(animId);
    };
  }, [isStreaming, micActive]);

  const setupAudio = (stream: MediaStream) => {
    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      audioContextRef.current = ctx;

      const source = ctx.createMediaStreamSource(stream);
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.8;
      source.connect(analyser);
      analyserRef.current = analyser;

      const processor = ctx.createScriptProcessor(2048, 1, 1);
      source.connect(processor);
      processor.connect(ctx.destination);

      processor.onaudioprocess = (e) => {
        if (recordingAudioRef.current) {
          const input = e.inputBuffer.getChannelData(0);
          audioChunksRef.current.push(new Float32Array(input));
        }
      };
    } catch (err) {
      console.warn('Audio initialization warning:', err);
    }
  };

  const startCamera = async () => {
    try {
      let stream: MediaStream;
      let hasMic = false;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
          audio: true
        });
        hasMic = true;
      } catch (audioErr) {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' }
        });
      }

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
        setError(null);
        setScanStatus('ready');
        setMicAvailable(hasMic);
        setMicActive(hasMic);

        if (hasMic) {
          setupAudio(stream);
        }
      }
    } catch (err) {
      setError('Unable to access camera. Please grant camera permission.');
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      if (audioContextRef.current) {
        audioContextRef.current.close().catch(() => {});
        audioContextRef.current = null;
      }
      recordingAudioRef.current = false;
      audioChunksRef.current = [];
      setIsStreaming(false);
      setIsAutoCapture(false);
      setIsCalibrating(false);
      setProtocolActive(false);
      setScanStatus('ready');
      setMicActive(false);
    }
  };

  const captureAndAnalyze = useCallback(async () => {
    if (!videoRef.current || !canvasRef.current || isAnalyzing) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const imageData = canvas.toDataURL('image/jpeg', 0.8);
    setIsAnalyzing(true);
    setScanStatus('scanning');

    try {
      const response = await fetch('http://localhost:8000/analyze-base64', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
      });

      const data: AnalysisResult = await response.json();
      setResult(data);
      setError(null);

      if (data.success && data.data?.face_detected) {
        setScanStatus('success');
      } else {
        setScanStatus('no-face');
      }
    } catch (err) {
      setError('Analysis failed. Please check connection to the backend.');
      setScanStatus('ready');
    } finally {
      setIsAnalyzing(false);
    }
  }, [isAnalyzing]);

  // Phase 2 & 3: Start Guided Clinical Protocol with Multimodal Ingestion
  const startGuidedProtocol = async () => {
    if (!isStreaming) {
      await startCamera();
    }
    setIsAutoCapture(false);
    setProtocolReportData(null);
    audioChunksRef.current = [];
    recordingAudioRef.current = true;
    if (audioContextRef.current && audioContextRef.current.state === 'suspended') {
      audioContextRef.current.resume();
    }

    try {
      const res = await fetch('http://localhost:8000/protocol/start', { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        setProtocolActive(true);
        setProtocolPhaseInfo(data.phase_info);
      }
    } catch (err) {
      console.error('Failed to start protocol', err);
    }
  };

  const cancelGuidedProtocol = async () => {
    recordingAudioRef.current = false;
    audioChunksRef.current = [];
    try {
      await fetch('http://localhost:8000/protocol/cancel', { method: 'POST' });
    } catch (err) {}
    setProtocolActive(false);
    setProtocolPhaseInfo(null);
  };

  // Calibration routine (8 frames neutral face)
  const startCalibration = async () => {
    if (!isStreaming) {
      await startCamera();
    }
    setIsCalibrating(true);
    setCalibrationStep(0);
    setCalibrationProgress(0);
    setIsAutoCapture(false);

    let collected = 0;
    const interval = setInterval(async () => {
      if (!videoRef.current || !canvasRef.current) return;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      ctx.drawImage(video, 0, 0);
      const imageData = canvas.toDataURL('image/jpeg', 0.8);

      try {
        const res = await fetch('http://localhost:8000/temporal/calibrate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image: imageData })
        });
        const data = await res.json();
        if (data.success) {
          collected = data.samples_collected;
          setCalibrationStep(collected);
          setCalibrationProgress(data.progress);
          if (data.completed) {
            clearInterval(interval);
            setIsCalibrating(false);
            setIsBaselineCalibrated(true);
            await fetch('http://localhost:8000/temporal/reset', { method: 'POST' });
          }
        }
      } catch (err) {
        console.error('Calibration error', err);
      }
    }, 450);
  };

  const clearBaseline = async () => {
    try {
      await fetch('http://localhost:8000/temporal/clear-baseline', { method: 'POST' });
      setIsBaselineCalibrated(false);
      setCalibrationStep(0);
      setCalibrationProgress(0);
    } catch (err) {
      console.error('Clear baseline error', err);
    }
  };

  const resetTemporal = async () => {
    try {
      await fetch('http://localhost:8000/temporal/reset', { method: 'POST' });
      setResult(prev => {
        if (!prev || !prev.data) return prev;
        return {
          ...prev,
          data: {
            ...prev.data,
            temporal_analysis: undefined
          }
        };
      });
    } catch (err) {
      console.error('Reset temporal error', err);
    }
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('en-US', { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });
  };

  const formatTime = (date: Date) => {
    const hours = date.getHours().toString().padStart(2, '0');
    const minutes = date.getMinutes().toString().padStart(2, '0');
    const seconds = date.getSeconds().toString().padStart(2, '0');
    return { hours, minutes, seconds };
  };

  const time = formatTime(currentTime);
  const temporal = result?.data?.temporal_analysis;

  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(180deg, #F5F7FA 0%, #E8EEF5 100%)',
      padding: '24px 36px',
      fontFamily: '"Inter", system-ui, -apple-system, sans-serif'
    }}>
      {/* Header Bar */}
      <div style={{
        background: `linear-gradient(135deg, ${colors.primary} 0%, ${colors.primaryDark} 100%)`,
        borderRadius: '16px',
        padding: '18px 28px',
        marginBottom: '20px',
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        boxShadow: '0 4px 20px rgba(21, 101, 192, 0.3)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <img
            src="/logo.svg"
            alt="FacialPulse"
            style={{ height: '48px', width: '48px', objectFit: 'contain' }}
          />
          <div>
            <h1 style={{ margin: 0, fontSize: '1.5rem', fontWeight: '700', color: 'white', letterSpacing: '-0.3px' }}>
              FacialPulse
            </h1>
            <p style={{ margin: 0, fontSize: '0.85rem', color: 'rgba(255,255,255,0.85)' }}>
              Clinical Affect &amp; Stimulus-Response Depression Screening
            </p>
          </div>
        </div>
        <div style={{ textAlign: 'right', color: 'white' }}>
          <div style={{ fontSize: '1.8rem', fontWeight: '300' }}>
            {time.hours}:{time.minutes}<span style={{ fontSize: '1rem', opacity: 0.8 }}>:{time.seconds}</span>
          </div>
          <div style={{ fontSize: '0.8rem', opacity: 0.8 }}>{formatDate(currentTime)}</div>
        </div>
      </div>

      {/* Control Banner & Guided Protocol Trigger */}
      <div style={{
        maxWidth: '1500px',
        margin: '0 auto 20px',
        background: 'white',
        borderRadius: '14px',
        padding: '14px 20px',
        boxShadow: '0 2px 12px rgba(0,0,0,0.04)',
        display: 'flex',
        flexWrap: 'wrap',
        alignItems: 'center',
        justifyContent: 'space-between',
        gap: '15px'
      }}>
        {/* Left: Baseline Status */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
          <div style={{
            width: '12px',
            height: '12px',
            borderRadius: '50%',
            background: isBaselineCalibrated ? colors.success : isCalibrating ? colors.warning : '#B0BEC5'
          }} />
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '0.9rem', fontWeight: '600', color: colors.text }}>
                {isBaselineCalibrated ? 'Morphological Baseline: Active' : isCalibrating ? 'Calibrating Neutral Face...' : 'Baseline: Not Calibrated'}
              </span>
              {isBaselineCalibrated && (
                <span style={{ fontSize: '0.75rem', padding: '2px 8px', borderRadius: '10px', background: '#E8F5E9', color: colors.success, fontWeight: '600' }}>
                  Resting Bias Normalized
                </span>
              )}
            </div>
            <p style={{ margin: '2px 0 0', fontSize: '0.78rem', color: colors.textLight }}>
              {isCalibrating ? `Hold a relaxed, neutral expression... (${calibrationStep}/8 samples)` :
               isBaselineCalibrated ? 'All expressions normalized against your natural resting morphology' :
               'Recommended: Run quick calibration or take the Guided 4-Phase Protocol'}
            </p>
          </div>
        </div>

        {/* Right: Actions */}
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center', flexWrap: 'wrap' }}>
          {/* Phase 3: Microphone & Multimodal Status Pill */}
          {isStreaming && (
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              padding: '6px 12px',
              borderRadius: '20px',
              background: micActive ? '#E8F5E9' : '#ECEFF1',
              border: `1px solid ${micActive ? '#A5D6A7' : '#CFD8DC'}`,
              fontSize: '0.8rem',
              fontWeight: '600',
              color: micActive ? '#2E7D32' : colors.textLight
            }}>
              <span>🎙️</span>
              <span>{micActive ? (voiceDetected ? 'Voice Active' : 'Mic Active (Multimodal)') : 'Visual-Only Mode'}</span>
              {micActive && (
                <div style={{
                  width: '8px',
                  height: '8px',
                  borderRadius: '50%',
                  background: voiceDetected ? '#00C853' : '#81C784',
                  boxShadow: voiceDetected ? '0 0 8px #00C853' : 'none'
                }} />
              )}
            </div>
          )}

          {!isBaselineCalibrated && !isCalibrating && !protocolActive && (
            <button
              onClick={startCalibration}
              style={{
                padding: '8px 14px',
                background: colors.lightBg,
                color: colors.primary,
                border: `1.5px solid ${colors.primary}`,
                borderRadius: '8px',
                fontSize: '0.85rem',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              🎯 Quick Calibrate (4s)
            </button>
          )}

          {isBaselineCalibrated && !protocolActive && (
            <button
              onClick={clearBaseline}
              style={{
                padding: '6px 12px',
                background: '#F5F5F5',
                color: colors.textLight,
                border: '1px solid #CFD8DC',
                borderRadius: '6px',
                fontSize: '0.8rem',
                fontWeight: '500',
                cursor: 'pointer'
              }}
            >
              Reset Baseline
            </button>
          )}

          {/* Phase 2: Guided Interactive Protocol Button */}
          {!protocolActive ? (
            <button
              onClick={startGuidedProtocol}
              style={{
                padding: '10px 18px',
                background: `linear-gradient(135deg, ${colors.primary} 0%, ${colors.primaryDark} 100%)`,
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '0.9rem',
                fontWeight: '700',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '8px',
                boxShadow: '0 4px 14px rgba(21, 101, 192, 0.3)'
              }}
            >
              <span>⚡</span> Start Guided 4-Phase Screening
            </button>
          ) : (
            <button
              onClick={cancelGuidedProtocol}
              style={{
                padding: '8px 16px',
                background: colors.accentLight,
                color: colors.accent,
                border: `1px solid ${colors.accent}`,
                borderRadius: '8px',
                fontSize: '0.85rem',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              Cancel Protocol
            </button>
          )}
        </div>
      </div>

      {/* Phase 2: Active Protocol Interactive HUD (During Assessment) */}
      {protocolActive && protocolPhaseInfo && (
        <div style={{
          maxWidth: '1500px',
          margin: '0 auto 20px',
          background: 'white',
          borderRadius: '16px',
          padding: '20px 24px',
          boxShadow: '0 8px 30px rgba(0,0,0,0.1)',
          borderLeft: `6px solid ${protocolPhaseInfo.phase_key === 'positive' ? colors.gold : protocolPhaseInfo.phase_key === 'cognitive' ? colors.primary : colors.purple}`
        }}>
          {/* Phase Stepper Header */}
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <span style={{
                background: colors.primary,
                color: 'white',
                padding: '4px 10px',
                borderRadius: '20px',
                fontSize: '0.75rem',
                fontWeight: '700'
              }}>
                STEP {protocolPhaseInfo.phase_index} OF {protocolPhaseInfo.total_phases}
              </span>
              <h3 style={{ margin: 0, fontSize: '1.15rem', color: colors.text, fontWeight: '700' }}>
                {protocolPhaseInfo.title}
              </h3>
            </div>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              padding: '6px 14px',
              borderRadius: '20px',
              background: protocolPhaseInfo.time_remaining_sec <= 3 ? colors.accentLight : '#ECEFF1',
              color: protocolPhaseInfo.time_remaining_sec <= 3 ? colors.accent : colors.text,
              fontWeight: '700',
              fontSize: '0.95rem'
            }}>
              <span>⏱️</span>
              <span>{protocolPhaseInfo.time_remaining_sec}s remaining</span>
            </div>
          </div>

          {/* Stepper Progress Bar */}
          <div style={{ display: 'flex', gap: '6px', marginBottom: '16px' }}>
            {['Baseline (8s)', 'Positive Affect (12s)', 'Reading Task (15s)', 'Recovery (8s)'].map((label, idx) => (
              <div key={idx} style={{ flex: 1 }}>
                <div style={{
                  height: '6px',
                  borderRadius: '3px',
                  background: idx + 1 < protocolPhaseInfo.phase_index ? colors.success :
                              idx + 1 === protocolPhaseInfo.phase_index ? colors.primary : '#ECEFF1'
                }} />
                <div style={{ fontSize: '0.7rem', color: colors.textLight, marginTop: '4px' }}>{label}</div>
              </div>
            ))}
          </div>

          {/* Instruction & Stimulus Display */}
          <div style={{
            background: protocolPhaseInfo.phase_key === 'positive' ? '#FFFDE7' :
                        protocolPhaseInfo.phase_key === 'cognitive' ? '#F0F7FF' : '#F5F5F5',
            padding: '16px 20px',
            borderRadius: '12px',
            border: '1px solid #E0E0E0'
          }}>
            <p style={{ margin: 0, fontSize: '1rem', color: colors.text, fontWeight: '600' }}>
              {protocolPhaseInfo.instruction}
            </p>

            {/* Reading Task Passage (Step 3) with Real-Time Acoustic Monitor */}
            {protocolPhaseInfo.reading_passage && (
              <div>
                <div style={{
                  marginTop: '12px',
                  padding: '14px 18px',
                  background: 'white',
                  borderRadius: '8px',
                  borderLeft: `4px solid ${colors.primary}`,
                  fontSize: '1.05rem',
                  lineHeight: '1.6',
                  color: '#263238',
                  fontStyle: 'italic'
                }}>
                  "{protocolPhaseInfo.reading_passage}"
                </div>

                {/* Multimodal Acoustic Reading Monitor */}
                <div style={{
                  marginTop: '10px',
                  padding: '10px 16px',
                  background: voiceDetected ? '#E8F5E9' : '#ECEFF1',
                  borderRadius: '8px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  gap: '15px',
                  transition: 'background 0.2s',
                  border: `1px solid ${voiceDetected ? '#A5D6A7' : '#CFD8DC'}`
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <span style={{ fontSize: '1.2rem' }}>🎙️</span>
                    <div>
                      <div style={{ fontSize: '0.85rem', fontWeight: '700', color: voiceDetected ? '#1B5E20' : colors.text }}>
                        {voiceDetected ? 'Voice Detected — Reading Passage in Progress' : micActive ? 'Listening... Please read the passage aloud' : 'Microphone Muted (Visual-Only Mode)'}
                      </div>
                      <div style={{ fontSize: '0.72rem', color: voiceDetected ? '#2E7D32' : colors.textLight }}>
                        Capturing fundamental frequency (F0), pause-to-speech ratio, and vocal perturbation
                      </div>
                    </div>
                  </div>

                  {/* Real-time Equalizer Waveform */}
                  <div style={{ display: 'flex', alignItems: 'flex-end', gap: '3px', height: '24px' }}>
                    {freqBins.map((bin, bIdx) => (
                      <div
                        key={bIdx}
                        style={{
                          width: '4px',
                          height: `${Math.max(4, (bin / 100) * 24)}px`,
                          background: voiceDetected ? '#2E7D32' : '#90A4AE',
                          borderRadius: '2px',
                          transition: 'height 0.08s ease'
                        }}
                      />
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* Step 2 Live Smile Reactivity Feedback */}
            {protocolPhaseInfo.phase_key === 'positive' && protocolLiveFeatures && (
              <div style={{ marginTop: '10px', display: 'flex', alignItems: 'center', gap: '15px' }}>
                <span style={{ fontSize: '0.85rem', color: colors.textLight }}>Live Smile Reactivity:</span>
                <span style={{ fontSize: '1rem', fontWeight: '700', color: colors.success }}>
                  {Math.round((protocolLiveFeatures.smile_probability || 0) * 100)}%
                </span>
                <div style={{ flex: 1, maxWidth: '200px', height: '6px', background: '#E0E0E0', borderRadius: '3px', overflow: 'hidden' }}>
                  <div style={{
                    width: `${Math.min(100, (protocolLiveFeatures.smile_probability || 0) * 100)}%`,
                    height: '100%',
                    background: colors.success,
                    transition: 'width 0.2s'
                  }} />
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Phase 2 & 3: Post-Assessment Comprehensive Multimodal Clinical Report Modal */}
      {protocolReportData && (
        <div style={{
          maxWidth: '1500px',
          margin: '0 auto 25px',
          background: 'white',
          borderRadius: '18px',
          padding: '28px',
          boxShadow: '0 10px 40px rgba(0,0,0,0.12)',
          borderTop: `6px solid ${protocolReportData.composite_score > 60 ? colors.accent : protocolReportData.composite_score > 35 ? colors.warning : colors.success}`
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '20px', flexWrap: 'wrap', gap: '15px' }}>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px', flexWrap: 'wrap' }}>
                <span style={{
                  background: protocolReportData.severity_code === 'high' ? colors.accentLight : protocolReportData.severity_code === 'moderate' ? '#FFF8E1' : '#E8F5E9',
                  color: protocolReportData.severity_code === 'high' ? colors.accent : protocolReportData.severity_code === 'moderate' ? colors.warning : colors.success,
                  padding: '4px 12px',
                  borderRadius: '12px',
                  fontSize: '0.8rem',
                  fontWeight: '700'
                }}>
                  {protocolReportData.severity.toUpperCase()}
                </span>
                <span style={{
                  background: protocolReportData.multimodal?.mode === 'multimodal_fusion' ? '#E8F5E9' : '#ECEFF1',
                  color: protocolReportData.multimodal?.mode === 'multimodal_fusion' ? '#1B5E20' : colors.textLight,
                  padding: '4px 10px',
                  borderRadius: '12px',
                  fontSize: '0.75rem',
                  fontWeight: '700'
                }}>
                  {protocolReportData.multimodal?.mode === 'multimodal_fusion' ? '⚡ MULTIMODAL FUSION: ACTIVE' : 'VISUAL-ONLY TRACK'}
                </span>
                <span style={{ fontSize: '0.8rem', color: colors.textLight }}>Session ID: {protocolReportData.session_id}</span>
              </div>
              <h2 style={{ margin: '8px 0 4px', fontSize: '1.4rem', color: colors.text, fontWeight: '700' }}>
                Guided 4-Phase Affective &amp; Multimodal Screening Report
              </h2>
              <p style={{ margin: 0, fontSize: '0.9rem', color: colors.textLight }}>
                {protocolReportData.summary}
              </p>
            </div>

            <div style={{ textAlign: 'right' }}>
              <div style={{ fontSize: '2.4rem', fontWeight: '800', color: protocolReportData.composite_score > 60 ? colors.accent : protocolReportData.composite_score > 35 ? colors.warning : colors.success }}>
                {protocolReportData.composite_score}%
              </div>
              <div style={{ fontSize: '0.75rem', fontWeight: '600', color: colors.textLight }}>Composite Risk Score</div>
              {protocolReportData.multimodal?.mode === 'multimodal_fusion' && (
                <div style={{ fontSize: '0.75rem', color: colors.textLight, marginTop: '3px' }}>
                  Visual: {protocolReportData.multimodal.visual_score}% • Acoustic: {protocolReportData.multimodal.acoustic_score}%
                </div>
              )}
            </div>
          </div>

          {/* Phase 3: Cross-Modal Concordance Banner */}
          {protocolReportData.multimodal?.mode === 'multimodal_fusion' && (
            <div style={{
              background: '#F1F8E9',
              borderRadius: '12px',
              padding: '14px 18px',
              marginBottom: '22px',
              border: '1px solid #C5E1A5',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              flexWrap: 'wrap',
              gap: '12px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span style={{ fontSize: '1.4rem' }}>🔗</span>
                <div>
                  <div style={{ fontSize: '0.9rem', fontWeight: '700', color: '#33691E' }}>
                    Cross-Modal Concordance: {protocolReportData.multimodal.concordance_rate}% ({protocolReportData.multimodal.confidence.level} Confidence)
                  </div>
                  <div style={{ fontSize: '0.78rem', color: '#558B2F', marginTop: '2px' }}>
                    {protocolReportData.multimodal.affective_congruence}
                  </div>
                </div>
              </div>
              <div style={{
                fontSize: '0.75rem',
                padding: '4px 12px',
                borderRadius: '14px',
                background: '#DCEDC8',
                color: '#33691E',
                fontWeight: '700'
              }}>
                AUDIO-VISUAL SYNCHRONIZED
              </div>
            </div>
          )}

          {/* Core Diagnostic Subscores (Multimodal 4-Domain or Visual 3-Domain) */}
          {protocolReportData.multimodal?.mode === 'multimodal_fusion' ? (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(4, 1fr)',
              gap: '16px',
              marginBottom: '24px'
            }}>
              {/* Domain 1: Affective Flatness / Hypomimia */}
              <div style={{ background: '#FFFDE7', padding: '16px', borderRadius: '12px', border: '1px solid #FFF59D' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: '700', color: '#F57F17' }}>Affective Flatness</span>
                  <span style={{ fontSize: '1.2rem', fontWeight: '800', color: '#F57F17' }}>
                    {Math.round(protocolReportData.multimodal.symptom_domains.affective_flatness)}%
                  </span>
                </div>
                <p style={{ margin: '8px 0 0', fontSize: '0.78rem', color: '#5D4037', lineHeight: '1.4' }}>
                  Blunted Facial Expressivity &amp; Acoustic Monotone Prosody
                </p>
              </div>

              {/* Domain 2: Psychomotor Retardation */}
              <div style={{ background: '#E3F2FD', padding: '16px', borderRadius: '12px', border: '1px solid #BBDEFB' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: '700', color: colors.primary }}>Psychomotor Retardation</span>
                  <span style={{ fontSize: '1.2rem', fontWeight: '800', color: colors.primary }}>
                    {Math.round(protocolReportData.multimodal.symptom_domains.psychomotor_retardation)}%
                  </span>
                </div>
                <p style={{ margin: '8px 0 0', fontSize: '0.78rem', color: '#1A237E', lineHeight: '1.4' }}>
                  Head-Down Dwell ({protocolReportData.phase_metrics.cognitive.head_down_pct}%) &amp; Speech Pause Latency
                </p>
              </div>

              {/* Domain 3: Anhedonia */}
              <div style={{ background: '#FFF3E0', padding: '16px', borderRadius: '12px', border: '1px solid #FFE0B2' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: '700', color: '#E65100' }}>Anhedonia / Positive Deficit</span>
                  <span style={{ fontSize: '1.2rem', fontWeight: '800', color: '#E65100' }}>
                    {Math.round(protocolReportData.multimodal.symptom_domains.anhedonia)}%
                  </span>
                </div>
                <p style={{ margin: '8px 0 0', fontSize: '0.78rem', color: '#BF360C', lineHeight: '1.4' }}>
                  Peak Smile: {Math.round(protocolReportData.phase_metrics.positive.smile_peak * 100)}% • Duchenne: {protocolReportData.phase_metrics.positive.duchenne_pct}%
                </p>
              </div>

              {/* Domain 4: Stress & Cognitive Tension */}
              <div style={{ background: '#F3E5F5', padding: '16px', borderRadius: '12px', border: '1px solid #E1BEE7' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: '700', color: colors.purple }}>Cognitive Tension</span>
                  <span style={{ fontSize: '1.2rem', fontWeight: '800', color: colors.purple }}>
                    {Math.round(protocolReportData.multimodal.symptom_domains.cognitive_tension)}%
                  </span>
                </div>
                <p style={{ margin: '8px 0 0', fontSize: '0.78rem', color: '#4A148C', lineHeight: '1.4' }}>
                  AU04 Brow Furrow ({protocolReportData.phase_metrics.cognitive.brow_tension_pct}%) &amp; Vocal Strain
                </p>
              </div>
            </div>
          ) : (
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(3, 1fr)',
              gap: '16px',
              marginBottom: '24px'
            }}>
              {/* Subscore 1: Anhedonia */}
              <div style={{ background: '#FFFDE7', padding: '16px', borderRadius: '12px', border: '1px solid #FFF59D' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: '700', color: '#F57F17' }}>Anhedonia / Positive Deficit</span>
                  <span style={{ fontSize: '1.2rem', fontWeight: '800', color: '#F57F17' }}>
                    {protocolReportData.subscores.anhedonia_score}%
                  </span>
                </div>
                <p style={{ margin: '8px 0 0', fontSize: '0.78rem', color: '#5D4037', lineHeight: '1.4' }}>
                  Peak Smile: {Math.round(protocolReportData.phase_metrics.positive.smile_peak * 100)}% • Duchenne Marker: {protocolReportData.phase_metrics.positive.duchenne_pct}%
                  {protocolReportData.phase_metrics.positive.smile_latency_sec !== null ? ` • Latency: ${protocolReportData.phase_metrics.positive.smile_latency_sec}s` : ' • No smile peak'}
                </p>
              </div>

              {/* Subscore 2: Cognitive Psychomotor & Tension */}
              <div style={{ background: '#E3F2FD', padding: '16px', borderRadius: '12px', border: '1px solid #BBDEFB' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: '700', color: colors.primary }}>Cognitive Burden &amp; Tension</span>
                  <span style={{ fontSize: '1.2rem', fontWeight: '800', color: colors.primary }}>
                    {protocolReportData.subscores.cognitive_burden_score}%
                  </span>
                </div>
                <p style={{ margin: '8px 0 0', fontSize: '0.78rem', color: '#1A237E', lineHeight: '1.4' }}>
                  Head-Down Dwell: {protocolReportData.phase_metrics.cognitive.head_down_pct}% • Corrugator Brow Tension: {protocolReportData.phase_metrics.cognitive.brow_tension_pct}%
                </p>
              </div>

              {/* Subscore 3: Affective Recovery */}
              <div style={{ background: '#F3E5F5', padding: '16px', borderRadius: '12px', border: '1px solid #E1BEE7' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span style={{ fontSize: '0.85rem', fontWeight: '700', color: colors.purple }}>Recovery Impairment</span>
                  <span style={{ fontSize: '1.2rem', fontWeight: '800', color: colors.purple }}>
                    {protocolReportData.subscores.recovery_impairment_score}%
                  </span>
                </div>
                <p style={{ margin: '8px 0 0', fontSize: '0.78rem', color: '#4A148C', lineHeight: '1.4' }}>
                  Residual Brow Tension: +{protocolReportData.phase_metrics.recovery.residual_brow_diff} • Residual Sadness: +{protocolReportData.phase_metrics.recovery.residual_sadness_diff}
                </p>
              </div>
            </div>
          )}

          {/* Phase 3: Dedicated Acoustic Speech Biomarkers Panel */}
          {protocolReportData.multimodal?.acoustic_breakdown && (
            <div style={{
              background: '#F0F7FF',
              padding: '18px 20px',
              borderRadius: '12px',
              border: '1px solid #BBDEFB',
              marginBottom: '24px'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', flexWrap: 'wrap', gap: '8px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <span style={{ fontSize: '1.1rem' }}>🎙️</span>
                  <h4 style={{ margin: 0, fontSize: '0.95rem', color: colors.primary, fontWeight: '700' }}>
                    Acoustic Speech Biomarkers (Rainbow Passage Analysis)
                  </h4>
                </div>
                <span style={{ fontSize: '0.8rem', color: colors.textLight }}>
                  Duration: {protocolReportData.multimodal.acoustic_breakdown.duration_sec}s • Sample Rate: 16 kHz
                </span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '12px' }}>
                <div style={{ background: 'white', padding: '12px', borderRadius: '8px', border: '1px solid #E3F2FD' }}>
                  <div style={{ fontSize: '0.75rem', color: colors.textLight, fontWeight: '600' }}>Pitch Variation (F0 Std)</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: '800', color: colors.text, margin: '4px 0' }}>
                    {protocolReportData.multimodal.acoustic_breakdown.f0_std_hz} Hz
                  </div>
                  <div style={{ fontSize: '0.7rem', color: protocolReportData.multimodal.acoustic_breakdown.f0_std_hz < 20 ? colors.accent : colors.success }}>
                    {protocolReportData.multimodal.acoustic_breakdown.f0_std_hz < 20 ? 'Monotone Prosody' : 'Normal Inflection'}
                  </div>
                </div>

                <div style={{ background: 'white', padding: '12px', borderRadius: '8px', border: '1px solid #E3F2FD' }}>
                  <div style={{ fontSize: '0.75rem', color: colors.textLight, fontWeight: '600' }}>Speech Pause Ratio</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: '800', color: colors.text, margin: '4px 0' }}>
                    {Math.round(protocolReportData.multimodal.acoustic_breakdown.pause_ratio * 100)}%
                  </div>
                  <div style={{ fontSize: '0.7rem', color: protocolReportData.multimodal.acoustic_breakdown.pause_ratio > 0.45 ? colors.accent : colors.success }}>
                    {protocolReportData.multimodal.acoustic_breakdown.pause_ratio > 0.45 ? 'Elevated Silence' : 'Normal Fluency'}
                  </div>
                </div>

                <div style={{ background: 'white', padding: '12px', borderRadius: '8px', border: '1px solid #E3F2FD' }}>
                  <div style={{ fontSize: '0.75rem', color: colors.textLight, fontWeight: '600' }}>Avg Pause Duration</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: '800', color: colors.text, margin: '4px 0' }}>
                    {Math.round(protocolReportData.multimodal.acoustic_breakdown.avg_pause_duration_ms)} ms
                  </div>
                  <div style={{ fontSize: '0.7rem', color: protocolReportData.multimodal.acoustic_breakdown.avg_pause_duration_ms > 750 ? colors.warning : colors.success }}>
                    {protocolReportData.multimodal.acoustic_breakdown.avg_pause_duration_ms > 750 ? 'Prolonged Pauses' : 'Fluent Rhythm'}
                  </div>
                </div>

                <div style={{ background: 'white', padding: '12px', borderRadius: '8px', border: '1px solid #E3F2FD' }}>
                  <div style={{ fontSize: '0.75rem', color: colors.textLight, fontWeight: '600' }}>Vocal Strain / Jitter</div>
                  <div style={{ fontSize: '1.2rem', fontWeight: '800', color: colors.text, margin: '4px 0' }}>
                    {protocolReportData.multimodal.acoustic_breakdown.vocal_strain_index}%
                  </div>
                  <div style={{ fontSize: '0.7rem', color: protocolReportData.multimodal.acoustic_breakdown.vocal_strain_index > 40 ? colors.accent : colors.success }}>
                    {protocolReportData.multimodal.acoustic_breakdown.vocal_strain_index > 40 ? 'Perturbation Alert' : 'Normal Quality'}
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Clinical Findings Summary */}
          <div style={{ background: '#F8F9FA', padding: '16px 20px', borderRadius: '12px', marginBottom: '24px' }}>
            <h4 style={{ margin: '0 0 8px', fontSize: '0.9rem', color: colors.text, fontWeight: '700' }}>
              Key Affective &amp; Psychomotor Findings:
            </h4>
            <ul style={{ margin: 0, paddingLeft: '18px', fontSize: '0.85rem', color: '#37474F', lineHeight: '1.6' }}>
              {protocolReportData.clinical_findings.map((finding, idx) => (
                <li key={idx}>{finding}</li>
              ))}
            </ul>
          </div>

          {/* Phase-by-Phase Interactive Timeline Chart */}
          {protocolReportData.timeline && protocolReportData.timeline.length > 2 && (
            <div style={{ marginBottom: '20px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <h4 style={{ margin: 0, fontSize: '0.9rem', color: colors.text, fontWeight: '700' }}>
                  Multi-Phase Stimulus-Response Trajectory (Full 43s Session)
                </h4>
                <div style={{ display: 'flex', gap: '12px' }}>
                  <LegendItem color="#43A047" label="Smile Probability" />
                  <LegendItem color="#FB8C00" label="AU04 Brow Furrow" />
                  <LegendItem color="#E53935" label="AU15 Lip Depressor" />
                </div>
              </div>
              <MultiPhaseTimelineChart timeline={protocolReportData.timeline} />
            </div>
          )}

          {/* Action buttons */}
          <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '12px' }}>
            <button
              onClick={() => window.print()}
              style={{
                padding: '10px 18px',
                background: '#ECEFF1',
                color: colors.text,
                border: 'none',
                borderRadius: '8px',
                fontSize: '0.85rem',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              🖨️ Print Clinical Summary
            </button>
            <button
              onClick={() => setProtocolReportData(null)}
              style={{
                padding: '10px 20px',
                background: colors.primary,
                color: 'white',
                border: 'none',
                borderRadius: '8px',
                fontSize: '0.85rem',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              Close &amp; Return to Live Stream
            </button>
          </div>
        </div>
      )}

      {/* Main 3-Column Working Grid */}
      <div style={{
        maxWidth: '1500px',
        margin: '0 auto',
        display: 'grid',
        gridTemplateColumns: '320px 1fr 450px',
        gap: '25px',
        alignItems: 'start'
      }}>
        {/* Left Side - Instructions */}
        <div>
          <div style={{
            background: 'white',
            borderRadius: '16px',
            padding: '22px',
            boxShadow: '0 2px 15px rgba(0,0,0,0.05)',
            marginBottom: '20px',
            borderTop: `4px solid ${colors.primary}`
          }}>
            <h3 style={{
              margin: '0 0 18px',
              fontSize: '0.9rem',
              color: colors.primary,
              fontWeight: '700',
              letterSpacing: '0.5px'
            }}>
              SCREENING PROTOCOL
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <InstructionItem
                icon={<svg width="18" height="18" viewBox="0 0 24 24" fill={colors.primary}><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/></svg>}
                title="1. Resting Baseline"
                subtitle="Captures resting facial morphology to eliminate false-positive bias"
                color={colors.primary}
              />
              <InstructionItem
                icon={<svg width="18" height="18" viewBox="0 0 24 24" fill={colors.gold}><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/></svg>}
                title="2. Positive Elicitation"
                subtitle="Evaluates anhedonia & Duchenne smile latency"
                color={colors.gold}
              />
              <InstructionItem
                icon={<svg width="18" height="18" viewBox="0 0 24 24" fill={colors.primary}><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z"/></svg>}
                title="3. Cognitive Reading"
                subtitle="Assesses downward gaze dwell & corrugator brow tension"
                color={colors.primary}
              />
              <InstructionItem
                icon={<svg width="18" height="18" viewBox="0 0 24 24" fill={colors.purple}><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/></svg>}
                title="4. Affective Recovery"
                subtitle="Measures return rate to baseline & negative carryover"
                color={colors.purple}
              />
            </div>
          </div>
        </div>

        {/* Center - Camera Container */}
        <div>
          <div style={{
            background: 'white',
            borderRadius: '20px',
            padding: '25px',
            boxShadow: '0 8px 30px rgba(0,0,0,0.08)'
          }}>
            {/* Status Badge */}
            <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '18px' }}>
              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                padding: '8px 18px',
                background: protocolActive ? '#FFF8E1' :
                           scanStatus === 'success' ? '#E8F5E9' :
                           scanStatus === 'scanning' ? '#FFF8E1' :
                           scanStatus === 'no-face' ? colors.accentLight : colors.lightBg,
                borderRadius: '25px'
              }}>
                <div style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  background: protocolActive ? colors.warning :
                             scanStatus === 'success' ? colors.success :
                             scanStatus === 'scanning' ? colors.warning :
                             scanStatus === 'no-face' ? colors.accent : colors.primary,
                  animation: (scanStatus === 'scanning' || protocolActive) ? 'pulse 1s infinite' : 'none'
                }} />
                <span style={{
                  fontSize: '0.85rem',
                  fontWeight: '500',
                  color: protocolActive ? colors.warning :
                         scanStatus === 'success' ? colors.success :
                         scanStatus === 'scanning' ? colors.warning :
                         scanStatus === 'no-face' ? colors.accent : colors.primary
                }}>
                  {protocolActive && protocolPhaseInfo ? `${protocolPhaseInfo.title} (${protocolPhaseInfo.time_remaining_sec}s)` :
                   scanStatus === 'success' ? 'Face Detected & Tracking' :
                   scanStatus === 'scanning' ? 'Analyzing Expressions...' :
                   scanStatus === 'no-face' ? 'No Face Detected' :
                   'Ready to Scan'}
                </span>
              </div>
            </div>

            {/* Camera Viewport */}
            <div style={{
              position: 'relative',
              background: 'linear-gradient(145deg, #2C3E50 0%, #1A252F 100%)',
              borderRadius: '16px',
              overflow: 'hidden',
              aspectRatio: '4/3'
            }}>
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                style={{
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  transform: 'scaleX(-1)'
                }}
              />

              {/* Face Guide Oval */}
              <div style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                width: '260px',
                height: '320px',
                pointerEvents: 'none'
              }}>
                <svg width="100%" height="100%" viewBox="0 0 260 320">
                  <ellipse
                    cx="130"
                    cy="160"
                    rx="110"
                    ry="145"
                    fill="none"
                    stroke={protocolActive ? colors.gold :
                           scanStatus === 'success' ? colors.success :
                           scanStatus === 'scanning' ? colors.warning : colors.primary}
                    strokeWidth="3"
                    strokeDasharray={(scanStatus === 'scanning' || protocolActive) ? '10,5' : 'none'}
                  />
                  <path d="M 30 70 L 30 30 L 70 30" fill="none" stroke={colors.accent} strokeWidth="4" strokeLinecap="round"/>
                  <path d="M 230 70 L 230 30 L 190 30" fill="none" stroke={colors.accent} strokeWidth="4" strokeLinecap="round"/>
                  <path d="M 30 250 L 30 290 L 70 290" fill="none" stroke={colors.accent} strokeWidth="4" strokeLinecap="round"/>
                  <path d="M 230 250 L 230 290 L 190 290" fill="none" stroke={colors.accent} strokeWidth="4" strokeLinecap="round"/>
                </svg>
              </div>

              {!isStreaming && (
                <div style={{
                  position: 'absolute',
                  top: 0, left: 0, right: 0, bottom: 0,
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  background: 'linear-gradient(145deg, #34495E 0%, #2C3E50 100%)'
                }}>
                  <svg width="70" height="70" viewBox="0 0 24 24" fill="#7F8C8D" style={{ marginBottom: '15px' }}>
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-5-6c.78 2.34 2.72 4 5 4s4.22-1.66 5-4H7z"/>
                  </svg>
                  <p style={{ color: '#BDC3C7', fontSize: '1rem', margin: 0 }}>Click Start Camera to begin</p>
                </div>
              )}
            </div>

            <canvas ref={canvasRef} style={{ display: 'none' }} />

            {/* Action Buttons */}
            <div style={{ display: 'flex', gap: '10px', marginTop: '20px' }}>
              {!isStreaming ? (
                <button
                  onClick={startCamera}
                  style={{
                    flex: 1,
                    padding: '14px 20px',
                    background: `linear-gradient(135deg, ${colors.primary} 0%, ${colors.primaryDark} 100%)`,
                    color: 'white',
                    border: 'none',
                    borderRadius: '10px',
                    fontSize: '1rem',
                    fontWeight: '600',
                    cursor: 'pointer',
                    boxShadow: '0 4px 15px rgba(21, 101, 192, 0.3)'
                  }}
                >
                  Start Camera
                </button>
              ) : (
                <>
                  <button
                    onClick={captureAndAnalyze}
                    disabled={isAnalyzing || protocolActive}
                    style={{
                      flex: 1,
                      padding: '14px 20px',
                      background: isAnalyzing ? '#B0BEC5' : `linear-gradient(135deg, ${colors.primary} 0%, ${colors.primaryDark} 100%)`,
                      color: 'white',
                      border: 'none',
                      borderRadius: '10px',
                      fontSize: '1rem',
                      fontWeight: '600',
                      cursor: (isAnalyzing || protocolActive) ? 'not-allowed' : 'pointer'
                    }}
                  >
                    {isAnalyzing ? 'Analyzing...' : 'Scan Now'}
                  </button>
                  <button
                    onClick={() => setIsAutoCapture(!isAutoCapture)}
                    disabled={protocolActive}
                    style={{
                      padding: '14px 18px',
                      background: isAutoCapture ? colors.warning : '#ECEFF1',
                      color: isAutoCapture ? 'white' : colors.textLight,
                      border: 'none',
                      borderRadius: '10px',
                      fontSize: '0.95rem',
                      fontWeight: '600',
                      cursor: protocolActive ? 'not-allowed' : 'pointer'
                    }}
                  >
                    {isAutoCapture ? 'Pause Auto' : 'Auto Monitor'}
                  </button>
                  <button
                    onClick={resetTemporal}
                    title="Reset Temporal Window Buffer"
                    style={{
                      padding: '14px 14px',
                      background: '#ECEFF1',
                      color: colors.textLight,
                      border: 'none',
                      borderRadius: '10px',
                      fontSize: '0.9rem',
                      fontWeight: '600',
                      cursor: 'pointer'
                    }}
                  >
                    ↺ Reset
                  </button>
                  <button
                    onClick={stopCamera}
                    style={{
                      padding: '14px 18px',
                      background: colors.accentLight,
                      color: colors.accent,
                      border: 'none',
                      borderRadius: '10px',
                      fontSize: '0.95rem',
                      fontWeight: '600',
                      cursor: 'pointer'
                    }}
                  >
                    Stop
                  </button>
                </>
              )}
            </div>

            {error && (
              <div style={{
                marginTop: '14px',
                padding: '12px 14px',
                background: colors.accentLight,
                borderRadius: '8px',
                color: colors.accent,
                fontSize: '0.9rem'
              }}>
                {error}
              </div>
            )}
          </div>
        </div>

        {/* Right Side - Analysis & Temporal Biomarkers Results */}
        <div style={{
          background: 'white',
          borderRadius: '20px',
          padding: '24px',
          boxShadow: '0 8px 30px rgba(0,0,0,0.08)',
          maxHeight: 'calc(100vh - 120px)',
          overflowY: 'auto',
          borderTop: `4px solid ${colors.accent}`
        }}>
          <h2 style={{
            margin: '0 0 16px',
            fontSize: '1.1rem',
            color: colors.primary,
            fontWeight: '700',
            display: 'flex',
            alignItems: 'center',
            gap: '10px'
          }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill={colors.primary}>
              <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z"/>
            </svg>
            Clinical Affect Analysis
          </h2>

          {!result?.success || !result.data ? (
            <div style={{ textAlign: 'center', padding: '40px 20px', color: colors.textLight }}>
              <svg width="60" height="60" viewBox="0 0 24 24" fill="#E0E0E0" style={{ marginBottom: '15px' }}>
                <path d="M9 11.75c-.69 0-1.25.56-1.25 1.25s.56 1.25 1.25 1.25 1.25-.56 1.25-1.25-.56-1.25-1.25-1.25zm6 0c-.69 0-1.25.56-1.25 1.25s.56 1.25 1.25 1.25 1.25-.56 1.25-1.25-.56-1.25-1.25-1.25zM12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/>
              </svg>
              <p style={{ margin: 0, fontSize: '0.95rem' }}>Scan face or start guided screening to see results</p>
              <p style={{ margin: '5px 0 0', fontSize: '0.85rem' }}>Continuous stimulus-response dynamics eliminate baseline bias</p>
            </div>
          ) : (
            <>
              {/* Temporal Aggregation Depression Risk Card */}
              {temporal && (
                <div style={{
                  background: temporal.smoothed_score > 50 ? colors.accentLight : colors.lightBg,
                  borderRadius: '14px',
                  padding: '18px',
                  marginBottom: '20px',
                  borderLeft: `5px solid ${temporal.smoothed_score > 50 ? colors.accent : temporal.smoothed_score > 30 ? colors.warning : colors.primary}`
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '8px' }}>
                    <div>
                      <span style={{
                        fontSize: '0.9rem',
                        fontWeight: '700',
                        color: temporal.smoothed_score > 50 ? colors.accent : colors.primary
                      }}>
                        Temporal Biomarker Risk
                      </span>
                      <div style={{ fontSize: '0.75rem', color: colors.textLight, marginTop: '2px' }}>
                        Window: {temporal.frames_count} frames {temporal.baseline_active ? '• Normalized' : '• Raw'}
                      </div>
                    </div>
                    <div style={{ textAlign: 'right' }}>
                      <span style={{
                        fontSize: '1.6rem',
                        fontWeight: '800',
                        color: temporal.smoothed_score > 50 ? colors.accent : temporal.smoothed_score > 30 ? colors.warning : colors.primary
                      }}>
                        {temporal.smoothed_score}%
                      </span>
                      <div style={{ fontSize: '0.7rem', fontWeight: '600', color: temporal.smoothed_score > 50 ? colors.accent : colors.primary }}>
                        {temporal.smoothed_score > 55 ? 'Elevated Clinical Risk' : temporal.smoothed_score > 30 ? 'Moderate Concern' : 'Low / Minimal Risk'}
                      </div>
                    </div>
                  </div>

                  {/* Progress bar */}
                  <div style={{ height: '8px', background: '#E0E0E0', borderRadius: '4px', overflow: 'hidden', margin: '10px 0' }}>
                    <div style={{
                      width: `${temporal.smoothed_score}%`,
                      height: '100%',
                      background: temporal.smoothed_score > 50
                        ? `linear-gradient(90deg, ${colors.warning}, ${colors.accent})`
                        : `linear-gradient(90deg, ${colors.success}, ${colors.primary})`,
                      borderRadius: '4px',
                      transition: 'width 0.4s ease'
                    }} />
                  </div>

                  {/* Affect Flatness Gauge */}
                  <div style={{ marginTop: '12px', padding: '10px', background: 'white', borderRadius: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: '600', color: colors.text }}>
                      <span>Affect Flatness / Immobility Index:</span>
                      <span style={{ color: temporal.metrics.affect_flatness_index > 0.7 ? colors.accent : colors.primary }}>
                        {temporal.metrics.affect_flatness_index.toFixed(2)} / 1.00
                      </span>
                    </div>
                    <div style={{ height: '5px', background: '#ECEFF1', borderRadius: '3px', marginTop: '6px', overflow: 'hidden' }}>
                      <div style={{
                        width: `${temporal.metrics.affect_flatness_index * 100}%`,
                        height: '100%',
                        background: temporal.metrics.affect_flatness_index > 0.7 ? colors.accent : colors.primary
                      }} />
                    </div>
                  </div>

                  {/* Temporal Dwell Breakdowns */}
                  <div style={{
                    display: 'grid',
                    gridTemplateColumns: '1fr 1fr',
                    gap: '8px',
                    marginTop: '10px'
                  }}>
                    <div style={{ background: 'white', padding: '8px 10px', borderRadius: '8px', fontSize: '0.75rem' }}>
                      <div style={{ color: colors.textLight }}>Brow Tension Dwell</div>
                      <div style={{ fontWeight: '700', color: colors.text, fontSize: '0.9rem' }}>{temporal.metrics.au04_dwell_pct}%</div>
                    </div>
                    <div style={{ background: 'white', padding: '8px 10px', borderRadius: '8px', fontSize: '0.75rem' }}>
                      <div style={{ color: colors.textLight }}>Lip Depression Dwell</div>
                      <div style={{ fontWeight: '700', color: colors.text, fontSize: '0.9rem' }}>{temporal.metrics.au15_dwell_pct}%</div>
                    </div>
                    <div style={{ background: 'white', padding: '8px 10px', borderRadius: '8px', fontSize: '0.75rem' }}>
                      <div style={{ color: colors.textLight }}>Head Down Dwell</div>
                      <div style={{ fontWeight: '700', color: colors.text, fontSize: '0.9rem' }}>{temporal.metrics.head_down_pct}%</div>
                    </div>
                    <div style={{ background: 'white', padding: '8px 10px', borderRadius: '8px', fontSize: '0.75rem' }}>
                      <div style={{ color: colors.textLight }}>Peak Smile Response</div>
                      <div style={{ fontWeight: '700', color: colors.text, fontSize: '0.9rem' }}>{(temporal.metrics.smile_max * 100).toFixed(0)}%</div>
                    </div>
                  </div>

                  {temporal.factors.length > 0 && (
                    <div style={{ marginTop: '12px', fontSize: '0.78rem', color: '#5D4037', lineHeight: '1.5' }}>
                      <strong>Temporal Findings:</strong>
                      <ul style={{ margin: '4px 0 0', paddingLeft: '16px' }}>
                        {temporal.factors.map((f, i) => <li key={i}>{f}</li>)}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* Real-Time Expression Timeline Sparkline Chart */}
              {temporal && temporal.timeline && temporal.timeline.length > 1 && (
                <div style={{
                  background: '#FAFAFA',
                  border: '1px solid #ECEFF1',
                  borderRadius: '12px',
                  padding: '14px',
                  marginBottom: '20px'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                    <span style={{ fontSize: '0.82rem', fontWeight: '700', color: colors.text }}>
                      Expression Dynamics Trajectory
                    </span>
                    <span style={{ fontSize: '0.7rem', color: colors.textLight }}>
                      Last {temporal.timeline.length} updates
                    </span>
                  </div>

                  <ExpressionTimelineChart timeline={temporal.timeline} />

                  <div style={{ display: 'flex', justifyContent: 'center', gap: '12px', marginTop: '8px', flexWrap: 'wrap' }}>
                    <LegendItem color="#FB8C00" label="AU04 Brow" />
                    <LegendItem color="#E53935" label="AU15 Sadness" />
                    <LegendItem color="#43A047" label="Smile" />
                    <LegendItem color="#1E88E5" label="Pitch Down" />
                  </div>
                </div>
              )}

              {/* Head Pose Section */}
              {result.data.head_pose && (
                <ResultSection title="Head Pose" color={colors.primary}>
                  <ResultBar label="Pitch (X) - Up/Down" value={result.data.head_pose.pitch} min={-45} max={45} unit="°" color={colors.primary} />
                  <ResultBar label="Yaw (Y) - Left/Right" value={result.data.head_pose.yaw} min={-45} max={45} unit="°" color={colors.primary} />
                  <ResultBar label="Roll (Z) - Tilt" value={result.data.head_pose.roll} min={-45} max={45} unit="°" color={colors.primary} />
                </ResultSection>
              )}

              {/* Eye Analysis Section */}
              <ResultSection title="Eye Analysis" color={colors.primary}>
                <ResultBar label="Left Eye" value={result.data.eye_analysis.left_eye_openness} min={0} max={1} unit="%" multiplier={100} color={colors.primary} />
                <ResultBar label="Right Eye" value={result.data.eye_analysis.right_eye_openness} min={0} max={1} unit="%" multiplier={100} color={colors.primary} />
                <ResultBar label="Average Openness" value={result.data.eye_analysis.average_openness} min={0} max={1} unit="%" multiplier={100} color={colors.primaryDark} />
              </ResultSection>

              {/* Expression Section */}
              <ResultSection title="Expressions" color={colors.primary}>
                <ResultBar label="Smile" value={result.data.expressions.smile_probability} min={0} max={1} unit="%" multiplier={100} color={colors.success} />
              </ResultSection>

              {/* Action Units Section */}
              {result.data.action_units && (
                <ResultSection title="Action Units" color={colors.primary} subtitle="Clinical Indicators">
                  {Object.entries(result.data.action_units).map(([name, value]) => {
                    const auNumber = name.split(' - ')[0];
                    const auDesc = name.split(' - ')[1] || '';
                    const isDepression = ['AU04', 'AU15', 'AU01'].some(au => name.includes(au));
                    return (
                      <ResultBar
                        key={name}
                        label={auNumber}
                        sublabel={auDesc}
                        value={value}
                        min={0}
                        max={1}
                        unit="%"
                        multiplier={100}
                        color={isDepression && value > 0.4 ? colors.accent : colors.primary}
                        highlight={isDepression && value > 0.4}
                      />
                    );
                  })}
                </ResultSection>
              )}
            </>
          )}
        </div>
      </div>

      <style jsx global>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
        ::-webkit-scrollbar {
          width: 6px;
        }
        ::-webkit-scrollbar-track {
          background: #f1f1f1;
          border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb {
          background: #c1c1c1;
          border-radius: 3px;
        }
      `}</style>
    </div>
  );
}

// Sparkline Timeline SVG Component
function ExpressionTimelineChart({ timeline }: { timeline: Array<{ au04: number; au15: number; smile: number; pitch: number }> }) {
  const width = 380;
  const height = 90;
  const padding = 6;

  const pointsCount = timeline.length;
  if (pointsCount < 2) return null;

  const getX = (idx: number) => padding + (idx / (pointsCount - 1)) * (width - 2 * padding);
  const getY = (val: number) => height - padding - Math.min(1, Math.max(0, val)) * (height - 2 * padding);
  const normalizePitch = (pitch: number) => Math.min(1, Math.max(0, (-pitch - 5) / 20));

  const createPath = (getter: (p: typeof timeline[0]) => number) => {
    return timeline
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getY(getter(p)).toFixed(1)}`)
      .join(' ');
  };

  return (
    <div style={{ width: '100%', overflow: 'hidden' }}>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: `${height}px`, background: '#F8F9FA', borderRadius: '8px' }}>
        <line x1={padding} y1={height * 0.25} x2={width - padding} y2={height * 0.25} stroke="#E0E0E0" strokeDasharray="3 3" />
        <line x1={padding} y1={height * 0.50} x2={width - padding} y2={height * 0.50} stroke="#E0E0E0" strokeDasharray="3 3" />
        <line x1={padding} y1={height * 0.75} x2={width - padding} y2={height * 0.75} stroke="#E0E0E0" strokeDasharray="3 3" />

        <path d={createPath(p => normalizePitch(p.pitch))} fill="none" stroke="#1E88E5" strokeWidth="1.8" opacity="0.6" strokeDasharray="4 2" />
        <path d={createPath(p => p.smile)} fill="none" stroke="#43A047" strokeWidth="2.2" />
        <path d={createPath(p => p.au04)} fill="none" stroke="#FB8C00" strokeWidth="2.2" />
        <path d={createPath(p => p.au15)} fill="none" stroke="#E53935" strokeWidth="2.2" />
      </svg>
    </div>
  );
}

// Multi-Phase Timeline Chart (Full Session 43s with Phase Bands)
function MultiPhaseTimelineChart({ timeline }: { timeline: ProtocolReport['timeline'] }) {
  const width = 800;
  const height = 140;
  const padding = 10;
  const n = timeline.length;
  if (n < 2) return null;

  const totalTime = Math.max(1, timeline[n - 1].time);
  const getX = (t: number) => padding + (t / totalTime) * (width - 2 * padding);
  const getY = (val: number) => height - padding - Math.min(1, Math.max(0, val)) * (height - 2 * padding);

  const createPath = (getter: (p: typeof timeline[0]) => number) => {
    return timeline
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${getX(p.time).toFixed(1)} ${getY(getter(p)).toFixed(1)}`)
      .join(' ');
  };

  return (
    <div style={{ width: '100%', overflowX: 'auto' }}>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: `${height}px`, background: '#FAFAFA', borderRadius: '10px', border: '1px solid #ECEFF1' }}>
        {/* Phase Bands */}
        <rect x={getX(0)} y={0} width={getX(8) - getX(0)} height={height} fill="#EEEEEE" opacity="0.5" />
        <rect x={getX(8)} y={0} width={getX(20) - getX(8)} height={height} fill="#FFF9C4" opacity="0.4" />
        <rect x={getX(20)} y={0} width={getX(35) - getX(20)} height={height} fill="#E1F5FE" opacity="0.4" />
        <rect x={getX(35)} y={0} width={getX(totalTime) - getX(35)} height={height} fill="#F3E5F5" opacity="0.4" />

        {/* Phase Labels */}
        <text x={getX(1)} y={16} fontSize="10" fill="#757575" fontWeight="600">Phase 1: Baseline</text>
        <text x={getX(9)} y={16} fontSize="10" fill="#F57F17" fontWeight="600">Phase 2: Positive Stimulus</text>
        <text x={getX(21)} y={16} fontSize="10" fill="#0288D1" fontWeight="600">Phase 3: Reading Task</text>
        <text x={getX(36)} y={16} fontSize="10" fill="#7B1FA2" fontWeight="600">Phase 4: Recovery</text>

        {/* Gridlines */}
        <line x1={padding} y1={height * 0.5} x2={width - padding} y2={height * 0.5} stroke="#E0E0E0" strokeDasharray="3 3" />

        {/* Data Paths */}
        <path d={createPath(p => p.smile)} fill="none" stroke="#43A047" strokeWidth="2.4" />
        <path d={createPath(p => p.au04)} fill="none" stroke="#FB8C00" strokeWidth="2.2" />
        <path d={createPath(p => p.au15)} fill="none" stroke="#E53935" strokeWidth="2.2" />
      </svg>
    </div>
  );
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
      <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: color }} />
      <span style={{ fontSize: '0.72rem', color: '#546E7A', fontWeight: '500' }}>{label}</span>
    </div>
  );
}

function InstructionItem({ icon, title, subtitle, color }: { icon: React.ReactNode; title: string; subtitle: string; color: string }) {
  return (
    <div style={{ display: 'flex', gap: '12px', alignItems: 'flex-start' }}>
      <div style={{
        width: '36px',
        height: '36px',
        background: '#E3F2FD',
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0
      }}>
        {icon}
      </div>
      <div>
        <p style={{ margin: 0, fontWeight: '600', color: '#1A237E', fontSize: '0.95rem' }}>{title}</p>
        <p style={{ margin: '2px 0 0', color: '#546E7A', fontSize: '0.85rem' }}>{subtitle}</p>
      </div>
    </div>
  );
}

function ResultSection({ title, subtitle, color, children }: { title: string; subtitle?: string; color: string; children: React.ReactNode }) {
  return (
    <div style={{ marginBottom: '20px' }}>
      <div style={{ marginBottom: '12px' }}>
        <h4 style={{ margin: 0, fontSize: '0.9rem', color: color, fontWeight: '600' }}>{title}</h4>
        {subtitle && <span style={{ fontSize: '0.75rem', color: '#9E9E9E' }}>{subtitle}</span>}
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
        {children}
      </div>
    </div>
  );
}

function ResultBar({
  label,
  sublabel,
  value,
  min,
  max,
  unit,
  multiplier = 1,
  color,
  highlight = false
}: {
  label: string;
  sublabel?: string;
  value: number;
  min: number;
  max: number;
  unit: string;
  multiplier?: number;
  color: string;
  highlight?: boolean;
}) {
  const normalizedValue = Math.min(100, Math.max(0, ((value - min) / (max - min)) * 100));
  const displayValue = (value * multiplier).toFixed(1);

  return (
    <div style={{
      padding: highlight ? '10px 12px' : '0',
      background: highlight ? 'rgba(227, 25, 55, 0.08)' : 'transparent',
      borderRadius: highlight ? '8px' : '0',
      borderLeft: highlight ? '3px solid #E31937' : 'none'
    }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
        <div>
          <span style={{ fontSize: '0.85rem', color: '#37474F', fontWeight: '500' }}>{label}</span>
          {sublabel && <span style={{ fontSize: '0.75rem', color: '#9E9E9E', marginLeft: '6px' }}>{sublabel}</span>}
        </div>
        <span style={{ fontSize: '0.9rem', fontWeight: '600', color: color }}>{displayValue}{unit}</span>
      </div>
      <div style={{
        height: '6px',
        background: '#ECEFF1',
        borderRadius: '3px',
        overflow: 'hidden'
      }}>
        <div style={{
          width: `${normalizedValue}%`,
          height: '100%',
          background: `linear-gradient(90deg, ${color}, ${color}dd)`,
          borderRadius: '3px',
          transition: 'width 0.3s ease'
        }} />
      </div>
    </div>
  );
}
