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

  // Phase 1: Baseline Calibration & Temporal Screening States
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [calibrationStep, setCalibrationStep] = useState(0);
  const [isBaselineCalibrated, setIsBaselineCalibrated] = useState(false);
  const [baselineData, setBaselineData] = useState<Record<string, number> | null>(null);

  const [assessmentActive, setAssessmentActive] = useState(false);
  const [assessmentTimeLeft, setAssessmentTimeLeft] = useState(30);

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
    purple: '#7B1FA2'
  };

  // Update time every second
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  // Check API and temporal status on mount
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
          setBaselineData(data.baseline);
        }
      })
      .catch(() => {});
  }, []);

  // Auto capture interval
  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isAutoCapture && isStreaming && !isCalibrating) {
      interval = setInterval(() => {
        captureAndAnalyze();
      }, 1200);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isAutoCapture, isStreaming, isCalibrating]);

  // Timed 30-second assessment countdown
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (assessmentActive && assessmentTimeLeft > 0) {
      timer = setInterval(() => {
        setAssessmentTimeLeft(prev => {
          if (prev <= 1) {
            setAssessmentActive(false);
            setIsAutoCapture(false);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [assessmentActive, assessmentTimeLeft]);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setIsStreaming(true);
        setError(null);
        setScanStatus('ready');
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
      setIsStreaming(false);
      setIsAutoCapture(false);
      setAssessmentActive(false);
      setIsCalibrating(false);
      setScanStatus('ready');
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
            setBaselineData(data.baseline);
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
      setBaselineData(null);
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

  const startTimedAssessment = async () => {
    if (!isStreaming) {
      await startCamera();
    }
    await resetTemporal();
    setAssessmentActive(true);
    setAssessmentTimeLeft(30);
    setIsAutoCapture(true);
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
              AI Mental Wellness &amp; Facial Expression Dynamics
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

      {/* Phase 1: Baseline Calibration & Assessment Control Banner */}
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
        {/* Left: Calibration Status */}
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
               isBaselineCalibrated ? 'All expressions normalized against your natural resting face' :
               'Recommended: Calibrate for 4s with a neutral face to remove natural resting-face bias'}
            </p>
          </div>
        </div>

        {/* Right: Actions */}
        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          {isCalibrating ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
              <div style={{ width: '120px', height: '8px', background: '#ECEFF1', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{ width: `${calibrationProgress * 100}%`, height: '100%', background: colors.warning, transition: 'width 0.3s' }} />
              </div>
              <span style={{ fontSize: '0.8rem', color: colors.warning, fontWeight: '600' }}>
                {Math.round(calibrationProgress * 100)}%
              </span>
            </div>
          ) : !isBaselineCalibrated ? (
            <button
              onClick={startCalibration}
              style={{
                padding: '8px 16px',
                background: colors.lightBg,
                color: colors.primary,
                border: `1.5px solid ${colors.primary}`,
                borderRadius: '8px',
                fontSize: '0.85rem',
                fontWeight: '600',
                cursor: 'pointer'
              }}
            >
              🎯 Calibrate Neutral Face (4s)
            </button>
          ) : (
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

          {/* Timed Assessment Trigger */}
          <button
            onClick={startTimedAssessment}
            disabled={assessmentActive || isCalibrating}
            style={{
              padding: '8px 16px',
              background: assessmentActive ? colors.warning : `linear-gradient(135deg, ${colors.primary} 0%, ${colors.primaryDark} 100%)`,
              color: 'white',
              border: 'none',
              borderRadius: '8px',
              fontSize: '0.85rem',
              fontWeight: '600',
              cursor: assessmentActive ? 'default' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            {assessmentActive ? `⏱️ Screening: ${assessmentTimeLeft}s` : '⏱️ 30s Timed Screening'}
          </button>
        </div>
      </div>

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
          {/* Instructions Card */}
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
              INSTRUCTIONS
            </h3>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
              <InstructionItem
                icon={<svg width="18" height="18" viewBox="0 0 24 24" fill={colors.primary}><path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/></svg>}
                title="Remove Accessories"
                subtitle="Please remove glasses, face mask, or headwear"
                color={colors.primary}
              />
              <InstructionItem
                icon={<svg width="18" height="18" viewBox="0 0 24 24" fill={colors.primary}><path d="M9 11.75c-.69 0-1.25.56-1.25 1.25s.56 1.25 1.25 1.25 1.25-.56 1.25-1.25-.56-1.25-1.25-1.25zm6 0c-.69 0-1.25.56-1.25 1.25s.56 1.25 1.25 1.25 1.25-.56 1.25-1.25-.56-1.25-1.25-1.25zM12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2z"/></svg>}
                title="Position Face"
                subtitle="Center your face within the guide frame"
                color={colors.primary}
              />
              <InstructionItem
                icon={<svg width="18" height="18" viewBox="0 0 24 24" fill={colors.primary}><path d="M11.99 2C6.47 2 2 6.48 2 12s4.47 10 9.99 10C17.52 22 22 17.52 22 12S17.52 2 11.99 2zM12 20c-4.42 0-8-3.58-8-8s3.58-8 8-8 8 3.58 8 8-3.58 8-8 8zm3.5-9c.83 0 1.5-.67 1.5-1.5S16.33 8 15.5 8 14 8.67 14 9.5s.67 1.5 1.5 1.5zm-7 0c.83 0 1.5-.67 1.5-1.5S9.33 8 8.5 8 7 8.67 7 9.5 7.67 11 8.5 11zm3.5 6.5c2.33 0 4.31-1.46 5.11-3.5H6.89c.8 2.04 2.78 3.5 5.11 3.5z"/></svg>}
                title="Baseline Normalization"
                subtitle="Calibrate neutral face before running temporal checks"
                color={colors.primary}
              />
            </div>
          </div>

          {/* Temporal Indicators Card */}
          <div style={{
            background: colors.accentLight,
            borderRadius: '16px',
            padding: '20px',
            borderLeft: `4px solid ${colors.accent}`
          }}>
            <h4 style={{ margin: '0 0 12px', fontSize: '0.85rem', color: colors.accent, fontWeight: '700' }}>
              Temporal Affective Biomarkers
            </h4>
            <ul style={{ margin: 0, padding: '0 0 0 18px', fontSize: '0.8rem', color: '#5D4037', lineHeight: '1.8' }}>
              <li><strong>AU04 Persistence</strong> - Chronic brow furrowing (worry/tension)</li>
              <li><strong>AU15 Persistence</strong> - Lip corner depressor holding (sadness)</li>
              <li><strong>Flat Affect Index</strong> - Blunted expressive variance over time</li>
              <li><strong>Head-Down Dwell</strong> - Prolonged downward pitch posture</li>
              <li><strong>Smile Deficit</strong> - Low peak &amp; low dynamic smile range</li>
            </ul>
          </div>
        </div>

        {/* Center - Camera */}
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
                background: isCalibrating ? '#FFF8E1' :
                           scanStatus === 'success' ? '#E8F5E9' :
                           scanStatus === 'scanning' ? '#FFF8E1' :
                           scanStatus === 'no-face' ? colors.accentLight : colors.lightBg,
                borderRadius: '25px'
              }}>
                <div style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  background: isCalibrating ? colors.warning :
                             scanStatus === 'success' ? colors.success :
                             scanStatus === 'scanning' ? colors.warning :
                             scanStatus === 'no-face' ? colors.accent : colors.primary,
                  animation: (scanStatus === 'scanning' || isCalibrating) ? 'pulse 1s infinite' : 'none'
                }} />
                <span style={{
                  fontSize: '0.85rem',
                  fontWeight: '500',
                  color: isCalibrating ? colors.warning :
                         scanStatus === 'success' ? colors.success :
                         scanStatus === 'scanning' ? colors.warning :
                         scanStatus === 'no-face' ? colors.accent : colors.primary
                }}>
                  {isCalibrating ? `Calibrating Neutral Resting Face (${calibrationStep}/8)...` :
                   assessmentActive ? `Assessment Active (${assessmentTimeLeft}s remaining)` :
                   scanStatus === 'success' ? 'Face Detected & Tracking' :
                   scanStatus === 'scanning' ? 'Scanning Expression Dynamics...' :
                   scanStatus === 'no-face' ? 'No Face Detected' :
                   'Ready to Scan'}
                </span>
              </div>
            </div>

            {/* Camera Container */}
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

              {/* Face Guide Overlay */}
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
                    stroke={isCalibrating ? colors.warning :
                           scanStatus === 'success' ? colors.success :
                           scanStatus === 'scanning' ? colors.warning : colors.primary}
                    strokeWidth="3"
                    strokeDasharray={(scanStatus === 'scanning' || isCalibrating) ? '10,5' : 'none'}
                  />
                  <path d="M 30 70 L 30 30 L 70 30" fill="none" stroke={colors.accent} strokeWidth="4" strokeLinecap="round"/>
                  <path d="M 230 70 L 230 30 L 190 30" fill="none" stroke={colors.accent} strokeWidth="4" strokeLinecap="round"/>
                  <path d="M 30 250 L 30 290 L 70 290" fill="none" stroke={colors.accent} strokeWidth="4" strokeLinecap="round"/>
                  <path d="M 230 250 L 230 290 L 190 290" fill="none" stroke={colors.accent} strokeWidth="4" strokeLinecap="round"/>
                </svg>
              </div>

              {/* Calibration Overlay Banner */}
              {isCalibrating && (
                <div style={{
                  position: 'absolute',
                  top: '16px',
                  left: '16px',
                  right: '16px',
                  background: 'rgba(0, 0, 0, 0.75)',
                  backdropFilter: 'blur(4px)',
                  borderRadius: '10px',
                  padding: '12px 16px',
                  color: 'white',
                  textAlign: 'center',
                  fontSize: '0.85rem'
                }}>
                  Please relax facial muscles and look straight into camera
                </div>
              )}

              {/* Camera Off State */}
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

            {/* Buttons */}
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
                    disabled={isAnalyzing || isCalibrating}
                    style={{
                      flex: 1,
                      padding: '14px 20px',
                      background: isAnalyzing ? '#B0BEC5' : `linear-gradient(135deg, ${colors.primary} 0%, ${colors.primaryDark} 100%)`,
                      color: 'white',
                      border: 'none',
                      borderRadius: '10px',
                      fontSize: '1rem',
                      fontWeight: '600',
                      cursor: (isAnalyzing || isCalibrating) ? 'not-allowed' : 'pointer'
                    }}
                  >
                    {isAnalyzing ? 'Analyzing...' : 'Scan Now'}
                  </button>
                  <button
                    onClick={() => {
                      if (assessmentActive) setAssessmentActive(false);
                      setIsAutoCapture(!isAutoCapture);
                    }}
                    disabled={isCalibrating}
                    style={{
                      padding: '14px 18px',
                      background: isAutoCapture ? colors.warning : '#ECEFF1',
                      color: isAutoCapture ? 'white' : colors.textLight,
                      border: 'none',
                      borderRadius: '10px',
                      fontSize: '0.95rem',
                      fontWeight: '600',
                      cursor: isCalibrating ? 'not-allowed' : 'pointer'
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
              <p style={{ margin: 0, fontSize: '0.95rem' }}>Scan face or start auto monitor to see results</p>
              <p style={{ margin: '5px 0 0', fontSize: '0.85rem' }}>Temporal aggregation activates over continuous observation</p>
            </div>
          ) : (
            <>
              {/* Phase 1: Temporal Aggregation Depression Risk Card */}
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
                        Temporal Depression Biomarker Risk
                      </span>
                      <div style={{ fontSize: '0.75rem', color: colors.textLight, marginTop: '2px' }}>
                        Window: {temporal.frames_count} frames {temporal.baseline_active ? '• Baseline Normalized' : '• Raw Morphology'}
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

                  {/* Affect Flatness / Immobility Index */}
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
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', color: colors.textLight, marginTop: '3px' }}>
                      <span>Dynamic Affect</span>
                      <span>Blunted / Rigid Affect</span>
                    </div>
                  </div>

                  {/* Key Temporal Dwell Breakdowns */}
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

                  {/* Temporal Contributing Factors */}
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

              {/* Phase 1: Real-Time Expression Timeline Sparkline Chart */}
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

                  {/* Legend */}
                  <div style={{ display: 'flex', justifyContent: 'center', gap: '12px', marginTop: '8px', flexWrap: 'wrap' }}>
                    <LegendItem color="#FB8C00" label="AU04 Brow Furrow" />
                    <LegendItem color="#E53935" label="AU15 Lip Depressor" />
                    <LegendItem color="#43A047" label="Smile" />
                    <LegendItem color="#1E88E5" label="Pitch (Down)" />
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

              {/* Metadata */}
              <div style={{
                marginTop: '16px',
                padding: '12px 14px',
                background: colors.lightBg,
                borderRadius: '10px',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}>
                <span style={{ fontSize: '0.85rem', color: colors.textLight }}>Landmarks Detected</span>
                <span style={{ fontSize: '1.1rem', fontWeight: '700', color: colors.primary }}>{result.data.landmarks_count}</span>
              </div>
            </>
          )}
        </div>
      </div>

      {/* API Status Floating Pill */}
      <div style={{
        position: 'fixed',
        bottom: '20px',
        right: '20px',
        display: 'flex',
        alignItems: 'center',
        gap: '8px',
        padding: '10px 16px',
        background: 'white',
        borderRadius: '25px',
        boxShadow: '0 2px 15px rgba(0,0,0,0.1)'
      }}>
        <div style={{
          width: '8px',
          height: '8px',
          borderRadius: '50%',
          background: apiStatus === 'connected' ? colors.success : apiStatus === 'checking' ? colors.warning : colors.accent
        }} />
        <span style={{ fontSize: '0.8rem', color: colors.textLight }}>
          {apiStatus === 'connected' ? 'API Connected' : apiStatus === 'checking' ? 'Connecting...' : 'API Offline'}
        </span>
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

  // Function to map index to X
  const getX = (idx: number) => padding + (idx / (pointsCount - 1)) * (width - 2 * padding);

  // Function to map normalized 0..1 to Y (inverted)
  const getY = (val: number) => height - padding - Math.min(1, Math.max(0, val)) * (height - 2 * padding);

  // Normalize pitch (-25° to +15° -> 1 to 0 where -25 is 1 high down)
  const normalizePitch = (pitch: number) => {
    return Math.min(1, Math.max(0, (-pitch - 5) / 20));
  };

  const createPath = (getter: (p: typeof timeline[0]) => number) => {
    return timeline
      .map((p, i) => `${i === 0 ? 'M' : 'L'} ${getX(i).toFixed(1)} ${getY(getter(p)).toFixed(1)}`)
      .join(' ');
  };

  const pathAu04 = createPath(p => p.au04);
  const pathAu15 = createPath(p => p.au15);
  const pathSmile = createPath(p => p.smile);
  const pathPitch = createPath(p => normalizePitch(p.pitch));

  return (
    <div style={{ width: '100%', overflow: 'hidden' }}>
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: `${height}px`, background: '#F8F9FA', borderRadius: '8px' }}>
        {/* Horizontal gridlines */}
        <line x1={padding} y1={height * 0.25} x2={width - padding} y2={height * 0.25} stroke="#E0E0E0" strokeDasharray="3 3" />
        <line x1={padding} y1={height * 0.50} x2={width - padding} y2={height * 0.50} stroke="#E0E0E0" strokeDasharray="3 3" />
        <line x1={padding} y1={height * 0.75} x2={width - padding} y2={height * 0.75} stroke="#E0E0E0" strokeDasharray="3 3" />

        {/* Traces */}
        <path d={pathPitch} fill="none" stroke="#1E88E5" strokeWidth="1.8" opacity="0.6" strokeDasharray="4 2" />
        <path d={pathSmile} fill="none" stroke="#43A047" strokeWidth="2.2" />
        <path d={pathAu04} fill="none" stroke="#FB8C00" strokeWidth="2.2" />
        <path d={pathAu15} fill="none" stroke="#E53935" strokeWidth="2.2" />
      </svg>
    </div>
  );
}

function LegendItem({ color, label }: { color: string; label: string }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '5px' }}>
      <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: color }} />
      <span style={{ fontSize: '0.7rem', color: '#546E7A', fontWeight: '500' }}>{label}</span>
    </div>
  );
}

// Instruction Item Component
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

// Component for result sections
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

// Component for result bars
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
