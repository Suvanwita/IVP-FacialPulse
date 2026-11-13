# FacialPulse

**Clinical Multimodal Affect Analysis & Stimulus-Response Depression Screening Platform**

FacialPulse is an AI-powered clinical screening system designed for psychiatric assessments. It combines facial Action Unit temporal dynamics (FACS), acoustic speech prosody, and machine learning calibrated against the DAIC-WOZ / AVEC benchmark to provide standardized DSM-5 depressive symptom indicators (PHQ-8: 0–24) and longitudinal therapeutic trajectory tracking.

---

## Key Clinical Capabilities & Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │               FACIALPULSE WEB CLIENT                   │
                    │  (Next.js 14 • Web Audio API • Canvas Stream Renderer) │
                    └──────────────────────────┬─────────────────────────────┘
                                               │
                                       REST / JSON / WAV
                                               │
                    ┌──────────────────────────▼─────────────────────────────┐
                    │              FASTAPI BACKEND RUNTIME                   │
                    │                                                        │
                    │  ┌─────────────────────────┐  ┌─────────────────────┐  │
                    │  │ Phase 1: Temporal &     │  │ Phase 2: Guided     │  │
                    │  │ Baseline Normalization  │  │ 4-Phase Protocol    │  │
                    │  └───────────┬─────────────┘  └──────────┬──────────┘  │
                    │              │                           │             │
                    │  ┌───────────▼─────────────┐  ┌──────────▼──────────┐  │
                    │  │ Phase 3: Acoustic &     │  │ Phase 4: DAIC-WOZ   │  │
                    │  │ Multimodal Fusion       │  │ ML Scoring (PHQ-8)  │  │
                    │  └───────────┬─────────────┘  └──────────┬──────────┘  │
                    │              │                           │             │
                    │  ┌───────────▼───────────────────────────▼──────────┐  │
                    │  │ Phase 5: Privacy, HIPAA/GDPR & Longitudinal Engine│  │
                    │  └──────────────────────────────────────────────────┘  │
                    └────────────────────────────────────────────────────────┘
```

---

## 5-Phase Clinical Engineering Roadmap

### Phase 1: Temporal Dynamics & Morphological Baseline Calibration
* **Individual Resting Baseline Normalization**: Quick 4-second calibration establishes individual neutral morphology, removing false-positive resting sadness or natural brow bias.
* **Moving-Window Exponential Moving Average (EMA)**: Eliminates micro-saccadic frame noise ($\alpha = 0.35$).
* **Clinical Affect Metrics**:
  * **Affect Flatness Index**: Measures lack of facial dynamism and expressivity blunting.
  * **Psychomotor Velocity**: Quantifies facial transition latency and muscle agility.
  * **Dwell Times**: Tracks sustained corrugator furrow (AU04) and lip corner depressor (AU15).

### Phase 2: Guided Interactive Elicitation Protocol
Standardized 43-second clinical stimulus-response assessment:
1. **Phase 1: Resting Baseline (8s)** — Measures resting facial state and neutral affective tone.
2. **Phase 2: Positive Affect Elicitation (12s)** — Delivers humorous prompt and measures peak smile intensity, smile onset latency, and genuine **Duchenne marker** activation (AU06 Cheek Raiser + AU12 Lip Puller).
3. **Phase 3: Standardized Cognitive Reading Task (15s)** — Displays the phonetically balanced **Rainbow Passage** while recording audio and tracking downward head tilt and persistent AU04 brow furrow under cognitive load.
4. **Phase 4: Affective Recovery & Residual Holding (8s)** — Measures emotional flexibility and residual muscle tension after stimulus offset.

### Phase 3: Multimodal Audio-Visual Fusion & Acoustic Biomarkers
* **Speech Prosody Analysis**: Fundamental frequency ($F_0$) pitch standard deviation, semitone dispersion, and monotone index.
* **Psychomotor Speech Retardation**: Voice-activity detection (VAD), speech-to-pause ratio, and prolonged pause latency detection (>750ms).
* **Vocal Perturbation & Dynamic Range**: Local jitter (%), shimmer (dB), and vocal strain index.
* **Cross-Modal Concordance & Affective Incongruence**: Detects masked affect (e.g. smiling presentation masking vocal psychomotor retardation).

### Phase 4: DAIC-WOZ / AVEC Calibrated Machine Learning Engine
* **20-Dimensional Statistical Functional Vector**: Extracts summary functionals across both visual and acoustic tracks.
* **Ensemble ML Architecture**: Gradient-Boosted Regressor with regularized Ridge regression stacking calibrated against published DAIC-WOZ / AVEC multimodal depression distributions.
* **Calibrated PHQ-8 Output (0–24)**: Predicts continuous and integer scores with a **95% Confidence Interval**.
* **DSM-5 Symptom Domain Decomposition**:
  * **Item 1: Anhedonia** (Peak smile blunting & delayed latency)
  * **Item 2: Depressed Mood** (Persistent AU15 lip depressor dwell)
  * **Item 4: Energy Loss & Fatigue** (Affect flatness index & acoustic dynamic compression)
  * **Item 7: Concentration Burden** (AU04 corrugator dwell during reading)
  * **Item 8: Psychomotor Retardation** (Speech pause ratio >45% & prolonged pause durations)
* **Explainable Feature Attribution**: Local feature contribution weights ranking the top drivers of the assessment score.

### Phase 5: Privacy, Security, Clinical Reporting & Longitudinal Tracking
* **Zero Raw Media Disk Persistence**: All video frames and microphone audio streams are analyzed in volatile RAM buffers and immediately discarded. No video or audio files are ever saved to disk.
* **Cryptographic De-Identification (HIPAA § 164.514(b))**: Salted SHA-256 pseudonymous identifiers (`SUBJ-XXXXXXXXXX`). Zero PII/PHI stored.
* **GDPR Article 17 "Right to be Forgotten"**: One-click complete deletion of longitudinal history (`DELETE /longitudinal/purge`).
* **Longitudinal Psychiatric Tracking Engine**: Session-over-session $\Delta \text{PHQ-8}$ tracking relative to previous session and baseline with DSM-5/APA clinical response criteria ($\ge \pm 5$ points).
* **Hospital-Grade EHR/EMR Print Generator**: Printable clinical summary (`@media print`) with institutional letterhead, patient pseudonym, acoustic tables, and FDA SaMD 21 CFR 882 disclaimer.

---

## Project Structure

```
FacialPulse/
├── backend/
│   ├── main.py              # FastAPI server & route handlers
│   ├── temporal.py          # Phase 1: Sliding-window EMA & baseline calibrator
│   ├── protocol.py          # Phase 2: 4-phase elicitation state machine
│   ├── audio.py             # Phase 3: Acoustic speech biomarker analyzer
│   ├── fusion.py            # Phase 3: Audio-visual multimodal fusion engine
│   ├── ml_scoring.py        # Phase 4: DAIC-WOZ calibrated PHQ-8 ensemble regressor
│   ├── longitudinal.py      # Phase 5: HIPAA/GDPR longitudinal tracking & purge engine
│   ├── test_pipeline.py     # Automated end-to-end verification test suite
│   ├── requirements.txt     # Python dependencies
│   └── data/                # De-identified session storage (gitignored)
├── frontend/
│   ├── app/
│   │   ├── layout.tsx       # Root layout & Inter font typography
│   │   └── page.tsx         # Clinical dashboard, modals, charts & EHR print layout
│   └── package.json         # Next.js 14 frontend dependencies
├── models/
│   ├── face_landmarker.task # MediaPipe BlazeFace 478-landmark model
│   └── AU_200.tflite        # Action Unit 200x200 ResNet classifier
├── start.sh                 # Unified daemon launcher with port cleanup
└── README.md
```

---

## Quickstart & Setup

### Option A: Unified Launcher (Linux / macOS)

```bash
# Make executable (first time only)
chmod +x start.sh

# Start both backend (port 8000) and frontend (port 3000)
./start.sh

# If ports 8000 or 3000 are already in use, restart cleanly with:
./start.sh --restart
```

Open **http://localhost:3000** in your browser.

---

### Option B: Manual Setup

#### 1. Backend Setup (Python 3.10+)

```bash
cd backend

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run backend API server
uvicorn main:app --host 0.0.0.0 --port 8000
```

#### 2. Frontend Setup (Node.js 18+)

```bash
cd frontend

# Install dependencies
npm install

# Run Next.js development server
npm run dev
```

---

## Running the Automated Test Suite

FacialPulse includes an automated pipeline test suite covering all 5 engines:

```bash
cd backend
source venv/bin/activate
python test_pipeline.py
```

Expected output:
```
==================================================
      FacialPulse Comprehensive Engine Test       
==================================================
Testing Phase 1: Temporal Aggregator & Baseline Calibrator...
  ✓ Temporal metrics calculated
Testing Phase 2: Guided Interactive Elicitation Protocol...
  ✓ Protocol report generated
Testing Phase 3: Acoustic Analysis & Multimodal Fusion...
  ✓ Acoustic analysis verified
  ✓ Multimodal fusion verified
Testing Phase 4: DAIC-WOZ / AVEC Calibrated ML Scoring Engine...
  ✓ Calibrated PHQ-8 Predicted
Testing Phase 5: Privacy, Security & Longitudinal Tracking Engine...
  ✓ Longitudinal tracking, HIPAA export & GDPR purge verified
==================================================
   ALL TESTS PASSED: PHASES 1 - 5 100% OPERATIONAL
==================================================
```

---

## API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Healthcheck and loaded model status |
| `POST` | `/analyze` | Analyze uploaded multipart image file |
| `POST` | `/analyze-base64` | Analyze single base64 JPEG image frame |
| `POST` | `/temporal/frame` | Ingest frame into continuous temporal sliding window |
| `GET` | `/temporal/status` | Current temporal buffer metrics and baseline state |
| `POST` | `/temporal/calibrate` | Ingest neutral sample into morphological calibrator |
| `POST` | `/temporal/reset` | Clear baseline and reset temporal buffer |
| `POST` | `/protocol/start` | Initialize 4-phase guided screening session |
| `POST` | `/protocol/frame` | Ingest frame into active protocol phase |
| `POST` | `/protocol/advance` | Manually advance to next protocol phase |
| `POST` | `/protocol/audio` | Upload 16kHz base64 WAV speech recording |
| `GET` | `/protocol/status` | Current protocol phase, time remaining, and prompts |
| `POST` | `/protocol/report` | Finalize screening, compute PHQ-8, and persist longitudinal trajectory |
| `POST` | `/protocol/cancel` | Cancel active protocol session |
| `POST` | `/audio/analyze` | Standalone acoustic feature extraction on base64 WAV |
| `POST` | `/ml/predict` | Predict calibrated PHQ-8 from active session features |
| `GET` | `/longitudinal/history` | Retrieve chronological session history for subject |
| `DELETE` | `/longitudinal/purge` | GDPR Art. 17 data destruction (permanently wipes history) |
| `GET` | `/longitudinal/export` | Export HIPAA-compliant de-identified JSON record |

---

## Clinical Biomarkers & Action Unit Mapping

| Action Unit | Facial Landmark Target | Psychiatric Significance |
|---|---|---|
| **AU04** | Corrugator Supercilii | Brow lowerer; persistent dwell indicates cognitive tension, distress, or dysphoria |
| **AU15** | Depressor Anguli Oris | Lip corner depressor; key marker of negative affect and depressive sadness |
| **AU12** | Zygomaticus Major | Lip corner puller; smile intensity; reduced amplitude reflects anhedonia |
| **AU06** | Orbicularis Oculi | Cheek raiser; genuine **Duchenne marker** when co-activated with AU12 |
| **AU01** | Frontalis Pars Medialis | Inner brow raiser; elevated in distress and psychomotor holding |

---

## Investigational Regulatory Disclaimer

> **INVESTIGATIONAL DEVICE NOTICE (FDA 21 CFR 882):**  
> FacialPulse is an investigational software-as-a-medical-device (SaMD) tool intended for adjunctive psychiatric screening and research. This assessment is not a standalone diagnostic evaluation. Clinical correlation by a licensed healthcare professional is required. All patient data is de-identified pursuant to HIPAA § 164.514(b) and GDPR Art. 17.
