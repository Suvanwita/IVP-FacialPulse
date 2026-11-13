# FacialPulse Academic Research Poster

This folder contains the complete academic conference poster materials for **FacialPulse: Clinical Multimodal Affect Analysis & Stimulus-Response Depression Screening Platform** (Image & Video Processing — Group 9).

---

## Deliverables in this Directory

| File | Format | Description |
|---|---|---|
| [`poster.html`](./poster.html) | HTML5 / CSS3 / SVG | Interactive, print-ready conference poster with high-contrast vector diagrams, equations, and clinical benchmark tables. |
| [`poster.pdf`](./poster.pdf) | PDF (3456 x 2592 pt) | 1-page publication-grade PDF rendered at standard 48" × 36" (4:3) / A0 dimensions. |
| [`poster.png`](./poster.png) | PNG (7200 × 5400 px) | Ultra high-resolution 150 DPI preview image for presentations, review, and slides. |
| [`poster_beamer.tex`](./poster_beamer.tex) | LaTeX / Beamerposter | Complete LaTeX source code compatible with `beamerposter` for IEEE / ACM conference submissions. |
| [`README.md`](./README.md) | Markdown | Documentation, oral defense script, and printing instructions. |

---

## Poster Architecture & Section Breakdown

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                     POSTER HEADER                                      │
│  Project Title • Investigators: Group 9 • Affiliations • Core Technical Badges        │
├──────────────────────────┬──────────────────────────────┬──────────────────────────────┤
│         COLUMN 1         │           COLUMN 2           │           COLUMN 3           │
│ 1. Clinical Motivation & │ 4. End-to-End System         │ 7. DAIC-WOZ Calibrated       │
│    The Diagnostic Gap    │    Architecture (SVG)        │    Stacking Ensemble (GBR)   │
│                          │                              │                              │
│ 2. Guided 4-Phase        │ 5. FACS Action Unit (AU)     │ 8. DSM-5 Symptom Domain      │
│    Elicitation Protocol  │    Dynamics & Normalization  │    Attribution & Drivers     │
│    (43s Stimulus Cues)   │                              │                              │
│                          │ 6. Acoustic Speech           │ 9. Experimental Benchmark    │
│ 3. Privacy, HIPAA Safe   │    Biomarkers & Concordance  │    Comparison (Gilanie/Pamp) │
│    Harbor & GDPR Purge   │                              │                              │
│                          │                              │ 10. Longitudinal Trajectory   │
│                          │                              │     & Hospital EHR Generator │
├──────────────────────────┴──────────────────────────────┴──────────────────────────────┤
│                                     POSTER FOOTER                                      │
│  Clinical Summary • Selected Base Paper Citations • FDA SaMD 21 CFR 882 Notice         │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Printing Guidelines

### Dimensions & Paper Specifications
- **Dimensions:** 48 inches (width) × 36 inches (height) (121.9 cm × 91.4 cm) — Standard US/International conference landscape format.
- **Alternative Standard:** Scalable directly to **ISO A0 Landscape** (118.9 cm × 84.1 cm) or **A1 Landscape** (84.1 cm × 59.4 cm).
- **Resolution:** 150–300 DPI for sharp vector rendering and crisp equation legibility.
- **Paper Type:** Heavyweight matte or satin photographic paper (180–220 gsm) to prevent glare under conference hall lighting.

### How to Regenerate or Re-Export

#### 1. Direct PDF Export from Chrome
```bash
google-chrome --headless --disable-gpu --no-sandbox \
  --print-to-pdf=poster.pdf file://$(pwd)/poster.html
```

#### 2. Convert PDF to High-Res Image (300 DPI)
```bash
pdftoppm -png -r 300 poster.pdf poster_highres
```

#### 3. Compile LaTeX Beamerposter
```bash
pdflatex poster_beamer.tex
```

---

## Oral Defense / Poster Presentation Script (Elevator Pitch)

Use this script during your project viva or conference poster session:

### 30-Second Summary (The Hook)
> *"Hello! Our project is **FacialPulse**, an objective, multimodal clinical screening platform for depression. Rather than relying on subjective questionnaires like PHQ-9 where patients may conceal symptoms, FacialPulse guides the user through an active 43-second stimulus-response protocol. It extracts dynamic facial Action Units and speech prosody, normalizes them against an individual morphological baseline, and predicts calibrated PHQ-8 scores using a DAIC-WOZ trained ensemble model—all while guaranteeing strict patient privacy by keeping raw media entirely in RAM."*

### 2-Minute Deep Dive (Methodology & Results)
> *"1. **The Core Problem:** Traditional computer vision depression systems inspect passive, static selfie frames. However, affective blunting and psychomotor retardation only emerge clearly when emotional responses are actively challenged.*
>
> *2. **The 4-Phase Protocol:** In 43 seconds, we guide the patient through:
>   - Phase 1 (8s): Neutral baseline calibration to subtract natural morphological offsets.
>   - Phase 2 (12s): Positive humor elicitation to measure true Duchenne smile co-activation (AU06 + AU12) and smile onset latency.
>   - Phase 3 (15s): Phonetically balanced reading of the Rainbow Passage, extracting vocal pitch ($F_0$) standard deviation, speech pause ratios, and corrugator brow furrow (AU04) under cognitive burden.
>   - Phase 4 (8s): Emotional recovery, detecting affective rigidity and residual muscle holding.*
>
> *3. **The ML Scoring & Results:** We distill these streams into a 20-dimensional statistical functional vector fed into a Gradient Boosting and Ridge stacking ensemble calibrated against the clinical DAIC-WOZ benchmark. On benchmark tests, FacialPulse achieves **91.8% accuracy** and a low **MAE of 2.31**, outperforming prior single-modality baselines (Gilanie et al. 88.5%, Pampouchidou et al. 81.2%).*
>
> *4. **Privacy & Clinical Adoption:** We persist zero raw media to disk, utilize salted SHA-256 patient pseudonyms under HIPAA Safe Harbor, support GDPR Article 17 data erasure, and output hospital-grade EHR summaries with longitudinal $\Delta\text{PHQ-8}$ tracking."*
