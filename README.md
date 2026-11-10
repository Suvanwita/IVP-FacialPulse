# FacialPulse

AI-powered web platform for facial behavior and mental wellness screening on desktop.

## Features

- Real-time webcam capture
- Face detection using MediaPipe
- Action Unit detection using TensorFlow Lite model
- Head pose estimation
- Eye openness detection
- Smile probability

## Project Structure

```
FacialPulse/
├── backend/          # Python FastAPI backend
│   ├── main.py       # API server
│   └── requirements.txt
├── frontend/         # Next.js frontend
│   ├── app/
│   │   ├── layout.tsx
│   │   └── page.tsx
│   └── package.json
├── models/           # ML models
│   └── AU_200.tflite # Action Unit detection model
└── README.md
```

## Setup Instructions

### 1. Backend Setup (Python)

```bash
cd backend

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt

# Run backend
python main.py
# or: uvicorn main:app --reload --port 8000
```

Backend will run at: http://localhost:8000

### 2. Frontend Setup (Next.js)

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will run at: http://localhost:3000

## Usage

1. Start the backend first (port 8000)
2. Start the frontend (port 3000)
3. Open http://localhost:3000 in your browser
4. Click "Start Camera" to enable webcam
5. Click "Capture & Analyze" or "Auto Capture" to analyze faces

## API Endpoints

- `GET /` - Health check
- `POST /analyze` - Analyze uploaded image file
- `POST /analyze-base64` - Analyze base64 encoded image

## Depression Indicators (Action Units)

| Action Unit | Description | Depression Correlation |
|-------------|-------------|------------------------|
| AU04 | Brow Lowerer | Often elevated |
| AU15 | Lip Corner Depressor | Associated with sadness |
| AU12 | Lip Corner Puller | Reduced (less smiling) |
| AU01 | Inner Brow Raiser | Can indicate distress |

## Requirements

- Python 3.8+
- Node.js 18+
- Webcam
