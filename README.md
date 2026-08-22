# Emotion Detection System

**AI & Robotics Field Training — Assignment 1 (Project Idea Card) → Working Prototype**
**Track:** Software AI project (Computer Vision) — no physical robot required

A real-time facial emotion detector: point a webcam at someone and the system
draws a bounding box around their face and labels it with the predicted
emotion (happy, sad, angry, neutral, surprised, etc.) and a confidence score.

---

## 1. Problem

Reading a person's emotional state usually requires a human observer. This
project automates that with a live camera feed, classifying facial
expression in real time — no hardware robot involved.

## 2. Target User

Students, instructors, or lab visitors who want a quick demo of an AI
perception system: a receptionist desk, a classroom engagement monitor, or a
simple HRI (human-robot interaction) research prototype.

## 3. AI Feature

- Face detection on each camera frame.
- Emotion classification using a pre-trained deep learning model (the
  [`fer`](https://github.com/justinshenk/fer) library, which wraps a CNN
  trained on the FER2013 dataset).
- Real-time overlay: bounding box + emotion name + confidence score.

## 4. System Architecture

```
   Input                Processing                    Output
┌───────────┐     ┌────────────────────────┐     ┌──────────────────┐
│  Webcam   │ --> │ Face detection          │ --> │ On-screen overlay │
│  feed     │     │ (OpenCV Haar / MTCNN)   │     │ - bounding box     │
│(OpenCV    │     │        │                │     │ - emotion label    │
│ VideoCap) │     │        v                │     │ - confidence score │
└───────────┘     │ Emotion classification  │     └──────────────────┘
                   │ (FER pre-trained CNN)   │              │
                   └────────────────────────┘              v
                                                    Optional CSV log
                                                    (logs/emotion_log.csv)
```

| Stage | Component | Description |
|---|---|---|
| Input | Webcam feed | Live video frames captured with OpenCV |
| Processing | Face detection + emotion model | Detect face region(s), then classify expression with the pre-trained FER model |
| Output | On-screen overlay | Bounding box, emotion label, and confidence score drawn on the video window |

## 5. Project Structure

```
Emotion_Detection_Project/
├── main.py                  # Full app: face detection + emotion classification + overlay
├── test_face_detection.py   # Lightweight smoke test — face detection only, no FER/TensorFlow needed
├── requirements.txt         # Python dependencies
├── logs/                    # Created at runtime when --log is used (emotion_log.csv)
├── docs/
│   └── testing_results.md   # Fill in with your actual test results (Assignment 3 evidence)
└── README.md                # This file
```

## 6. Setup

Requires **Python 3.9–3.11** (TensorFlow, used by `fer`, does not yet support
every newer Python version — check `python --version` first).

```bash
# 1. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

If you're on a machine without a discrete GPU, TensorFlow will run on CPU —
that's fine for this project; the default Haar-cascade face detector keeps
frame rate reasonable.

## 7. Usage

```bash
# Quick face-detection-only test first (confirms camera + OpenCV work)
python test_face_detection.py

# Full app — default camera, fast Haar-cascade face detector
python main.py

# Use a different camera index (e.g. an external USB webcam)
python main.py --camera 1

# Use MTCNN instead of Haar cascade for more accurate face detection (slower)
python main.py --mtcnn

# Log every detection (timestamp, emotion, confidence, box) to logs/emotion_log.csv
python main.py --log
```

Press **`q`** in the video window to quit.

## 8. Expected Demo (60–90 seconds)

Open the webcam. As a person's face appears, the system draws a bounding box
around it and displays the predicted emotion label live on screen. The
presenter changes expression (smile, frown, surprised) to show the label
updating in real time.

## 9. Testing Plan

| Test Case | Method | Expected Result |
|---|---|---|
| Face detected correctly | Run `test_face_detection.py` (or `main.py`) with 3 different people, varied lighting | Bounding box appears around each face |
| Emotion label accuracy | Tester deliberately smiles / frowns / stays neutral | Label changes to match the expression within 1–2 seconds |
| No face in frame | Point camera away from any face | System shows "No face detected" and does not crash |

Record your actual results in `docs/testing_results.md` as evidence for the
Assignment 3 final demo checklist.

## 10. Risks & Data

- **No personal data is stored** by default — video is processed live and
  not saved to disk. The `--log` flag only writes emotion labels and box
  coordinates (no images) to a local CSV.
- **Lighting/angle sensitivity** — accuracy depends on lighting and camera
  angle; low light may reduce detection accuracy.
- **Model bias** — the pre-trained model may be less accurate on expressions
  or demographics underrepresented in its training data (FER2013). Acknowledge
  this as a limitation in the report; do not present outputs as ground truth
  about a person's actual internal emotional state.
- **Privacy** — inform anyone being filmed during the demo and avoid
  recording bystanders who haven't consented.

## 11. Timeline Fit (per 6-week field training plan)

- **Week 1–2:** Set up Python/OpenCV, test basic face detection (`test_face_detection.py`).
- **Week 3:** Integrate emotion classification model (`main.py`), first prototype.
- **Week 4–5:** (Optional stretch) map an emotion output to a Yanshee/NAO
  response — e.g. robot says "You seem happy today!" when `happy` confidence > 0.6.
- **Week 5–6:** Polish UI overlay, run the testing table, write documentation, prepare demo.

## 12. Limitations & Future Work

- Currently single-modal (vision only); no voice or context is factored in.
- No temporal smoothing — the label can flicker frame-to-frame near a decision
  boundary. A future version could average predictions over a short rolling
  window for a more stable label.
- Stretch goal: wire a detected emotion into a Yanshee/NAO behavior (e.g.
  greeting tone, LED color) to turn this into a full "Software AI → Robot
  behavior → Integrated demo" capstone per the field training rubric.
