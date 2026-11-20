# EE 471 - Week 8: MediaPipe Vision, Text & Audio Tasks

This repository contains implementations for **Task 1** (22 MediaPipe Solution Demos), **Task 2** (Pose Estimation & Arm Raise Classification), and **Task 3** (Face Detection & Looking Direction Classification) using Google's **MediaPipe** library containerized with **Cog** and **Docker**.

---

## Repository Structure

```text
ee471-week8/
├── task-1/                           # 22 MediaPipe demonstration scripts
│   ├── download_models.sh            # One-click downloader for model assets
│   ├── models/                       # Downloaded .task / .tflite / .litertlm models
│   ├── sample_data/                  # Minimal test data (sample_face.png, sample_pose.jpg, sample_audio.wav)
│   ├── audio_classifier.py           # -> audio_classifier.avi
│   ├── audio_transcriber.py          # -> audio_transcriber.avi
│   ├── decision_maker.py             # -> decision_maker.avi
│   ├── face_detector.py              # -> face_detector.avi
│   ├── face_landmarker.py            # -> face_landmarker.avi
│   ├── gesture_recognizer.py         # -> gesture_recognizer.avi
│   ├── hand_landmarker.py            # -> hand_landmarker.avi
│   ├── holistic_landmarker.py        # -> holistic_landmarker.avi
│   ├── image_classifier.py           # -> image_classifier.avi
│   ├── image_embedder.py             # -> image_embedder.avi
│   ├── image_segmenter.py            # -> image_segmenter.avi
│   ├── interactive_segmenter.py      # -> interactive_segmenter.avi
│   ├── language_detector.py          # -> language_detector.avi
│   ├── object_detector.py            # -> object_detector.avi
│   ├── pose_landmarker.py            # -> pose_landmarker.avi
│   ├── semantic_retriever.py         # -> semantic_retriever.avi
│   ├── text_classifier.py            # -> text_classifier.avi
│   ├── text_embedder.py              # -> text_embedder.avi
│   ├── text_proofreader.py           # -> text_proofreader.avi
│   ├── text_summarizer.py            # -> text_summarizer.avi
│   ├── text_to_speech.py             # -> text_to_speech.avi
│   └── universal_embedder.py         # -> universal_embedder.avi
├── task-2/
│   ├── classify_arm.py               # Arm raise classifier (CLI + Cog Predictor)
│   ├── cog.yaml                      # Cog container specification
│   ├── Dockerfile                    # Standalone Dockerfile alternative
│   ├── pose_landmarker.task          # MediaPipe Pose Landmarker model asset
│   ├── pose-1.jpg, pose-2.jpg, pose-3.jpg
├── task-3/
│   ├── classify_face.py              # Looking direction classifier (CLI + Cog Predictor)
│   ├── cog.yaml                      # Cog container specification
│   ├── Dockerfile                    # Standalone Dockerfile alternative
│   ├── blaze_face_short_range.tflite # MediaPipe Face Detector model asset
│   ├── face-1.png, face-2.png, face-3.png
├── Week-8-InClass-Exercise.pdf
└── README.md
```

---

## Task 1: 22 MediaPipe Demonstrations

All 22 scripts follow Google's official MediaPipe Tasks 4-step pattern:
1. Initialize `BaseOptions` with the model asset path.
2. Create the task object (`create_from_options`).
3. Load sample input from `sample_data/` (or default string/file).
4. Run inference and print human-readable output to stdout.

### Setup & Running
```bash
cd task-1

# When on a fast internet connection, download all models in one click:
./download_models.sh

# Run any task script directly in Python:
python3 face_detector.py
python3 pose_landmarker.py
python3 language_detector.py
python3 image_classifier.py
```

---

## Task 2: Arm Raise Classification

Classifies which arm is raised in an image using MediaPipe's **Pose Landmarker**.

### Classification Algorithm
- **Landmarks extracted**: Left shoulder (11), Left wrist (15), Right shoulder (12), Right wrist (16).
- In image coordinates, origin $(0, 0)$ is top-left ($y$ increases downwards).
- **Condition**:
  - Left arm up: `left_wrist.y < left_shoulder.y`
  - Right arm up: `right_wrist.y < right_shoulder.y`
- **Output**: `"left"`, `"right"`, `"both"`, or `"None"`

### Running Task 2
```bash
cd task-2

# Direct Python CLI
python3 classify_arm.py pose-1.jpg  # Output: left
python3 classify_arm.py pose-2.jpg  # Output: right
python3 classify_arm.py pose-3.jpg  # Output: both

# Using Cog
cog run -i image=@pose-1.jpg
```

---

## Task 3: Face Looking Direction Classification

Classifies whether a person is looking **left**, **right**, or **straight** using MediaPipe's **Face Detector** (`blaze_face_short_range.tflite`).

### Classification Algorithm
The face detector returns 6 key facial landmarks (eyes, nose, mouth, ears).
The algorithm calculates the horizontal displacement of the nose relative to the midpoint between both eyes, normalized by inter-ocular distance:

$$\Delta = \frac{\text{nose.x} - \frac{\text{right\_eye.x} + \text{left\_eye.x}}{2}}{|\text{left\_eye.x} - \text{right\_eye.x}|}$$

- **$\Delta < -0.15$**: Nose is shifted to the left $\rightarrow$ `"left"`
- **$\Delta > +0.15$**: Nose is shifted to the right $\rightarrow$ `"right"`
- **$-0.15 \le \Delta \le +0.15$**: Nose is centered $\rightarrow$ `"straight"`

### Running Task 3
```bash
cd task-3

# Direct Python CLI
python3 classify_face.py face-1.png  # Output: left
python3 classify_face.py face-2.png  # Output: right
python3 classify_face.py face-3.png  # Output: straight

# Using Cog
cog run -i image=@face-1.png
```

---

## Deliverables Checklist
- [x] Public GitHub repository link
- [ ] Task 1 video demonstrations: 22 `.avi` files (e.g. `audio_classifier.avi`, `face_detector.avi`, etc.)
- [ ] Task 2 video demonstration: `task-2-demo.avi`
- [ ] Task 2 code explanation: `task-2-explanation.avi`
- [ ] Task 3 video demonstration: `task-3-demo.avi`
- [ ] Task 3 code explanation: `task-3-explanation.avi`
