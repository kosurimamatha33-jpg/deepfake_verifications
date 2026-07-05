# 🔍 AI-Powered Content Authenticity System

> **Detect deepfakes and AI-generated faces using Eyebrow Pattern Analysis as the core detection engine — no deep learning model required.**

---

## 📌 Overview

This is a real-time content authenticity verification system built with Python and Streamlit. It analyzes images, videos, and live camera feeds to determine whether the content is **human-created** or **AI-generated / deepfake**.

The system's main detection feature is a **5-Factor Eyebrow Pattern Analysis** engine, which examines lighting conditions, image quality, face angle, facial visibility, and multi-frame consistency — alongside eyebrow shape, hair density, continuity, texture, and symmetry — to make a final authenticity decision.

---

## ✨ Features

- 👁️ **Eyebrow Pattern Analysis** — Main detection engine with 10 sub-metrics across 5 quality factors
- 🎥 **Live Camera Verification** — Real-time analysis with live status feedback and step-by-step instructions
- 📸 **Image Upload Analysis** — Auto-analyzes on upload, no button click needed
- 🎬 **Video Upload Analysis** — Frame-by-frame sampling with consistency graph and average scores
- 🤖 **Deepfake Detection** — Blur heuristic fallback; upgradeable with a real TFLite model
- ✨ **Liveness Detection** — Blink detection and head movement tracking
- 📊 **Detailed Reports** — Color-coded 10-tile breakdown with percentage scores for every factor
- 🚫 **No MediaPipe, No TensorFlow** — Runs on OpenCV only, lightweight and easy to install

---

## 🧠 How It Works

### Detection Weight

```
Final Score = Eyebrow Analysis (45%) + Deepfake Score (30%) + Liveness Check (25%)
```

### 5-Factor Eyebrow Analysis

The eyebrow module runs every frame through 5 quality factors before extracting eyebrow features:

| Factor | What It Checks |
|---|---|
| 💡 Lighting | Detects dark or overexposed frames; applies gamma correction + CLAHE |
| 📷 Image Quality | Measures sharpness via Laplacian variance |
| 📐 Face Angle | Estimates yaw from eye gap; multi-scale detection for tilted faces |
| 👤 Full Face Visible | Checks face-to-frame ratio, eye count, eyebrow zone cropping |
| 🎞️ Multi-Frame Consistency | 30-frame rolling variance — stable patterns = authentic |

Then 5 eyebrow features are extracted:

| Feature | What It Checks |
|---|---|
| 🌀 Shape | Aspect ratio of eyebrow arch — natural eyebrows have a 3–7 ratio |
| 🔬 Hair Density | Dark pixel density — too sparse or too dense = AI artifact |
| 〰️ Continuity | Row-by-row transition analysis — natural hair breaks naturally |
| 🪡 Texture | Laplacian edge variance — AI brows are often too smooth |
| ⚖️ Symmetry | Left vs right comparison — AI faces are often unnaturally perfect |

**Final Eyebrow Score = Features (60%) × Quality Factors (40%)**

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip
- A webcam (for Live Camera mode)

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/your-username/deepfake-verification-system.git
cd deepfake-verification-system
```

**2. Create and activate a virtual environment (recommended)**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Run the app**

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

---

## 📁 Project Structure

```
deepfake-verification-system/
│
├── app.py               # Main Streamlit application — all 3 modes + UI
├── eyebrow.py           # 5-factor eyebrow analysis engine (main feature)
├── liveness.py          # Blink detection + head movement tracking
├── deepfake.py          # Blur heuristic / TFLite deepfake detector
├── utils.py             # Image and video loaders
├── requirements.txt     # Python dependencies
├── README.md            # This file
│
└── models/              # (Optional) Place TFLite model here
    └── mesonet.tflite   # Drop in for real deepfake detection
```

---

## 🖥️ Usage Guide

### 🎥 Live Camera Verification

1. Select **Live Camera Verification** from the sidebar
2. Follow the on-screen instructions:
   - 💡 Face a window or lamp for good lighting
   - 👤 Keep both eyebrows fully in frame
   - 📐 Look straight at the camera
   - 👁️ Blink naturally 2–3 times
   - ↔️ Do a small left/right head turn
3. Check the real-time status panel for live feedback
4. View the full 10-tile eyebrow report after verification completes

### 📸 Upload Image

1. Select **Upload Image** from the sidebar
2. Upload any `.jpg`, `.jpeg`, or `.png` file
3. Analysis runs **automatically** — no button click needed
4. Review the overall verdict and the full eyebrow analysis panel

### 🎬 Upload Video

1. Select **Upload Video** from the sidebar
2. Upload any `.mp4`, `.avi`, or `.mov` file
3. The system samples up to 30 frames and analyzes each one
4. View average scores per factor, a frame-by-frame line chart, and the final verdict

---

## 📊 Understanding the Results

### Verdict Labels

| Label | Meaning |
|---|---|
| ✅ HUMAN-CREATED | All checks passed, content appears authentic |
| ⚠️ POTENTIALLY AI-GENERATED | Mixed signals, manual review recommended |
| ❌ AI-GENERATED | Significant anomalies detected, likely deepfake or AI |

### Score Colors

| Color | Score Range | Meaning |
|---|---|---|
| 🟢 Green | 75% – 100% | Authentic / Normal |
| 🟡 Yellow | 55% – 74% | Uncertain / Mixed |
| 🔴 Red | 0% – 54% | Anomaly / Fake |

---

## ⚙️ Optional: Real Deepfake Model

By default the system uses a sharpness-based blur heuristic for the deepfake check. To upgrade to a real model:

1. Download a MesoNet `.tflite` model
2. Create a `models/` folder in the project root
3. Place the file at `models/mesonet.tflite`
4. Install TensorFlow Lite: `pip install tensorflow`

The system will auto-detect and use the model on next run.

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Streamlit | ≥ 1.28.0 | Web UI framework |
| OpenCV | ≥ 4.8.0 | Face/eye detection, image processing |
| NumPy | ≥ 1.24.0 | Numerical computations |
| Pillow | ≥ 10.0.0 | Image handling |

---

## 🔬 Why Eyebrows?

Eyebrows are one of the hardest facial features for AI to replicate convincingly:

- **Complex structure** — individual hairs with natural variation
- **Subtle asymmetry** — human faces are naturally imperfect; AI tends to over-symmetrize
- **Texture depth** — natural hair texture is difficult to fake at the pixel level
- **Alignment precision** — deepfake generators frequently misplace or distort brow position
- **Consistency over time** — real eyebrows look stable across frames; AI artifacts flicker

This makes eyebrow analysis a highly reliable signal for detecting AI-generated or manipulated faces.

---

## 📋 Tips for Best Results

| Tip | Why It Matters |
|---|---|
| 💡 Use good lighting | Poor lighting reduces eyebrow texture visibility |
| 👤 Keep full face visible | Cropped eyebrows cannot be analyzed |
| 📐 Face the camera directly | Steep angles reduce eyebrow symmetry accuracy |
| 📷 Use high-quality images/video | Blur reduces texture and density detection |
| 🎞️ Use longer videos | More frames = stronger consistency score |

---

## 🤝 Contributing

Contributions are welcome! Here are some ideas:

- [ ] Add a real MesoNet / XceptionNet model integration
- [ ] Add PDF/CSV export for analysis reports
- [ ] Add face landmark detection for more precise eyebrow extraction
- [ ] Add batch image processing mode
- [ ] Improve multi-face support

To contribute:

```bash
git fork https://github.com/your-username/deepfake-verification-system.git
git checkout -b feature/your-feature-name
git commit -m "Add your feature"
git push origin feature/your-feature-name
# Open a Pull Request
```

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This tool is intended for **educational and research purposes only**. It uses heuristic methods and classical computer vision — not a certified forensic tool. Do not use results as sole evidence in legal or professional decisions. Accuracy depends on image quality, lighting, and face visibility.

---

## 🙏 Acknowledgements

- [OpenCV](https://opencv.org/) — Computer vision library
- [Streamlit](https://streamlit.io/) — App framework
- Haar Cascade classifiers from the OpenCV model zoo

---

<p align="center">Made with ❤️ using Python & OpenCV</p>
