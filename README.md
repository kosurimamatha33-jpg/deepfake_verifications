# Deepfake Detection for Identity Verification

## Setup
1. Clone this repo.
2. Create a virtual environment: `python -m venv venv`
3. Activate: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. (Optional) Download a pre-trained deepfake model (e.g., MesoNet) and place it as `models/mesonet.tflite`. If not present, a heuristic fallback is used.

## Run
```bash
streamlit run app.py