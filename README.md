# REEL/REAL

A complete local deepfake detection system featuring a web interface, a Chrome extension, a FastAPI backend connector, and a Vision Transformer detection model.

## Project Structure

- `site/`: The web application interface.
- `extension/`: The Chrome extension for browser-based checking.
- `server/`: The backend connector linking the frontend to the Python model.
- `ctf_pretrained/`: The core Vision Transformer detection pipeline and model weights.

---

## Setup & Running

Python 3.11 is required.

### 1. Install Dependencies (Run once)
```bash
cd server
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
.venv/Scripts/python.exe -m pip install "opencv-python-headless==4.10.0.84" "scikit-learn==1.5.2" "matplotlib==3.9.2"
.venv/Scripts/python.exe -m pip install --no-deps facenet-pytorch==2.6.0

### 2. Start the Backend Connector
cd server
PIPELINE_DIR=../ctf_pretrained .venv/Scripts/python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000

### 3. Start the Web Frontend
python -m http.server 8080
