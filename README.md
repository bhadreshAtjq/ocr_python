# MarkSheet OCR & Parsing Pipeline

This project implements a high-accuracy OCR and information extraction pipeline using **OCR.space** for text extraction and **Qwen2-VL** for structured JSON generation.

## 🚀 Features
- **Hybrid OCR**: Combines traditional OCR (clean text) with Vision-LLM (contextual understanding).
- **Structured Output**: Guarantees JSON output for easy integration.
- **Production Ready**: Includes a FastAPI wrapper with Pydantic validation.
- **Mock Data**: Includes a `marksheet_mockup.png` for immediate testing.

## 📦 Installation

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🛠 Usage

### 1. Basic Script
Run the basic pipeline on the mockup image:
```bash
python final_pipeline.py
```
*Note: Ensure you have an internet connection for OCR.space and enough disk space (~5GB) for the Qwen model.*

### 2. Production API
Start the FastAPI server:
```bash
python production_pipeline.py
```
Then send an image to `http://localhost:8000/parse-marksheet`:
```bash
curl -X 'POST' \
  'http://localhost:8000/parse-marksheet' \
  -H 'accept: application/json' \
  -H 'Content-Type: multipart/form-data' \
  -F 'file=@marksheet.png;type=image/png'
```

## ⚙️ Configuration
- **OCR_API_KEY**: Set your OCR.space API key in the script or as an environment variable.
- **MODEL_ID**: Default is `Qwen/Qwen2-VL-2B-Instruct`. You can upgrade to `7B` if you have more VRAM.

## 📄 File Structure
- `final_pipeline.py`: Main script for standalone use.
- `production_pipeline.py`: FastAPI implementation with validation.
- `marksheet.png`: Mockup marksheet for testing.
- `requirements.txt`: Project dependencies.
