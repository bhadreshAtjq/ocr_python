import os
import json
import re
import requests
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from PIL import Image
import io
import uvicorn
import logging
from openai import OpenAI
from dotenv import load_dotenv

# Load secrets from .env
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIG
OCR_API_KEY = os.getenv("OCR_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OR_MODEL_ID = os.getenv("OR_MODEL_ID", "nvidia/nemotron-nano-12b-v2-vl:free")

# Initialize OpenAI-compatible client for OpenRouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

app = FastAPI(title="MarkSheet AI Parser (OpenRouter Edition)")

# DATA MODELS
class Subject(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    code: str = Field(..., description="Course code")
    title: str = Field(..., description="Course title")
    credits: str = Field(..., description="Credit hours")
    grade: str = Field(..., description="Grade awarded")

class MarkSheetData(BaseModel):
    model_config = ConfigDict(coerce_numbers_to_str=True)
    name: str = Field(..., description="Student Name")
    registration_no: str = Field(..., description="Student Registration Number")
    subjects: List[Subject]
    gpa: str = Field(..., description="Grade Point Average")

# HELPER FUNCTIONS
def compress_image(image_bytes: bytes, max_kb: int = 1000):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    if len(image_bytes) <= max_kb * 1024:
        return image_bytes
    quality = 90
    while quality > 10:
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        if len(buffer.getvalue()) <= max_kb * 1024:
            return buffer.getvalue()
        quality -= 10
    img.thumbnail((1600, 1600))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=20)
    return buffer.getvalue()

def run_ocr(image_bytes: bytes):
    compressed_bytes = compress_image(image_bytes)
    url = "https://api.ocr.space/parse/image"
    files = {"file": ("image.jpg", compressed_bytes, "image/jpeg")}
    data = {"apikey": OCR_API_KEY, "language": "eng", "isTable": True, "OCREngine": 2}
    try:
        response = requests.post(url, files=files, data=data, timeout=60)
        result = response.json()
        if result.get("OCRExitCode") != 1:
            return f"OCR Failed: {result.get('ErrorMessage')}"
        return result["ParsedResults"][0]["ParsedText"]
    except Exception as e:
        return f"OCR Error: {e}"

def encode_image(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def generate_structured_data(image_bytes: bytes, ocr_text: str):
    # Convert image to base64 for vision processing
    base64_image = encode_image(image_bytes)
    
    prompt = f"""
Extract structured marksheet data from the image. Use the OCR text as a hint if helpful.

OCR TEXT FROM PRE-PROCESSING:
{ocr_text}

JSON FORMAT:
{{
  "name": "Full Student Name",
  "registration_no": "Registration No",
  "subjects": [
    {{
      "code": "Course Code",
      "title": "Title",
      "credits": "Credits",
      "grade": "Grade"
    }}
  ],
  "gpa": "GPA"
}}

Return ONLY the JSON object.
"""

    # Robust list of visions models + reasoning models
    models_to_try = [
        "nvidia/nemotron-nano-12b-v2-vl:free",
        "google/gemma-4-26b-a4b-it:free",
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3.1-8b-instruct:free",
        "google/gemma-2-9b-it:free"
    ]
    
    last_error = None
    for model_id in models_to_try:
        try:
            logger.info(f"Attempting extraction with {model_id}...")
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ]
            )
            
            response_text = response.choices[0].message.content
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                logger.info(f"Successfully used {model_id}!")
                return json.loads(match.group())
            
        except Exception as e:
            last_error = e
            error_str = str(e).lower()
            if "429" in error_str or "rate" in error_str:
                logger.warning(f"Model {model_id} is rate-limited, trying next...")
                continue
            elif "404" in error_str:
                logger.warning(f"Model {model_id} not found, trying next...")
                continue
            else:
                logger.error(f"Unexpected error with {model_id}: {e}")
                continue
                
    raise ValueError(f"All free models failed or rate-limited. Last error: {last_error}")

# API ENDPOINTS
@app.post("/parse-marksheet", response_model=MarkSheetData)
async def parse_marksheet(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        logger.info("Step 1: Running OCR.space...")
        ocr_text = run_ocr(contents)
        print("\n--- RAW OCR TEXT ---")
        print(ocr_text)
        print("--------------------\n")
        
        logger.info(f"Step 2: Processing with {OR_MODEL_ID} via OpenRouter...")
        structured_data = generate_structured_data(contents, ocr_text)
        
        return MarkSheetData(**structured_data)
        
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MarkSheet AI Parser</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root { --primary: #6366f1; --bg: #0f172a; --card-bg: rgba(30, 41, 59, 0.7); --text: #f8fafc; }
        body { font-family: 'Outfit', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; align-items: center; justify-content: center; margin: 0; padding: 20px; }
        .container { width: 100%; max-width: 800px; background: var(--card-bg); backdrop-filter: blur(12px); border-radius: 24px; padding: 40px; border: 1px solid rgba(255,255,255,0.1); }
        h1 { background: linear-gradient(to right, #818cf8, #c084fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .upload-area { border: 2px dashed rgba(255,255,255,0.2); border-radius: 16px; padding: 40px; text-align: center; cursor: pointer; position: relative; }
        .upload-area:hover { border-color: var(--primary); }
        .upload-area input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
        .btn { background: var(--primary); color: white; border: none; padding: 12px 32px; border-radius: 12px; font-weight: 600; cursor: pointer; margin-top: 20px; }
        #result { margin-top: 40px; display: none; }
        .subject-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .subject-table th, .subject-table td { text-align: left; padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .loader { display: none; margin-top: 20px; text-align: center; }
        .spinner { width: 32px; height: 32px; border: 3px solid rgba(255,255,255,.3); border-radius: 50%; border-top-color: var(--primary); animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <h1>MarkSheet OpenRouter Parser</h1>
        <p>Using Google Gemini Flash via OpenRouter (Bypassing Quota Limits)</p>
        <form id="uploadForm">
            <div class="upload-area">
                <span style="font-size: 48px;">📄</span>
                <p id="fileName">Select marksheet image</p>
                <input type="file" name="file" id="fileInput" accept="image/*" required>
            </div>
            <button type="submit" class="btn" id="submitBtn">Extract Data</button>
        </form>
        <div class="loader" id="loader"><div class="spinner"></div><p>AI Parsing...</p></div>
        <div id="result">
            <h3 id="resName" style="color: #818cf8;"></h3>
            <p id="resReg"></p>
            <table class="subject-table" id="subjectTable">
                <thead><tr><th>Code</th><th>Subject</th><th>Credits</th><th>Grade</th></tr></thead>
                <tbody></tbody>
            </table>
            <p><strong>GPA: <span id="resGPA"></span></strong></p>
        </div>
    </div>
    <script>
        const uploadForm = document.getElementById('uploadForm');
        uploadForm.onsubmit = async (e) => {
            e.preventDefault();
            document.getElementById('loader').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            const formData = new FormData();
            formData.append('file', document.getElementById('fileInput').files[0]);
            try {
                const res = await fetch('/parse-marksheet', { method: 'POST', body: formData });
                const data = await res.json();
                document.getElementById('resName').textContent = data.name;
                document.getElementById('resReg').textContent = data.registration_no;
                document.getElementById('resGPA').textContent = data.gpa;
                document.querySelector('#subjectTable tbody').innerHTML = data.subjects.map(s => `<tr><td>${s.code}</td><td>${s.title}</td><td>${s.credits}</td><td>${s.grade}</td></tr>`).join('');
                document.getElementById('result').style.display = 'block';
            } catch (err) { alert(err); } finally { document.getElementById('loader').style.display = 'none'; }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
