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

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIG
OCR_API_KEY = os.getenv("OCR_API_KEY", "K85146131088957")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")

app = FastAPI(title="MarkSheet AI Parser (Reasoning Edition)")

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
    base64_image = encode_image(image_bytes)
    
    prompt = f"""
You are an expert marksheet parser. Use the OCR text as your primary reference and the image for spatial verification.
Extract the student name, registration ID, every subject (code, title, credits, grade), and the final GPA.

OCR TEXT:
{ocr_text}

JSON FORMAT:
{{
  "name": "Full Student Name",
  "registration_no": "Registration No",
  "subjects": [
    {{"code": "Course Code", "title": "Title", "credits": "Credits", "grade": "Grade"}}
  ],
  "gpa": "GPA"
}}

Return ONLY the JSON.
"""

    # Using Nvidia Vision-Language model as primary (requested by user)
    models_to_try = [
        "nvidia/nemotron-nano-12b-v2-vl:free",
        "google/gemma-4-26b-a4b-it:free",
        "google/gemma-4-31b-it:free",
        "liquid/lfm-2.5-1.2b-thinking:free",
        "qwen/qwen3-next-80b-a3b-instruct:free"
    ]
    
    for model_id in models_to_try:
        try:
            logger.info(f"Attempting reasoning-enabled extraction with {model_id}...")
            payload = {
                "model": model_id,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                "reasoning": {"enabled": True}
            }
            
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            }
            
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=90)
            res_json = response.json()
            
            if "choices" not in res_json:
                logger.warning(f"Model {model_id} error: {res_json.get('error')}")
                continue
                
            choice = res_json["choices"][0]
            if "reasoning_details" in choice["message"]:
                logger.debug(f"AI Reasoning: {choice['message']['reasoning_details']}")
                
            response_text = choice["message"]["content"]
            match = re.search(r"\{.*\}", response_text, re.DOTALL)
            if match:
                logger.info(f"Successfully used {model_id}!")
                return json.loads(match.group())
                
        except Exception as e:
            logger.warning(f"Failed with {model_id}: {e}")
            continue
            
    raise ValueError("All reasoning-enabled models failed.")

# API ENDPOINTS
@app.post("/parse-marksheet", response_model=MarkSheetData)
async def parse_marksheet(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        ocr_text = run_ocr(contents)
        print(f"\n--- OCR TEXT ---\n{ocr_text}\n---------------\n")
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
    <title>AI MarkSheet Reasoning Parser</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root { --primary: #8b5cf6; --bg: #030712; --card: #111827; --text: #f9fafb; }
        body { font-family: 'Outfit', sans-serif; background: var(--bg); color: var(--text); min-height: 100vh; display: flex; align-items: center; justify-content: center; margin: 0; }
        .box { width: 90%; max-width: 900px; background: var(--card); border: 1px solid #1f2937; border-radius: 20px; padding: 30px; }
        .dropzone { border: 2px dashed #374151; border-radius: 12px; padding: 40px; text-align: center; cursor: pointer; transition: 0.3s; }
        .dropzone:hover { border-color: var(--primary); background: #1f2937; }
        .btn { background: var(--primary); color: white; border: none; padding: 12px 30px; border-radius: 8px; font-weight: 600; cursor: pointer; margin-top: 20px; width: 100%; transition: 0.2s; }
        .btn:hover { background: #7c3aed; }
        #results { margin-top: 30px; display: none; }
        table { width: 100%; border-collapse: collapse; margin-top: 20px; font-size: 0.9em; }
        th, td { padding: 12px; border-bottom: 1px solid #1f2937; text-align: left; }
        .loader { display: none; text-align: center; margin: 20px 0; color: #a78bfa; }
    </style>
</head>
<body>
    <div class="box">
        <h1 style="margin-top:0;">Brainy MarkSheet Parser</h1>
        <p style="color:#9ca3af;">Using Deep Reasoning LLMs via OpenRouter</p>
        <form id="pForm">
            <div class="dropzone" onclick="document.getElementById('f').click()">
                <span id="label">Select MarkSheet Scan</span>
                <input type="file" id="f" style="display:none" accept="image/*" onchange="document.getElementById('label').innerText=this.files[0].name">
            </div>
            <button type="submit" class="btn">Start AI Reasoning Pipeline</button>
        </form>
        <div class="loader" id="l">⚡ AI is Reasoning through your document...</div>
        <div id="results">
            <h2 id="n" style="margin-bottom:5px;"></h2>
            <p id="r" style="color:#9ca3af; margin-top:0;"></p>
            <table>
                <thead><tr><th>Code</th><th>Subject</th><th>Credits</th><th>Grade</th></tr></thead>
                <tbody id="b"></tbody>
            </table>
            <h4 style="margin-top:20px;">GPA: <span id="g"></span></h4>
        </div>
    </div>
    <script>
        document.getElementById('pForm').onsubmit = async (e) => {
            e.preventDefault();
            const f = document.getElementById('f').files[0];
            if(!f) return;
            document.getElementById('l').style.display='block';
            document.getElementById('results').style.display='none';
            const fd = new FormData(); fd.append('file', f);
            try {
                const r = await fetch('/parse-marksheet', {method:'POST', body:fd});
                const d = await r.json();
                document.getElementById('n').innerText = d.name;
                document.getElementById('r').innerText = 'Reg No: ' + d.registration_no;
                document.getElementById('g').innerText = d.gpa;
                document.getElementById('b').innerHTML = d.subjects.map(s => `<tr><td>${s.code}</td><td>${s.title}</td><td>${s.credits}</td><td>${s.grade}</td></tr>`).join('');
                document.getElementById('results').style.display='block';
            } catch(e) { alert('Thinking error: ' + e); }
            finally { document.getElementById('l').style.display='none'; }
        }
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
