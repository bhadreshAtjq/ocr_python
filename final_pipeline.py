import requests
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from PIL import Image
import torch
import json
import re
import os

# ------------------------
# CONFIG
# ------------------------
IMAGE_PATH = "marksheet.png"
OCR_API_KEY = "K85146131088957"  # free OCR.space key
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# ------------------------
# STEP 1: OCR
# ------------------------
def run_ocr(image_path):
    print(f"Sending {image_path} to OCR.space...")
    url = "https://api.ocr.space/parse/image"

    payload = {
        "apikey": OCR_API_KEY,
        "language": "eng",
        "isTable": True, # Optimized for marksheets
    }

    with open(image_path, "rb") as f:
        files = {
            "file": f
        }
        response = requests.post(url, data=payload, files=files)
    
    result = response.json()

    if result.get("OCRExitCode") != 1:
        print(f"OCR Error: {result.get('ErrorMessage')}")
        return ""

    try:
        text = result["ParsedResults"][0]["ParsedText"]
    except (KeyError, IndexError):
        text = ""

    return text


# ------------------------
# STEP 2: LOAD QWEN
# ------------------------
print(f"Loading model {MODEL_ID}...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

# ------------------------
# STEP 3: RUN QWEN
# ------------------------
def run_qwen(image_path, ocr_text):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": f"You are an expert document parser.\n\nExtract structured data from this marksheet. \n\nOCR TEXT:\n{ocr_text}\n\nReturn ONLY valid JSON in this format:\n\n{{\n  \"name\": \"\",\n  \"registration_no\": \"\",\n  \"subjects\": [\n    {{\n      \"code\": \"\",\n      \"title\": \"\",\n      \"credits\": \"\",\n      \"grade\": \"\"\n    }}\n  ],\n  \"gpa\": \"\"\n}}"}
            ]
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # Note: Qwen2-VL specific generation might need qwen_vl_utils if doing complex things
    # but the transformers implementation works as well.

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=1024)

    # Decoding logic for Qwen2-VL in transformers
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(inputs.input_ids, output)
    ]
    result = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    return result


# ------------------------
# STEP 4: CLEAN JSON
# ------------------------
def extract_json(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception as e:
            print(f"JSON Parsing Error: {e}")
            return match.group()
    return text


# ------------------------
# MAIN
# ------------------------
def main():
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: {IMAGE_PATH} not found. Please ensure the image exists.")
        return

    print("Running OCR...")
    ocr_text = run_ocr(IMAGE_PATH)
    
    if not ocr_text:
        print("Warning: OCR returned no text. Qwen will rely solely on visual cues.")

    print("Running Qwen...")
    raw_output = run_qwen(IMAGE_PATH, ocr_text)

    final_output = extract_json(raw_output)

    print("\nFINAL OUTPUT:\n")
    if isinstance(final_output, dict):
        print(json.dumps(final_output, indent=2))
    else:
        print(final_output)


if __name__ == "__main__":
    main()
