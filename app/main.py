from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import numpy as np
import easyocr
import cv2
import os
import time
import re
from dotenv import load_dotenv
from typing import List
from fastapi.middleware.cors import CORSMiddleware

import google.generativeai as genai

from .detect import load_yolo_model, run_detection
from .depth import load_midas_model, estimate_depth
from .utils import read_image_as_np, get_distance_cm
from .speech import speak_text




last_speak_time = 0
SPEAK_INTERVAL_SECONDS = 2  # speak every 2 seconds max


load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.0-flash")

app = FastAPI()

# ✅ Add this block to enable cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace "*" with ["http://your-android-app.com"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

yolo = load_yolo_model("models/yolov8n.pt")
midas, transform = load_midas_model()

# Store last detected centers for movement detection
last_centers = {}

def is_moving(label: str, current_center: tuple):
    """
    Compare current center with last center to detect movement.
    Return True if moved more than a threshold (e.g., 10 pixels).
    """
    global last_centers
    last_center = last_centers.get(label)
    if last_center is None:
        last_centers[label] = current_center
        return False
    dist = np.linalg.norm(np.array(current_center) - np.array(last_center))
    last_centers[label] = current_center
    return dist > 10  # Threshold in pixels

def generate_navigation_instructions(detections: List[dict], movement_flags: dict):
    """
    Generate instructions using Gemini model considering if object/person is moving.
    """
    if not detections:
        return "No objects detected."

    step_ratio_str = os.getenv("STEP_RATIO", "75")
    step_ratio = float(step_ratio_str.split()[0])  # Take only first token before any spaces/comments


    description_lines = []
    for det in detections:
        label = det["label"]
        dist_cm = det["distance_cm"]
        steps = round(dist_cm / step_ratio, 1)
        moving = movement_flags.get(label, False)
        state = "moving" if moving else "stationary"
        description_lines.append(f"{label} is {state} at {steps} steps ({dist_cm} cm) ahead")

    prompt = (
        "You are a navigation assistant for visually impaired users. "
        "Given the following detected objects, their movement status, and distance in steps and centimeters, "
        "generate clear, concise, and precise navigation steps to help the user avoid obstacles and move safely:\n"
        + "\n".join(description_lines)
    )

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"LLM error: {str(e)}"
    
def extract_clean_instruction(summary):
    """
    Extracts the core navigation instruction from Gemini output.
    Prioritizes text inside ** ** if available, otherwise extracts after colon.
    """
    bold_match = re.search(r"\*\*(.*?)\*\*", summary)
    if bold_match:
        return bold_match.group(1).strip()
    
    colon_split = summary.split(":", 1)
    if len(colon_split) > 1:
        return colon_split[1].strip()
    
    return summary.strip()
    

@app.post("/detect")
async def detect_object(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = read_image_as_np(contents)

    detections = run_detection(yolo, np_img)
    depth_map = estimate_depth(midas, transform, np_img)

    results = []
    movement_flags = {}

    for det in detections:
        label, _, center = det["label"], det["box"], det["center"]
        cx, cy = center
        if 0 <= cx < np_img.shape[1] and 0 <= cy < np_img.shape[0]:
            dist_cm = get_distance_cm(depth_map, cx, cy)
        else:
            dist_cm = 999

        results.append({"label": label, "distance_cm": int(dist_cm)})
        movement_flags[label] = is_moving(label, center)

    summary = generate_navigation_instructions(results, movement_flags)
    clean_summary = extract_clean_instruction(summary)
    speak_text(clean_summary)


    # Speak output (optional)
    # for item in results:
    #     speak_text(f"{item['label']} ahead at {item['distance_cm']} centimeters")

    # ✅ Move this inside the function
    response_payload = {
    "detections": results,
    "movement_flags": {k: bool(v) for k, v in movement_flags.items()},  # ensures native bools
    "summary": summary
}

    return JSONResponse(content=jsonable_encoder(response_payload))


# Text to Speech module





from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import re
import os
from dotenv import load_dotenv

import google.generativeai as genai
import easyocr

from .utils import read_image_as_np  # utility to convert bytes to numpy image
from .speech import speak_text        # your TTS function



# Initialize EasyOCR
reader = easyocr.Reader(['en'])

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-flash")



def get_gemini_summary(raw_text: str) -> str:
    prompt = (
        f"You are a helpful assistant. Given the product label or packaging text: '{raw_text}', "
        "generate one natural and human-understandable sentence that clearly tells what the product is, "
        "what brand it belongs to, and its features or use. Be specific and avoid vague answers."
    )
    try:
        response = gemini_model.generate_content(prompt)
        full_response = response.text.strip()
        sentences = re.split(r'(?<=[.!?]) +', full_response)
        return ' '.join(sentences[:2])  # limit to first 1-2 sentences
    except Exception as e:
        return f"Gemini Error: {e}"

@app.post("/read")
async def read_and_explain(file: UploadFile = File(...)):
    contents = await file.read()
    np_img = read_image_as_np(contents)

    if np_img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image file"})

    detected_texts = reader.readtext(np_img, detail=0)
    detected_text = ' '.join(detected_texts).strip()

    if not detected_text:
        # No speech here
        return JSONResponse(content={"message": "No text detected"})

    # Speak only if text detected
    speak_text(detected_text)

    summary = get_gemini_summary(detected_text)
    speak_text(summary)

    return JSONResponse({
        "detected_text": detected_text,
        "summary": summary
    })
