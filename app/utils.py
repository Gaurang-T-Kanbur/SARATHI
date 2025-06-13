import pyttsx3
import numpy as np
import cv2



def speak_text(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(text)
    engine.runAndWait()
    
def read_image_as_np(contents):
    import numpy as np
    import cv2
    image = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(image, cv2.IMREAD_COLOR)

def read_image_as_np(contents: bytes) -> np.ndarray:
    nparr = np.frombuffer(contents, np.uint8)
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img_np

import numpy as np

def get_distance_cm(depth_map, cx, cy, patch_size=15, step_length_cm=75):
    h, w = depth_map.shape
    half = patch_size // 2
    x1, x2 = max(0, cx - half), min(w, cx + half + 1)
    y1, y2 = max(0, cy - half), min(h, cy + half + 1)
    
    patch = depth_map[y1:y2, x1:x2]
    
    # Optional: smooth patch to reduce noise
    patch_smoothed = cv2.GaussianBlur(patch, (3,3), 0)
    
    median_depth = np.median(patch_smoothed)
    
    flattened = depth_map.flatten()
    low_percentile = np.percentile(flattened, 5)
    high_percentile = np.percentile(flattened, 95)
    
    clipped_depth = np.clip(median_depth, low_percentile, high_percentile)
    
    normalized = (clipped_depth - low_percentile) / (high_percentile - low_percentile + 1e-8)
    inverted = 1.0 - normalized
    
    scaled = inverted ** 2
    
    min_distance_cm = 20
    max_distance_cm = 350
    
    distance_cm = scaled * (max_distance_cm - min_distance_cm) + min_distance_cm
    
    return int(np.clip(distance_cm, min_distance_cm, max_distance_cm))


def get_distance_steps(depth_map, cx, cy, patch_size=15, step_length_cm=75):
    distance_cm = get_distance_cm(depth_map, cx, cy, patch_size)
    distance_steps = distance_cm / step_length_cm
    return round(distance_steps, 2)
