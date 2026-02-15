"""
Image Classifier
=================
Iterates over every image in ``dataset/``, classifies it using multiple
detection models, and copies it into the appropriate ``output/`` subfolder(s).

An image can land in **multiple** folders simultaneously.

Models used
-----------
* Blur   – multi-metric (Laplacian + FFT + Brenner gradient) via blur_detector.py
* Faces  – DeepFace with RetinaFace backend
* Smile  – DeepFace emotion == "happy"
* Eyes   – MediaPipe FaceMesh  → Eye Aspect Ratio (EAR)
* Animals– YOLOv8n (ultralytics, COCO-trained)
"""

import os
import json
import shutil
import glob
import warnings

import cv2
import numpy as np

# Suppress noisy warnings from TF / MediaPipe / Ultralytics
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

from deepface import DeepFace
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

from blur_detector import is_blurred, compute_blur_score

# ─── Paths ────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR   = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR    = os.path.join(BASE_DIR, "output")
DATA_JSON     = os.path.join(BASE_DIR, "data.json")

SUBFOLDERS = ["blurred", "solo", "group", "smiling", "eyes_closed", "animals", "no_human"]

# ─── COCO animal class names ─────────────────────────────────────────
ANIMAL_CLASSES = {
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}

# ─── EAR threshold for eyes-closed detection ─────────────────────────
EAR_THRESHOLD = 0.20

# ─── Model file for FaceLandmarker ───────────────────────────────────
FACE_LANDMARKER_MODEL = os.path.join(BASE_DIR, "face_landmarker.task")
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

# ─── Globals (loaded once) ───────────────────────────────────────────
_yolo_model     = None
_face_landmarker = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        print("[INFO] Loading YOLOv8n model …")
        _yolo_model = YOLO("yolov8n.pt")
    return _yolo_model


def _ensure_face_landmarker_model():
    """Download the face_landmarker.task model if it doesn't exist."""
    if not os.path.exists(FACE_LANDMARKER_MODEL):
        import urllib.request
        print("[INFO] Downloading face_landmarker.task …")
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, FACE_LANDMARKER_MODEL)
        print("[INFO] Download complete.")


def _get_face_landmarker():
    global _face_landmarker
    if _face_landmarker is None:
        _ensure_face_landmarker_model()
        base_options = mp_python.BaseOptions(
            model_asset_path=FACE_LANDMARKER_MODEL
        )
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=20,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            output_face_blendshapes=False,
        )
        _face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
    return _face_landmarker


# ─── Utility helpers ─────────────────────────────────────────────────

def _copy_to(image_path: str, subfolder: str) -> None:
    dst_dir = os.path.join(OUTPUT_DIR, subfolder)
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy2(image_path, dst_dir)


def _eye_aspect_ratio(landmarks, indices):
    """Compute EAR for one eye given landmark indices.
    landmarks is a list of NormalizedLandmark with .x, .y, .z
    """
    pts = np.array([(landmarks[i].x, landmarks[i].y) for i in indices])
    # Vertical distances
    v1 = np.linalg.norm(pts[1] - pts[5])
    v2 = np.linalg.norm(pts[2] - pts[4])
    # Horizontal distance
    hz = np.linalg.norm(pts[0] - pts[3])
    return (v1 + v2) / (2.0 * hz + 1e-6)


# MediaPipe FaceMesh eye landmark indices
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# ─── Detection functions ─────────────────────────────────────────────

def detect_faces_deepface(image_path: str):
    """
    Run DeepFace.analyze to get face analysis (emotion, age, gender).
    Returns a list of face-analysis dicts, or an empty list if no faces.
    """
    try:
        analysis = DeepFace.analyze(
            img_path=image_path,
            actions=["emotion", "age", "gender"],
            detector_backend="retinaface",
            enforce_detection=True,
            silent=True,
        )
        if isinstance(analysis, dict):
            analysis = [analysis]
        return analysis
    except Exception:
        return []


def has_animals(image_path: str) -> bool:
    """Return True if YOLOv8 detects any animal in the image."""
    model = _get_yolo()
    results = model(image_path, verbose=False)
    for r in results:
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])]
            if cls_name in ANIMAL_CLASSES:
                return True
    return False


def check_eyes_closed(image_path: str) -> bool:
    """
    Use MediaPipe FaceLandmarker + EAR to check if any face has closed eyes.
    Returns True if at least one face has eyes closed.
    """
    img = cv2.imread(image_path)
    if img is None:
        return False
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    landmarker = _get_face_landmarker()
    result = landmarker.detect(mp_image)

    if not result.face_landmarks:
        return False

    for face_lm in result.face_landmarks:
        left_ear  = _eye_aspect_ratio(face_lm, _LEFT_EYE)
        right_ear = _eye_aspect_ratio(face_lm, _RIGHT_EYE)
        avg_ear = (left_ear + right_ear) / 2.0
        if avg_ear < EAR_THRESHOLD:
            return True
    return False


def is_smiling(face_analysis: dict) -> bool:
    """Return True if the dominant emotion is 'happy'."""
    return face_analysis.get("dominant_emotion", "").lower() == "happy"


# ─── specialOps ──────────────────────────────────────────────────────

# Named-color lookup table  (BGR order for OpenCV)
_COLOR_TABLE = [
    ((0,   0,   0),   "black"),
    ((255, 255, 255), "white"),
    ((128, 128, 128), "gray"),
    ((0,   0,   200), "red"),
    ((200, 0,   0),   "blue"),
    ((0,   128, 0),   "green"),
    ((0,   255, 255), "yellow"),
    ((0,   165, 255), "orange"),
    ((203, 192, 255), "pink"),
    ((128, 0,   128), "purple"),
    ((42,  42,  165), "brown"),
    ((185, 218, 245), "beige"),
    ((0,   128, 128), "olive"),
    ((128, 128, 0),   "teal"),
    ((0,   0,   128), "maroon"),
    ((235, 206, 135), "light blue"),
]


def _nearest_color_name(bgr):
    """Map a BGR tuple to the closest human-readable colour name."""
    best_name = "unknown"
    best_dist = float("inf")
    for ref_bgr, name in _COLOR_TABLE:
        dist = sum((a - b) ** 2 for a, b in zip(bgr, ref_bgr))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def _get_dominant_color(image, region: dict) -> str:
    """
    Extract the torso region below a detected face and return the dominant
    colour name via K-means (k=3).

    Parameters
    ----------
    image  : BGR numpy array (full image)
    region : dict with keys x, y, w, h  (face bounding box from DeepFace)

    Returns
    -------
    str – human-readable colour name, or ``"unknown"``.
    """
    h_img, w_img = image.shape[:2]
    fx, fy, fw, fh = region["x"], region["y"], region["w"], region["h"]

    # Torso crop: start just below the face, extend 1.5× face-height down,
    # widen by 30 % on each side.
    expand = int(fw * 0.3)
    x1 = max(fx - expand, 0)
    x2 = min(fx + fw + expand, w_img)
    y1 = min(fy + fh, h_img)
    y2 = min(fy + fh + int(fh * 1.5), h_img)

    if y2 - y1 < 10 or x2 - x1 < 10:
        return "unknown"

    crop = image[y1:y2, x1:x2]
    pixels = crop.reshape(-1, 3).astype(np.float32)

    if len(pixels) < 30:
        return "unknown"

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centres = cv2.kmeans(
        pixels, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    # Pick the cluster with the most pixels
    counts = np.bincount(labels.flatten())
    dominant_bgr = centres[counts.argmax()].astype(int)

    return _nearest_color_name(tuple(dominant_bgr))


def _get_dress_type(image, region: dict) -> str:
    """
    Heuristic dress-type classification based on colour uniformity
    in the torso region.

    Returns one of: ``"formal"``, ``"casual"``, ``"patterned"``,
    or ``"unknown"``.
    """
    h_img, w_img = image.shape[:2]
    fx, fy, fw, fh = region["x"], region["y"], region["w"], region["h"]

    expand = int(fw * 0.3)
    x1 = max(fx - expand, 0)
    x2 = min(fx + fw + expand, w_img)
    y1 = min(fy + fh, h_img)
    y2 = min(fy + fh + int(fh * 1.5), h_img)

    if y2 - y1 < 10 or x2 - x1 < 10:
        return "unknown"

    crop = image[y1:y2, x1:x2]
    hsv  = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0].flatten().astype(np.float32)
    s_channel = hsv[:, :, 1].flatten().astype(np.float32)

    if len(h_channel) < 30:
        return "unknown"

    hue_std = np.std(h_channel)
    sat_mean = np.mean(s_channel)

    # High hue variation → patterned / multi-colour
    if hue_std > 35:
        return "patterned"
    # Low saturation + low hue variation → formal (dark suit, white shirt, etc.)
    if sat_mean < 50 and hue_std < 15:
        return "formal"
    # Otherwise classify as casual
    return "casual"


def _get_age_category(age_val):
    """
    Map numerical age to descriptive category.
    | 0–2:   Infant/Toddler
    | 3–5:   Preschool Child
    | 6–9:   Child
    | 10–14: Child/Youth
    | 15–19: Teen/Youth
    | 20–24: Young Adult
    | 25–64: Adult
    | 65+:   Senior
    """
    try:
        age_int = int(age_val)
    except (ValueError, TypeError):
        # Return raw value as string if it's not a number (e.g. "unknown")
        return str(age_val)

    if age_int <= 2:
        return "Infant/Toddler"
    elif age_int <= 5:
        return "Preschool Child"
    elif age_int <= 9:
        return "Child"
    elif age_int <= 14:
        return "Child/Youth"
    elif age_int <= 19:
        return "Teen/Youth"
    elif age_int <= 24:
        return "Young Adult"
    elif age_int <= 64:
        return "Adult"
    else:
        return "Senior"


def special_ops(image_path: str, analysis: list) -> dict:
    """
    Build a JSON-serialisable record for one image that contains persons.

    Returns
    -------
    dict  with keys: filename, faces (list of {face_number, dominant_emotion,
          age, gender, dress_color, dress_type})
    """
    # Load the image once for dress analysis
    img = cv2.imread(image_path)

    faces = []
    for i, face in enumerate(analysis):
        age_val = face.get("age", "unknown")
        age_cat = _get_age_category(age_val)

        # Dress attributes (need the face bounding-box region)
        region = face.get("region", {})
        if img is not None and region:
            dress_color = _get_dominant_color(img, region)
            dress_type  = _get_dress_type(img, region)
        else:
            dress_color = "unknown"
            dress_type  = "unknown"

        face_record = {
            "face_number": i + 1,
            "dominant_emotion": face.get("dominant_emotion", "unknown"),
            "age": age_cat,
            "gender": face.get("dominant_gender", "unknown"),
            "dress_color": dress_color,
            "dress_type": dress_type,
        }
        faces.append(face_record)

        # Console output
        print(f"  Face {i + 1}:")
        print(f"    Dominant Emotion : {face_record['dominant_emotion']}")
        print(f"    Probable Age     : {face_record['age']}")
        print(f"    Gender           : {face_record['gender']}")
        print(f"    Dress Colour     : {face_record['dress_color']}")
        print(f"    Dress Type       : {face_record['dress_type']}")
        print("    " + "-" * 30)

    return {
        "filename": os.path.basename(image_path),
        "faces": faces,
    }


# ─── Main classification pipeline ────────────────────────────────────

def classify_image(image_path: str) -> dict | None:
    """
    Classify a single image and copy it to the relevant output subfolders.

    Returns the specialOps record if the image contains persons, else None.
    """
    fname = os.path.basename(image_path)
    print(f"\n{'=' * 50}")
    print(f"Processing: {fname}")
    print(f"{'=' * 50}")

    # 1. Blur detection
    blur_score = compute_blur_score(image_path)
    blurred = blur_score < 0.30
    if blurred:
        _copy_to(image_path, "blurred")
        print(f"  [BLURRED]  score={blur_score:.4f}")

    # 2. Face / person detection
    analysis = detect_faces_deepface(image_path)
    person_count = len(analysis)

    if person_count > 0:
        # Solo / group
        if person_count == 1:
            _copy_to(image_path, "solo")
            print(f"  [SOLO]     1 person detected")
        else:
            _copy_to(image_path, "group")
            print(f"  [GROUP]    {person_count} persons detected")

        # Smile check (per face)
        any_smiling = any(is_smiling(face) for face in analysis)
        if any_smiling:
            _copy_to(image_path, "smiling")
            print(f"  [SMILING]  smile detected")

        # Eyes-closed check (MediaPipe EAR)
        if check_eyes_closed(image_path):
            _copy_to(image_path, "eyes_closed")
            print(f"  [EYES_CLOSED]")

        # specialOps
        record = special_ops(image_path, analysis)
        return record

    else:
        # 3. Animal detection
        if has_animals(image_path):
            _copy_to(image_path, "animals")
            print(f"  [ANIMALS]  animal(s) detected")
        else:
            _copy_to(image_path, "no_human")
            print(f"  [NO_HUMAN]")

    return None


# ─── Entry point ──────────────────────────────────────────────────────

def main():
    # Ensure output sub-folders exist
    for sf in SUBFOLDERS:
        os.makedirs(os.path.join(OUTPUT_DIR, sf), exist_ok=True)

    # Gather image files
    image_paths = sorted(
        glob.glob(os.path.join(DATASET_DIR, "*.png"))
        + glob.glob(os.path.join(DATASET_DIR, "*.jpg"))
        + glob.glob(os.path.join(DATASET_DIR, "*.jpeg"))
    )

    if not image_paths:
        print("[WARN] No images found in", DATASET_DIR)
        return

    print(f"[INFO] Found {len(image_paths)} images in dataset/")

    all_records = []  # specialOps records for images with persons

    for img_path in image_paths:
        record = classify_image(img_path)
        if record is not None:
            all_records.append(record)

    # Write data.json
    with open(DATA_JSON, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)
    print(f"\n[DONE] data.json written with {len(all_records)} records → {DATA_JSON}")

    # Summary
    print(f"\n{'=' * 50}")
    print("Classification Summary")
    print(f"{'=' * 50}")
    for sf in SUBFOLDERS:
        sf_path = os.path.join(OUTPUT_DIR, sf)
        count = len(os.listdir(sf_path)) if os.path.isdir(sf_path) else 0
        print(f"  {sf:15s}: {count} images")


if __name__ == "__main__":
    main()