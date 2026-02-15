# Photo Classifier

An AI-powered image classification pipeline that automatically analyzes and sorts images from a `dataset/` directory into categorized subfolders within `output/`. It uses multiple state-of-the-art models for face analysis, blur detection, and object recognition.

## 🚀 Features

The system evaluates each image against several criteria and can copy an image into **multiple** folders simultaneously:

- **📸 Face Detection**: Categorizes images into `solo` (1 person) or `group` (multiple people).
- **😊 Emotion Recognition**: Detects smiles (happy emotion) and moves images to `smiling`.
- **😴 Blink Detection**: Uses Eye Aspect Ratio (EAR) to detect closed eyes, saving them to `eyes_closed`.
- **🐻 Animal Detection**: Identifies animals (dogs, cats, birds, etc.) using YOLOv8, saving them to `animals`.
- **🌫️ Blur Detection**: Uses a multi-metric approach (Laplacian + FFT + Brenner) to filter out `blurred` images.
- **🚫 Human-Free**: Images with no humans or animals are moved to `no_human`.
- **📊 Detailed Metadata**: Generates a `data.json` file with facial analysis (emotion, age range, gender) for all detected persons.

## 🛠️ Models & Technologies

- **[DeepFace](https://github.com/serengil/deepface)**: Used for face detection (RetinaFace backend), emotion recognition, and demographic analysis.
- **[MediaPipe](https://github.com/google-ai-edge/mediapipe)**: Utilized for high-precision face landmarker detection to calculate EAR for eye closure.
- **[YOLOv8](https://github.com/ultralytics/ultralytics)**: Employs `yolov8n.pt` for efficient and accurate animal detection.
- **OpenCV & NumPy**: Core libraries for image processing and numerical calculations.

## 📋 Installation

1. **Clone the repository** (if applicable).
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: On first run, the script will automatically download the necessary model weights (`yolov8n.pt`, `face_landmarker.task`, and DeepFace weights).*

## 📖 Usage

1. Place your images in the `dataset/` directory (supported formats: `.png`, `.jpg`, `.jpeg`).
2. Run the classifier:
   ```bash
   python image_classifier.py
   ```
3. Check the results in the `output/` directory and the generated `data.json`.

## 📂 Output Structure

```text
├── output/
│   ├── animals/       # Images containing animals
│   ├── blurred/       # Images failing the sharpness threshold
│   ├── eyes_closed/   # Images where at least one person has eyes closed
│   ├── group/         # Images with 2+ people
│   ├── no_human/      # Images with no people or animals
│   ├── solo/          # Images with exactly 1 person
│   └── smiling/       # Images where a smile is detected
└── data.json          # Comprehensive facial analysis metadata
```

### Age Categorization Logic
The system maps numerical ages to descriptive categories in `data.json`:
- **0–2**: Infant/Toddler
- **3–5**: Preschool Child
- **6–9**: Child
- **10–14**: Child/Youth
- **15–19**: Teen/Youth
- **20–24**: Young Adult
- **25–64**: Adult
- **65+**: Senior
