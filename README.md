# SignLens

SignLens is a computer-vision pipeline that recognizes American Sign Language (ASL) hand signs in real time and converts them into text and speech. It combines a YOLO hand detector, MediaPipe landmark extraction, and a TensorFlow classifier, wrapped in a Streamlit app with live camera translation, video upload, and text-to-speech.

## Features

- **Live Translation** — detect and translate signs from your webcam in real time, with a rolling confidence chart and adjustable stability/confidence thresholds.
- **Upload Video** — run the same pipeline over a pre-recorded video file, with pause/resume support.
- **Text to Audio** — convert any typed text to speech using ElevenLabs voices.
- **Sentence building** — recognized letters are stabilized and assembled into words/sentences, with dedicated `Space`, `Delete`, and `Clear` gestures.

## How it works

```
Camera / Video Frame
        │
        ▼
   YOLO hand detector   →  crops the hand region
        │
        ▼
  MediaPipe landmarks   →  63 (x, y, z) hand-landmark features
        │
        ▼
 TensorFlow classifier  →  predicted letter + confidence
        │
        ▼
   Prediction stabilizer →  confirms a letter only once it's stable
        │
        ▼
    Sentence builder     →  accumulates letters into words/sentences
```

## Project structure

```
SignLens/
├── app/                  # Streamlit application
│   ├── main.py           # UI: live translation, video upload, text-to-audio
│   └── backend.py        # Glue between the UI and the inference pipeline, TTS
│
├── signlens/              # Core, reusable inference package
│   ├── yolo_detector.py   # YOLO-based hand detection
│   ├── landmark.py        # MediaPipe landmark extraction + feature normalization
│   ├── recognizer.py      # TensorFlow/Keras sign classifier
│   ├── stabilizer.py      # Temporal smoothing / debouncing of predictions
│   ├── sentence.py        # Letter → word/sentence accumulation
│   └── pipeline.py        # Orchestrates the full frame → sentence pipeline
│
├── models/                 # Trained model weights and metadata
│   ├── yolo_best_weights (50 epoch).pt
│   ├── landmark.pt
│   ├── landmark_model.keras
│   ├── landmark_meta.json
│   └── class_mapping.json
│
├── notebooks/               # Training notebooks
│   ├── train_yolo_best.ipynb
│   ├── train_landmark.ipynb
│   └── train_landmark_tf.ipynb
│
├── outputs/                  # Training artifacts (plots, metrics)
│   ├── confusion_matrix.png
│   ├── training_curves.png
│   ├── confidence_threshold_sweep.png
│   └── landmark_extraction_failures.csv
│
├── assets/                    # Static assets used by the app (e.g. logo)
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```

## Getting started

### 1. Clone and set up an environment

```bash
git clone https://github.com/Usf132/SignLens.git
cd SignLens
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure your API key

Text-to-speech uses [ElevenLabs](https://elevenlabs.io/). Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
```

```
ELEVENLABS_API_KEY=your_api_key_here
```

### 3. Run the app

```bash
streamlit run app/main.py
```

The app opens in your browser. Grant camera access for **Live Translation**, or use **Upload Video** to translate a pre-recorded clip.

## Model details

| Component | Model | Purpose |
|---|---|---|
| Hand detector | YOLO (`yolo_best_weights (50 epoch).pt`) | Locates the hand in each frame |
| Landmark extractor | MediaPipe Hands | Extracts 21 landmarks (x, y, z) per hand |
| Sign classifier | TensorFlow/Keras (`landmark_model.keras`) | Classifies normalized landmarks into a letter/command |

Class labels and the feature scaler used at inference time are stored in `models/landmark_meta.json` and `models/class_mapping.json`. Training notebooks for both the YOLO detector and the landmark classifier are in `notebooks/`.

## Retraining

- `notebooks/train_yolo_best.ipynb` — trains the YOLO hand detector.
- `notebooks/train_landmark.ipynb` / `train_landmark_tf.ipynb` — trains the landmark-based sign classifier.

Training outputs (confusion matrix, training curves, confidence sweep) are saved to `outputs/`.

## Roadmap / known limitations

- Currently recognizes individual letters/commands rather than full continuous ASL sentences.
- Single-hand detection only.
- Text-to-speech requires an ElevenLabs API key and network access.

## License

Distributed under the terms of the [MIT License](LICENSE).
