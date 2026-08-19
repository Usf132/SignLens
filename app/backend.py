import sys
from pathlib import Path

# Make the project root importable regardless of the working directory
# Streamlit launches from (e.g. `streamlit run app/main.py` from any cwd).
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import os

import cv2
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

from signlens.yolo_detector import YOLOHandDetector
from signlens.landmark import extract_landmarks, extract_features
from signlens.recognizer import SignRecognizer
from signlens.stabilizer import PredictionStabilizer
from signlens.sentence import SentenceBuilder


load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

MODELS_DIR = ROOT_DIR / "models"

yolo_detector = YOLOHandDetector(
    model_path=MODELS_DIR / "yolo_best_weights (50 epoch).pt",
    confidence=0.5
)

sign_recognizer = SignRecognizer(
    model_path=MODELS_DIR / "landmark_model.keras",
    metadata_path=MODELS_DIR / "landmark_meta.json"
)

stabilizer = PredictionStabilizer(window_size=5, min_confidence=0.70)
sentence_builder = SentenceBuilder()


def predict_asl_sign_lm(frame_bgr, confidence_threshold):

    annotated_frame = frame_bgr.copy()
    letter = ""
    confidence = 0.0

    cropped_hand, box = yolo_detector.detect(frame_bgr)

    if cropped_hand is not None and box is not None:
        x1, y1, x2, y2 = box

        cropped_with_landmarks, landmarks = extract_landmarks(cropped_hand)

        if landmarks is not None:

            annotated_frame[y1:y2, x1:x2] = cropped_with_landmarks

            features = extract_features(landmarks)

            if features is not None:

                pred_letter, pred_conf = sign_recognizer.predict(features)

                if pred_conf >= (confidence_threshold / 100.0):
                    letter = pred_letter
                    confidence = pred_conf

        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return annotated_frame, letter, confidence


def Text_to_speech(text, voice_id):
    if not text:
        return None

    if not os.getenv("ELEVENLABS_API_KEY"):
        print("Error! ELEVENLABS_API_KEY is not set (check your .env file).")
        return None

    try:
        audio_generator = client.text_to_speech.convert(
            text=text,
            voice_id=voice_id,
            model_id="eleven_multilingual_v2"
        )
        audio_bytes = b"".join([chunk for chunk in audio_generator])
        return audio_bytes
    except Exception as e:
        print(f"Error! {e}")
        return None
