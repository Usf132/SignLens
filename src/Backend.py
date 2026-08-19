import cv2
import numpy as np
import os
import json
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

from yolo_detector import YOLOHandDetector
from landmark import extract_landmarks, extract_features
from recognizer import SignRecognizer
from stabilizer import PredictionStabilizer
from sentence import SentenceBuilder


load_dotenv()

client = ElevenLabs(api_key=os.getenv('ELEVENLABS_API_KEY'))

yolo_detector = YOLOHandDetector(
    model_path="models\\yolo_best_weights.pt",
    confidence=0.5
)

sign_recognizer = SignRecognizer(
    model_path="models\\landmark_model.keras",
    metadata_path="models\\landmark_meta.json"
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

    if not os.getenv('ELEVENLABS_API_KEY'):
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
        print(f'Error! {e}')
        return None
