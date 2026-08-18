import cv2
import numpy as np
from ultralytics import YOLO
from elevenlabs.client import ElevenLabs
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

model = YOLO(r"F:\NTI(VISION)\New folder\SignLens\models\best_50_epoch.pt")

ASL_LABELS = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z"
    #26: " "

}

client = ElevenLabs(api_key="sk_a92f97f9dc2468187575c37f6c9a553e99801a24a384fe02")

def predict_asl_sign(frame, Confid=70):

    conf_value = float(Confid) / 100.0 if Confid > 1.0 else float(Confid)

    
    results = model(frame, imgsz=512, conf=conf_value, verbose=False)
    
    predicted_letter = "..."
    confidence = 0.0
    
    annotated_frame = results[0].plot()
    
    if len(results[0].boxes) > 0:
        box = results[0].boxes[0]
        class_id = int(box.cls[0])       
        confidence = float(box.conf[0])     
        predicted_letter = ASL_LABELS.get(class_id, "...") 
        
    return annotated_frame,predicted_letter, confidence

def Text_to_speech(text, voice_id):
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

    

