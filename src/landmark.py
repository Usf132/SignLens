import cv2
import mediapipe as mp
import numpy as np


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


WRIST_IDX = 0
MIDDLE_MCP_IDX = 9


def extract_landmarks(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    annotated_frame = frame.copy()

    if not results.multi_hand_landmarks:
        return annotated_frame, None

    hand = results.multi_hand_landmarks[0]

    mp_drawing.draw_landmarks(
        annotated_frame,
        hand,
        mp_hands.HAND_CONNECTIONS
    )

    landmarks = []

    for point in hand.landmark:
        landmarks.append([
            point.x,
            point.y,
            point.z
        ])

    landmarks = np.array(
        landmarks,
        dtype=np.float32
    )

    return annotated_frame, landmarks


def extract_features(landmarks):
    if landmarks is None:
        return None

    landmarks = np.asarray(
        landmarks,
        dtype=np.float32
    )

    origin = landmarks[WRIST_IDX]
    centered = landmarks - origin

    scale_ref = np.linalg.norm(
        centered[MIDDLE_MCP_IDX]
    )

    scale_ref = max(scale_ref, 1e-6)

    normalized = centered / scale_ref

    return normalized.flatten().astype(np.float32)