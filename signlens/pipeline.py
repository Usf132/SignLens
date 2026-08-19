from signlens.yolo_detector import YOLOHandDetector
from signlens.landmark import extract_landmarks, extract_features
from signlens.recognizer import SignRecognizer
from signlens.stabilizer import PredictionStabilizer
from signlens.sentence import SentenceBuilder


class SignPipeline:
    """
    Complete sign-language inference pipeline.

    Flow:
        Frame
          ↓
        YOLO
          ↓
        Hand Crop
          ↓
        MediaPipe
          ↓
        63 Features
          ↓
        TensorFlow Recognizer
          ↓
        Stabilizer
          ↓
        Sentence Builder
    """

    def __init__(
        self,
        yolo_model_path="models/yolo_best_weights (50 epoch).pt",
        landmark_model_path="models/landmark_model.keras",
        metadata_path="models/landmark_meta.json",
        window_size=5,
        min_confidence=0.70,
    ):
        # Models
        self.yolo = YOLOHandDetector(
            model_path=yolo_model_path
        )

        self.recognizer = SignRecognizer(
            model_path=landmark_model_path,
            metadata_path=metadata_path
        )

        # Post-processing
        self.stabilizer = PredictionStabilizer(
            window_size=window_size,
            min_confidence=min_confidence
        )

        self.sentence = SentenceBuilder()

    def process_frame(self, frame):
        """
        Process one webcam/video frame.

        Returns:
            dict with:
                detected: bool
                label: predicted label or None
                confidence: prediction confidence
                confirmed: newly confirmed label/command or None
                sentence: current sentence
                box: YOLO bounding box or None
        """

        # =========================
        # 1. YOLO
        # =========================

        crop, box = self.yolo.detect(frame)

        if crop is None:
            # No hand → allow next gesture
            self.stabilizer.release()

            return {
                "detected": False,
                "label": None,
                "confidence": 0.0,
                "confirmed": None,
                "sentence": self.sentence.get_text(),
                "box": None,
            }

        # =========================
        # 2. MediaPipe
        # =========================

        _, landmarks = extract_landmarks(crop)

        if landmarks is None:
            return {
                "detected": True,
                "label": None,
                "confidence": 0.0,
                "confirmed": None,
                "sentence": self.sentence.get_text(),
                "box": box,
            }

        # =========================
        # 3. Feature Extraction
        # =========================

        features = extract_features(landmarks)

        if features is None:
            return {
                "detected": True,
                "label": None,
                "confidence": 0.0,
                "confirmed": None,
                "sentence": self.sentence.get_text(),
                "box": box,
            }

        # =========================
        # 4. TensorFlow Recognition
        # =========================

        label, confidence = self.recognizer.predict(features)

        # =========================
        # 5. Stabilization
        # =========================

        confirmed = self.stabilizer.update(
            label,
            confidence
        )

        # =========================
        # 6. Sentence / Commands
        # =========================

        if confirmed is not None:

            if confirmed == "Space":
                self.sentence.add_space()

            elif confirmed == "Delete":
                self.sentence.backspace()

            elif confirmed == "Clear":
                self.sentence.clear()

            else:
                self.sentence.add_letter(confirmed)

        return {
            "detected": True,
            "label": label,
            "confidence": confidence,
            "confirmed": confirmed,
            "sentence": self.sentence.get_text(),
            "box": box,
        }

    def get_sentence(self):
        """Return the current sentence."""
        return self.sentence.get_text()

    def clear_sentence(self):
        """Clear the current sentence and reset stabilizer state."""
        self.sentence.clear()
        self.stabilizer.reset()

    def release(self):
        """Release the current gesture."""
        self.stabilizer.release()

    def reset(self):
        """Reset the entire pipeline state."""
        self.stabilizer.reset()
        self.sentence.clear()
