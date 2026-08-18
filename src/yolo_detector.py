from ultralytics import YOLO


class YOLOHandDetector:

    def __init__(
        self,
        model_path="models/yolo_best_weights.pt",
        confidence=0.5
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):

        results = self.model(
            frame,
            conf=self.confidence,
            verbose=False
        )

        if not results:
            return None, None

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return None, None

        # Select the box with the highest confidence
        best_idx = result.boxes.conf.argmax().item()

        box = result.boxes.xyxy[best_idx].cpu().numpy()

        x1, y1, x2, y2 = map(int, box)

        h, w = frame.shape[:2]

        # Keep coordinates inside image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None, None

        cropped_hand = frame[y1:y2, x1:x2]

        return cropped_hand, (x1, y1, x2, y2)