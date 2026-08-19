from collections import deque


class PredictionStabilizer:

    def __init__(
        self,
        window_size=5,
        min_confidence=0.70
    ):
        self.window_size = window_size
        self.min_confidence = min_confidence

        self.predictions = deque(
            maxlen=window_size
        )

        self.last_confirmed = None
        self.waiting_for_release = False

    def update(self, letter, confidence):

        # Ignore low-confidence predictions
        if confidence < self.min_confidence:
            self.predictions.clear()
            return None

        # If we're waiting for the previous
        # gesture to be released, don't accept
        # another prediction yet.
        if self.waiting_for_release:
            return None

        # Add current prediction
        self.predictions.append(letter)

        # Need enough stable predictions
        if len(self.predictions) < self.window_size:
            return None

        # Check if all predictions are identical
        if len(set(self.predictions)) == 1:

            current_letter = self.predictions[-1]

            self.last_confirmed = current_letter

            # Wait until the hand/sign is released
            self.waiting_for_release = True

            self.predictions.clear()

            return current_letter

        return None

    def release(self):
        """
        Allow the next gesture to be accepted.
        Call this when no hand is detected.
        """

        self.waiting_for_release = False
        self.predictions.clear()

    def reset(self):

        self.predictions.clear()
        self.last_confirmed = None
        self.waiting_for_release = False