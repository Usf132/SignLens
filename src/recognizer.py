import json
import numpy as np
import tensorflow as tf


class SignRecognizer:

    def __init__(
        self,
        model_path="models/landmark_model.keras",
        metadata_path="models/landmark_meta.json"
    ):
        # Load TensorFlow/Keras model
        self.model = tf.keras.models.load_model(model_path)

        # Load metadata
        with open(metadata_path, "r", encoding="utf-8") as f:
            self.checkpoint = json.load(f)

        self.model_config = self.checkpoint["model_config"]

        self.class_names = self.checkpoint["class_names"]

        self.scaler_mean = np.array(
            self.checkpoint["scaler"]["mean"],
            dtype=np.float32
        )

        self.scaler_scale = np.array(
            self.checkpoint["scaler"]["scale"],
            dtype=np.float32
        )

        print("Model loaded successfully!")
        print("Classes:", self.class_names)
        print("Input dimension:", self.model_config["input_dim"])
        print("Number of classes:", self.model_config["num_classes"])

    def predict(self, features):

        features = np.asarray(
            features,
            dtype=np.float32
        )

        if features.shape != (63,):
            raise ValueError(
                f"Expected 63 features, got shape {features.shape}"
            )

        # Apply the same scaler used during training
        features = (
            features - self.scaler_mean
        ) / self.scaler_scale

        # Add batch dimension
        x = np.expand_dims(features, axis=0)

        # TensorFlow inference
        logits = self.model(
            x,
            training=False
        ).numpy()

        # Convert logits to probabilities
        probabilities = tf.nn.softmax(
            logits,
            axis=1
        ).numpy()

        predicted_idx = int(
            np.argmax(probabilities[0])
        )

        confidence = float(
            probabilities[0][predicted_idx]
        )

        letter = self.class_names[predicted_idx]

        return letter, confidence