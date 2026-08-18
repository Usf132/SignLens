import torch
import torch.nn as nn
import numpy as np


class LandmarkMLP(nn.Module):

    def __init__(
        self,
        input_dim=63,
        hidden_dims=(128, 64),
        num_classes=26,
        dropout=0.2
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = h

        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SignRecognizer:

    def __init__(self, model_path="models/landmark.pt"):
        self.device = torch.device("cpu")

        self.checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=False
        )

        self.model_config = self.checkpoint["model_config"]

        self.model = LandmarkMLP(
            **self.model_config
        ).to(self.device)

        self.model.load_state_dict(
            self.checkpoint["model_state_dict"]
        )

        self.model.eval()

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
        features = np.asarray(features, dtype=np.float32)

        if features.shape != (63,):
            raise ValueError(
                f"Expected 63 features, got shape {features.shape}"
            )

        features = (
            features - self.scaler_mean
        ) / self.scaler_scale

        x = torch.from_numpy(features).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted_idx = torch.max(
                probabilities,
                dim=1
            )

        letter = self.class_names[predicted_idx.item()]
        confidence = confidence.item()

        return letter, confidence