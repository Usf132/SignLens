SignLens
A Computer Vision project for recognizing Sign Language using hand landmarks and deep learning.
📁 Project Structure
SignLens/
│
├── models/
│   ├── class_mapping.json
│   ├── landmark.pt
│   └── yolo_best_weights.pt
│
├── notebooks/
│   ├── train_landmark.ipynb
│   └── train_yolo_best.ipynb
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── landmark_extraction_failures.csv
│   ├── landmarks_dataset.npz
│   └── training_curves.png
│
├── src/
│   └── __init__.py
│
├── LICENSE
└── README.md

🧠 Models

landmark.pt — Landmark-based sign language classifier.
yolo_best_weights.pt — YOLO model weights.
class_mapping.json — Maps model classes to their corresponding signs.

📊 Outputs
The outputs/ directory contains training results, extracted landmark data, failed landmark extractions, and evaluation visualizations.
🛠️ Technologies

Python
Computer Vision
YOLO
Hand Landmarks
Deep Learning
PyTorch
Jupyter Notebook

