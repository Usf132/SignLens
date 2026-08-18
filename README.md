SignLens
A Computer Vision project for recognizing Sign Language using hand landmarks and deep learning.

### 📁 Project Structure

```
SignLens/
│
├── models/
│   ├── class_mapping.json
│   ├── landmark.pt
│   ├── landmark_model.keras
│   └── yolo_best_weights.pt
│
├── notebooks/
│   ├── train_landmark.ipynb
│   ├── train_landmark_tf.ipynb
│   └── train_yolo_best.ipynb
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── landmark_extraction_failures.csv
│   ├── landmarks_dataset.npz
│   └── training_curves.png
│
├── src/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── App.py
│   ├── Backend.py
│   ├── landmark.py
│   ├── recognizer.py
│   ├── sentence.py
│   ├── stabilizer.py
│   └── yolo_detector.py
│
├── LICENSE
├── README.md
└── requirements.txt

```
---
### 🔄 File Renaming

Src/best (1).pt → models/yolo_best_weights.pt

Src/final_project.ipynb → notebooks/train_yolo_best.ipynb

