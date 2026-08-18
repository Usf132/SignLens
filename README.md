SignLens
A Computer Vision project for recognizing Sign Language using hand landmarks and deep learning.

### 📁 Project Structure

```
SignLens/
│
├── models/
│   ├── class_mapping.json
│   ├── landmark.pt
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
│   ├── __init__.py
│   ├── landmark.py
│   └── recognizer.py
│
├── LICENSE
└── README.md
```
---
### 🔄 File Renaming

Src/best (1).pt → models/yolo_best_weights.pt

Src/final_project.ipynb → notebooks/train_yolo_best.ipynb

