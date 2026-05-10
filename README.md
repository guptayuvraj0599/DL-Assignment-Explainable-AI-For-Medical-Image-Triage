# Explainable AI for Medical Image Triage

## Project Description
This project uses a CNN-based deep learning model (**MobileNetV2**) to analyze chest X-ray images and prioritize patients based on severity. The system applies **Grad-CAM (Gradient-weighted Class Activation Mapping)** to generate explainable heatmaps that highlight important regions in the X-ray images responsible for the prediction.

The project aims to assist in medical image triage by automatically detecting pneumonia cases and categorizing patients into different priority levels based on severity.

---

# Features
- Pneumonia detection using Chest X-ray images
- Lightweight CNN model using MobileNetV2
- Explainable AI using Grad-CAM
- Patient severity scoring system
- Automated patient priority list generation
- Heatmap visualization for prediction interpretability

---

# Technologies Used
- Python 3.x
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- Pillow

---

# Dataset Structure

dataset/
│
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
├── val/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── test/
    ├── NORMAL/
    └── PNEUMONIA/

---

# Dataset

- The dataset used for this project can be downloaded from:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

# Installation

## Clone the Repository
- git clone https://github.com/guptayuvraj0599/DL-Assignment-Explainable-AI-For-Medical-Image-Triage
- cd DLAssignment_PneumoniaPriorityPrediction

## Install Required Libraries
pip install tensorflow keras numpy opencv-python matplotlib scikit-learn seaborn pillow

---

# How to Run

## Step 1 - Train the Model
python main.py

This will:
- Train the MobileNetV2 model
- Save trained model files
- Generate evaluation graphs
- Calculate performance metrics

Generated files:
- model.h5
- best_model.h5
- accuracy_curve.png
- loss_curve.png
- roc_curve.png
- confusion_matrix.png

## Step 2 - Run the Explainable AI Triage System
python gradcam.py

This will:
- Predict pneumonia severity
- Generate Grad-CAM heatmaps
- Create patient priority rankings
- Save heatmap images

Generated Folder:
- gradcam_outputs/

---

# Output

The system provides:
- Disease prediction
- Confidence score
- Severity score
- Patient priority level
- Grad-CAM heatmap visualization

---

# Priority Levels

1. **High 🚨** - Urgent medical attention required
2. **Medium** - Moderate severity
3. **Low** - Normal/Non-Critical Case

---

# Model Details

1. **Model** - MobileNetV2
2. **Framework** - TensorFlow/Keras
3. **Input Size** - 224*224
4. **Optimizer** - Adam
5. **Loss Function** - Binary Crossentropy
6. **Batch Size** - 32
7. **Epochs** - 15

---

# Explainable AI (Grad-CAM)

- Grad-CAM is used to visualize the important regions of chest X-ray images influencing the model's predictions. This improves transparency and interpretability of the AI system in healthcare applications.

---

# Performance Metrics

The project evaluates the model using:
- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC
- Confusion Matrix

---

# Future Scope

- Multi-disease chest X-ray classification
- Real-time hospital integration
- Web-based deployment
- Clinical validation with larger datasets

---

# Authors
- Yuvraj Gupta (23bcs104)
- Krish Verma (23bcs044)
- Kartik Sharma (23bcs038)

- SoCSE, SMVDU, Katra