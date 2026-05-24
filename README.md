# Pneumonia & COVID-19 Detection from Chest X-rays

## Project Overview
This project uses deep learning models (ResNet, EfficientNet, and a custom CNN) to classify chest X-rays into **Normal**, **COVID-19**, and **Pneumonia** categories.  
It demonstrates end-to-end workflow: dataset preparation, preprocessing, model training, evaluation, and visualization.

## Setup
```bash
    git clone https://github.com/skaiml1307/pneumonia_covid_detection.git
    cd pneumonia_covid_detection
    python -m venv venv
    source venv/bin/activate   # Windows: venv\Scripts\activate
    pip install -r requirements.txt

Dataset:
Source: Mendeley Data — COVID‑19, Pneumonia and Normal Chest X‑ray PA Dataset
Classes:
0 → Normal
1 → COVID‑19
2 → Pneumonia

Training:
Run any of the training scripts:

```bash
    python src/train.py          # Custom CNN (grayscale)
    python src/train_resnet.py   # ResNet50 baseline
    python src/train_transfer.py # EfficientNet transfer learning

Results:
ResNet50 → ~80% accuracy
ResNet101 → ~41% accuracy (overfitting, discarded)
EfficientNetB0/B3 → ~85–88% accuracy after fine‑tuning
Custom CNN → ~75% accuracy (grayscale baseline)

Visualizations:
Confusion Matrix
Training vs Validation Accuracy
Training vs Validation Loss
(Generated via src/utils.py)

Key Skills Demonstrated:
Deep Learning (CNNs, Transfer Learning)
Medical Image Processing
TensorFlow/Keras pipelines

Modular project structure
Recruiter‑friendly documentation

