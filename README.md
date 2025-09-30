# Autoencoder-Based Anomaly Detection in Brain MRI Scans
 Project Overview

This project implements an unsupervised anomaly detection system using convolutional autoencoders to identify chronic brain infarcts and tumor-like anomalies in MRI scans. Inspired by the work of van Hespen et al. (2021), our approach trains the autoencoder only on healthy brain MRI scans. When tested with abnormal scans, higher reconstruction errors indicate potential infarcts or other anomalies.

The goal is to support radiologists by providing an automated tool that flags suspicious regions, reducing human error and improving early diagnosis in neurological disorders.

 Key Features

Autoencoder Architecture — Convolutional encoder-decoder structure with symmetric layers for reconstruction.

Unsupervised Learning — Model trained only on healthy brain scans, making it effective for anomaly detection without labeled infarct data.

Pixel-wise Anomaly Scoring — Reconstruction error (MSE) is computed for each pixel to localize anomalies.

Dynamic Thresholding — Anomaly threshold calculated as

Threshold=μ(healthy scores)+1×σ(healthy scores)

Evaluation Metrics — Precision, Recall, F1-Score, and Confusion Matrix.

Visualization — Heatmaps highlight anomalous regions in test MRI scans.

 Methodology
1. Dataset

Source: Brain MRI Images for Brain Tumor Detection – Kaggle

Size: 253 images (healthy + abnormal).

Preprocessing:

Grayscale conversion

Resizing to 128×128 pixels

Normalization

2. Model Architecture

Input: 1-channel MRI (128×128)

Encoder:

Conv2d (1→32) → ReLU

Conv2d (32→64) → ReLU

Conv2d (64→128) → ReLU

Flatten → Linear → Latent Space (256 neurons)

Decoder:

Linear → Reshape

ConvTranspose2d (128→64) → ReLU

ConvTranspose2d (64→32) → ReLU

ConvTranspose2d (32→1) → Sigmoid

Loss Function: Mean Squared Error (MSE)

3. Training

Framework: PyTorch

Optimizer: Adam (lr = 0.001)

Batch size: 32

Epochs: 50

Early Stopping: Enabled

 Results

Confusion Matrix:

|                 | Predicted No Tumor | Predicted Tumor |
| --------------- | ------------------ | --------------- |
| Actual No Tumor | 78 (TN)            | 13 (FP)         |
| Actual Tumor    | 0 (FN)             | 154 (TP)        |


Metrics:

Precision: 0.92

Recall: 1.0

F1-Score: 0.96

Observations:

Clear heatmap-based localization of anomalous regions.

Small/low-contrast infarcts remain challenging.

False positives occasionally triggered by natural brain structures.

⚠️ Limitations

Operates on 2D slices only; no volumetric context.

Threshold selection is empirical, may vary across datasets.

Domain shift across scanners or acquisition protocols can impact performance.

🔮 Future Work

Extend to 2.5D or 3D autoencoders for richer spatial understanding.

Explore semi-supervised learning to leverage limited labeled infarct data.

Apply morphological post-processing to reduce false positives.

Investigate transfer learning for generalization across diverse MRI protocols.

🛠️ Tech Stack

Language: Python 3.x

Libraries: PyTorch, NumPy, Matplotlib, OpenCV, Scikit-learn

Environment: Jupyter Notebook / Google Colab

📚 References

Van Hespen, K.M., et al. (2021). An anomaly detection approach to identify chronic brain infarcts on MRI. Scientific Reports.

Kaggle Dataset — Brain MRI Images for Brain Tumor Detection
.

Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. ACM Computing Surveys.
