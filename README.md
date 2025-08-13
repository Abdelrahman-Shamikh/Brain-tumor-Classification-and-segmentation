<a id="top"></a>

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="https://sdmntprukwest.oaiusercontent.com/files/00000000-000c-6243-ba8e-e154f6172029/raw?se=2025-08-13T16%3A23%3A18Z&sp=r&sv=2024-08-04&sr=b&scid=3e58cce3-032e-59b8-875c-2543f1890b6d&skoid=f71d6506-3cac-498e-b62a-67f9228033a9&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2025-08-13T11%3A08%3A57Z&ske=2025-08-14T11%3A08%3A57Z&sks=b&skv=2024-08-04&sig=%2BLaAu0DGOY3jUIKoyTEu3VJfHrjEUPKzXA007dcminc%3D" width="150" alt="Brain Icon"/>

# BRAIN-TUMOR-CLASSIFICATION-AND-SEGMENTATION

<em>Built with the tools and technologies:</em>

<br />

[![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-F05032?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white)](https://kaggle.com/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

<br />

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Results](#results)
- [Project Structure](#project-structure)
    - [Project Index](#project-index)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
- [Roadmap](#roadmap)
- [Acknowledgments](#acknowledgments)

---

## Overview

**Grade:** A (4.0/4.0)  
This project implements a **deep learning system** to classify brain MRI scans into **glioma, meningioma, pituitary tumor, or no tumor**, and to segment tumor regions for surgical planning.  
It also features **3D reconstruction** and **tumor volume estimation** to support early diagnosis and treatment.  
Optimized for potential **mobile deployment** to assist in real-world medical settings.

---

## Features

- **Brain Tumor Classification** using a **custom CNN model** trained from scratch.
- **Tumor Segmentation** using **Attention U-Net** for high-precision region extraction.
- **3D Reconstruction** of tumor structure from MRI slices.
- **Tumor Volume Estimation** for quantitative analysis.
- **Optimized for Mobile Deployment** to enable real-time usage in hospitals.
- **Comprehensive Documentation** with final paper, poster, and seminar slides.

---
## Dataset 
### Segmantation
#### Training & Validation

The primary dataset for training and validation is sourced from:  
[Kaggle: Brain Tumor Segmentation](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation)

- Contains MRI images and corresponding tumor masks for segmentation.  
- Used for training the segmentation model and validating during training.

#### Testing

For testing, the dataset used is:  
[Kaggle: Brain Tumor Segmentation Dataset by Atika Akter](https://www.kaggle.com/datasets/atikaakter11/brain-tumor-segmentation-dataset)

- Contains MRI images with tumor annotations reserved for model evaluation on unseen data.

## Image Processing and Data Augmentation

### B. Image Processing for Classification

All input MRI images underwent a standardized preprocessing pipeline to ensure consistency and enhance model generalization:

- **Tensor Conversion:**  
  Images were converted to PyTorch tensors to enable GPU-accelerated processing.

- **Resizing:**  
  Uniformly resized to 150×150 pixels using bilinear interpolation with anti-aliasing to preserve structural integrity during downsampling.

- **Brightness Augmentation:**  
  Applied color jitter to randomly adjust brightness to 85–115% of original values, simulating clinical imaging variations.

- **Standardization:**  
  Normalized pixel values using ImageNet statistics through a standardized pixel equation.

---

### C. Image Preprocessing for Segmentation

All MRI scans and corresponding masks underwent the following preprocessing steps:

1. **Channel Handling:**  
   Grayscale images were expanded to 3-channel format for compatibility with pretrained encoders.

2. **Resizing:**  
   Uniformly resized to 256×256 pixels using bilinear interpolation to maintain aspect ratio.

3. **Normalization:**  
   - Pixel values scaled to [0, 1] range.  
   - Standardized using mean = 0.0 and std = 1.0.

4. **Mask Processing:**  
   Converted to binary format (0 = background, 1 = tumor) for segmentation tasks.

5. **Tensor Conversion:**  
   Transformed to PyTorch tensors via `ToTensorV2`.

---

### D. Data Augmentation Strategy for Segmentation

Applied only during training to increase dataset diversity:

| Augmentation Type   | Parameters                         |
|--------------------|----------------------------------|
| **Rotation & Resizing** | Random rotations (±15°), scale (0.9–1.1) |
| **Flipping**        | Vertical/Horizontal (p=0.5)       |
| **Intensity Variation** | Brightness/Contrast (±15%)         |
| **Noise Injection** | Gaussian noise (σ = 0.05)         |
    
## Results

### Classification
- **Model:** CNN from scratch  
- **Accuracy:** **98.32%**  
- **Classes:** Glioma, Meningioma, Pituitary Tumor, No Tumor

<img src="https://github.com/user-attachments/assets/95818fe3-167f-4f68-aa47-1c68e668b17a" alt="image" width="200" />


### Segmentation
- **Model:** Attention U-Net  
- **Dice Score:** **87%**  
- **IoU:** **79.5%**

<img width="335" height="115" alt="image" src="https://github.com/user-attachments/assets/ed946802-07f3-40b7-bc66-3a5c01d986d6" />

---

## Project Structure

└── Brain-tumor-Classification-and-segmentation/<br>
    ├── Classification<br>
    │   └── classification-notebook.ipynb<br>
    ├── Segmentation<br>
    │   └── attention unet-20250709T132506Z-1-001<br>
    ├── Tumor size<br>
    │   └── Tumor size.ipynb<br>
    ├── Demo.mp4<br>
    ├── Final paper.pdf<br>
    ├── final poster.pdf<br>
    ├── GP Final Documentation.pdf<br>
    ├── SEMINAR 3.pdf<br>
## Getting Started

### Prerequisites

- **Programming Language:** Python (Jupyter Notebook)
- **Required Libraries:**
  - [PyTorch](https://pytorch.org/)
  - [NumPy](https://numpy.org/)
  - [OpenCV](https://opencv.org/)
  - [Matplotlib](https://matplotlib.org/)
  - [Scikit-learn](https://scikit-learn.org/stable/)
### Installation

#### Clone the repository
git clone https://github.com/Abdelrahman-Shamikh/Brain-tumor-Classification-and-segmentation.git

#### Navigate to the project folder
cd Brain-tumor-Classification-and-segmentation
### Usage
#### open kaggle 
Then open and run the notebooks:

Classification/classification-notebook.ipynb — for brain tumor classification

Segmentation/attention-unet.ipynb — for tumor segmentation

Tumor size/Tumor size.ipynb — for tumor size estimation
## Roadmap

- [x] Classification with CNN  
- [x] Segmentation with Attention U-Net  
- [x] Tumor size calculation  
- [x] 3D tumor reconstruction  
- [x] Deploy to mobile devices  
- [ ] Add web-based demo interface  
## Acknowledgments

- Supervisors & Mentors for guidance(Dr Sally Saad and T.A Mohamed Essam)(Ainshams University)
- Open-source community for dataset and tools
<div align="left">

[![Back to Top](https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square)](#top)

</div>
