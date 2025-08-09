<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="readmeai/assets/logos/purple.svg" width="30%" alt="Project Logo"/>

# BRAIN-TUMOR-CLASSIFICATION-AND-SEGMENTATION

<em>Built with the tools and technologies:</em>

<br />

[![Python](https://img.shields.io/badge/Python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-F05032?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NiBabel](https://img.shields.io/badge/NiBabel-4E9CAE?style=for-the-badge&logo=python&logoColor=white)](https://nipy.org/nibabel/)

<br />
---
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
- [Contributing](#contributing)
- [License](#license)
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

## Results

### Classification
- **Model:** CNN from scratch  
- **Accuracy:** **98.32%**  
- **Classes:** Glioma, Meningioma, Pituitary Tumor, No Tumor

### Segmentation
- **Model:** Attention U-Net  
- **Dice Score:** **87%**  
- **IoU:** **79.5%**

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

# Clone the repository
git clone https://github.com/Abdelrahman-Shamikh/Brain-tumor-Classification-and-segmentation.git

# Navigate to the project folder
cd Brain-tumor-Classification-and-segmentation

# Install dependencies
pip install -r requirements.txt
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
<div align="right">
[![Back to Top](https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square)](#top)
</div>
