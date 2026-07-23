Product Requirements Document (PRD)
Core Needle Biopsy AI Model Project

Version: 1.0
Status: Draft
Project Type: AI/Deep Learning for Breast Cancer Histopathology Image Analysis

1. Project Overview
Project Name

Core Needle Biopsy AI Model Project

Purpose

The Core Needle Biopsy AI Model Project aims to develop an Artificial Intelligence system capable of classifying breast cancer histopathology images obtained from Core Needle Biopsy (CNB) Whole Slide Images (WSIs). The project focuses on improving image quality through multiple noise simulation and denoising techniques before training deep learning models.

The system will evaluate how different combinations of image degradation (noise) and restoration (filters) influence the diagnostic performance of CNN-based classification models.

2. Problem Statement

Histopathological images acquired from digital pathology scanners often suffer from various imaging artifacts and noise introduced during:

Tissue preparation
Slide staining
Scanner acquisition
Compression
Digital transmission

These quality degradations reduce the effectiveness of AI-based diagnosis.

The objective is to identify the most effective denoising technique for each noise type while maintaining important pathological structures.

3. Objectives
Primary Objectives
Build a reproducible AI pipeline for breast cancer classification.
Simulate realistic image degradation using multiple noise models.
Apply multiple denoising algorithms.
Compare classification accuracy across all preprocessing pipelines.
Identify the best filter for each noise type.
Secondary Objectives
Maintain reproducible experiments.
Modularize preprocessing pipeline.
Support future deployment.
Enable easy experimentation through configuration files.
4. Dataset
Dataset

BCNB (Breast Core Needle Biopsy) Dataset

Contains:

Whole Slide Images (WSIs)
Clinical metadata
Dataset split information

Directory

data/
 ├── raw/
 │    └── BCNB/
 │         ├── WSIs/
 │         ├── patient-clinical-data.xlsx
 │         ├── dataset-splitting/
 │         └── README.txt
5. Scope
In Scope

✔ Image preprocessing

✔ Noise simulation

✔ Image denoising

✔ CNN model training

✔ Model evaluation

✔ Performance comparison

✔ Model checkpointing

✔ Experiment reproducibility

Out of Scope
Clinical diagnosis
Real-time pathology scanner integration
Hospital deployment
Patient management system
6. Functional Requirements
FR-1 Data Loading

The system shall

Load BCNB images
Read train/validation/test splits
Support batch loading
Support configurable image size
FR-2 Image Preprocessing

The preprocessing pipeline shall include

Resize images
Color normalization (optional)
Image normalization
Tensor conversion
FR-3 Noise Generation

The system shall support the following synthetic noise types:

Noise Type	Description
Gaussian	Additive Gaussian Noise
Salt & Pepper	Random impulse noise
Speckle	Multiplicative noise
Poisson	Photon counting noise
Mixed Poisson + Gaussian	Combination of Poisson and Gaussian noise

Implementation

noise_types = [
    'gaussian',
    's&p',
    'speckle',
    'poisson',
    'mixed_poisson_gaussian'
]

Each image should be able to generate all noisy variants.

FR-4 Image Denoising

The system shall support the following denoising methods:

denoising_methods = [
    'median',
    'gaussian',
    'wiener',
    'bilateral',
    'non_local_means',
    'anscombe_wiener',
    'adaptive_median',
    'kuan'
]
Supported Filters
1. Median Filter

Purpose

Removes salt-and-pepper noise while preserving edges.

2. Gaussian Filter

Purpose

Smooths Gaussian noise.

3. Wiener Filter

Purpose

Adaptive noise reduction using local statistics.

4. Bilateral Filter

Purpose

Noise removal while preserving edges.

5. Non-Local Means

Purpose

High-quality denoising by averaging similar patches.

6. Anscombe + Wiener

Purpose

Variance stabilization for Poisson noise followed by Wiener filtering.

7. Adaptive Median Filter

Purpose

Improved removal of impulse noise.

8. Kuan Filter

Purpose

Adaptive filter for multiplicative (speckle) noise.

7. Noise-Filter Evaluation Matrix
Noise	Filters Applied
Gaussian	All filters
Salt & Pepper	All filters
Speckle	All filters
Poisson	All filters
Mixed Poisson + Gaussian	All filters

Total Experiments

5 Noise Types
×

8 Filters

=

40 preprocessing experiments

Each experiment is followed by model training and evaluation.

8. Model Training

The system shall support

CNN architecture
Transfer Learning
GPU acceleration
Checkpoint saving
Resume training

Possible backbone models

ResNet50
EfficientNet
DenseNet121
ConvNeXt
9. Model Evaluation

Metrics

Accuracy
Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
Loss Curves
10. Non-functional Requirements
Performance
GPU compatible
Efficient memory usage
Modular pipeline
Reliability
Deterministic random seed
Automatic checkpoint recovery
Maintainability
Modular source code
YAML configuration
Unit testing
Scalability

Future support for

Multiple datasets
New filters
Additional CNN architectures
Multi-class classification
11. Proposed Workflow
Raw BCNB Images
        │
        ▼
Image Preprocessing
        │
        ▼
Noise Injection
        │
        ▼
Denoising
        │
        ▼
Dataset Generation
        │
        ▼
CNN Training
        │
        ▼
Validation
        │
        ▼
Testing
        │
        ▼
Performance Comparison
        │
        ▼
Best Noise-Filter Combination
12. Expected Outputs

The project should generate

Clean dataset
Noisy dataset
Denoised dataset
Trained AI model
Saved checkpoints
Evaluation metrics
Confusion matrix
ROC curves
Training history
Best-performing preprocessing pipeline
13. Project Structure
core-needle-biopsy-ai-model-project/
├── .env
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   └── BCNB/
│   └── processed/
│
├── models/
│   └── checkpoint.pt
│
├── notebooks/
│   └── 01_eda.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── engine.py
│   ├── model.py
│   └── utils.py
│
└── tests/
    └── test_model.py
14. Future Enhancements
Whole Slide Image (WSI) patch extraction
Attention-based Multiple Instance Learning (MIL)
Explainable AI (Grad-CAM, Score-CAM)
Hyperparameter optimization using Optuna
Distributed training with PyTorch Lightning
ONNX/TensorRT model export
Web-based inference dashboard (Flask/FastAPI)
Docker and Kubernetes deployment
Clinical decision support integration
15. Success Criteria

The project will be considered successful if it achieves the following:

Successfully loads and preprocesses the BCNB dataset.
Generates all five specified noise variants and applies all eight denoising methods, producing 40 preprocessing experiment combinations.
Trains deep learning models consistently across all experiment pipelines.
Evaluates performance using Accuracy, Precision, Recall, F1-score, ROC-AUC, and Confusion Matrix.
Identifies the optimal noise–filter combination for breast cancer histopathology image classification.
Produces reproducible results through configuration management, checkpointing, and deterministic training.
Maintains a modular, testable, and deployment-ready codebase aligned with the proposed project structure.