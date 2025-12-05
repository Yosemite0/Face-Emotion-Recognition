# Face & Emotion Recognition System

**Course:** Statistical Methods in Artificial Intelligence (SMAI) - Assignment 2

**Author:** Yash Chordia (2024201029)

## Overview

This repository contains the implementation of a dual-task computer vision system:

1. **Face Recognition (Binary Classification):** A system to distinguish between "My Face" and "Other Faces" for biometric authentication.

2. **Emotion Recognition (Multiclass Classification):** A system to classify facial expressions into categories such as Angry, Happy, Neutral, and Surprise.

Both tasks explore Transfer Learning (VGG16, ResNet18) versus training from scratch, with extensive logging via Weights & Biases (WandB).

## Directory Structure

```text
.
├── Report/
│   ├── SMAI_Assign2_Q1.pdf          # Detailed Report for Part 1 (Face Rec)
│   └── SMAI_Assign2_Q2.pdf          # Detailed Report for Part 2 (Emotion Rec)
├── video/                           # Simulation videos for Face Recognition
│   ├── simulation_ResNet18_NotPretrained.mp4
│   ├── simulation_ResNet18_Pretrained.mp4
│   └── simulation_VGG16.mp4
├── video2/                          # Prediction visualization for Emotion Recognition
│   ├── resnet18_pretrained_predictions_video.mp4
│   ├── resnet18_scratch_predictions_video.mp4
│   └── vgg16_predictions_video.mp4
├── .gitignore
├── emotion_recognition.ipynb        # Jupyter Notebook for Task 2
├── face_recognition.ipynb           # Jupyter Notebook for Task 1
└── README.md
````

## Part 1: Face Recognition

**Goal:** Create a binary classifier (Label 1: My Face, Label 0: Others) robust to lighting, occlusion, and background changes.

### Models Implemented

1.  **VGG16 (Finetuned):** Pretrained on ImageNet, classifier head replaced. (Best Performance: \~99.79%)

2.  **ResNet18 (Pretrained):** Pretrained on ImageNet, fine-tuned entire network.

3.  **ResNet18 (Scratch):** Random initialization, trained entirely on custom data.

### Key Features

  * **Data Augmentation:** Random flips, rotations ($\pm15^{\circ}$), color jitter, and synthetic occlusions.

  * **Unlock Simulation:** A video simulation (located in `video/`) demonstrating a phone unlock mechanism based on model predictions.

## Part 2: Emotion Recognition

**Goal:** Extend the face recognition system to a $k$-way classifier ($k=4$) for emotions: **Angry, Happy, Neutral, Surprise**.

### Models Implemented

The same three architectures were adapted for multiclass classification:

1.  **VGG16 (Finetuned)**

2.  **ResNet18 (Scratch)**

3.  **ResNet18 (Pretrained)** - Achieved the highest accuracy and generalization.

### Key Features

  * **Class-wise Analysis:** Detailed evaluation of model performance on specific emotions.

  * **Visualizations:** Prediction grids and videos (located in `video2/`) showing real-time emotion inference.

## Setup & Usage

### Prerequisites

  * Python 3.x

  * PyTorch, Torchvision

  * Weights & Biases (wandb)

  * OpenCV, Matplotlib, Scikit-learn

### Running the Notebooks

The project is divided into two independent Jupyter notebooks. You can run them locally or on Kaggle/Colab.

1.  **Face Recognition:**
    Open `face_recognition.ipynb`. Ensure you have the dataset linked or downloaded.

      * *Kaggle Link:* [View Notebook](https://www.kaggle.com/code/yashchordia0/notebook8d61e57471)

2.  **Emotion Recognition:**
    Open `emotion_recognition.ipynb`.

      * *Kaggle Link:* [View Notebook](https://www.kaggle.com/code/yashchordia0/notebookea736b37d3)

## Resources & Links

### Datasets

  * **Face Data:** [GitHub - face\_data](https://github.com/Yosemite0/face_data.git)

  * **Emotion Data:** [GitHub - emotion\_data](https://github.com/Yosemite0/emotion_data)

### Experiment Tracking (WandB)

  * **Face Recognition Runs:** [WandB Project](https://www.google.com/search?q=https://wandb.ai/yash-chordia-iiit-hyderabad/face-recognition%3Fnw%3Dnwuseryashchordia)

  * **Emotion Recognition Runs:** [WandB Project](https://www.google.com/search?q=https://wandb.ai/yash-chordia-iiit-hyderabad/emotion-recognition/runs/1tkvdrlw%3Fnw%3Dnwuseryashchordia)

### Full Reports

  * [Face Recognition Report (PDF)](https://www.google.com/search?q=Report/SMAI_Assign2_Q1.pdf)

  * [Emotion Recognition Report (PDF)](https://www.google.com/search?q=Report/SMAI_Assign2_Q2.pdf)

