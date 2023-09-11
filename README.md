# Masked Face Classification

This GitHub repository contains a machine learning project aimed at classifying input images, videos, or live streams into two categories: "Masked" and "Unmasked." The classification is performed using two popular convolutional neural network (CNN) architectures: ResNet50V2 and MobileNetV2.

This project is a sub-algorithm of the HumaneX Masked Face Recognition Pipeline, designed to contribute to the larger HumaneX Ecosystem.

## Introduction
The goal of this project is to automatically determine whether a face in an input image or video stream is wearing a mask or not. This is a critical task for applications like access control, security, and public health monitoring.

## Features
- Classification of faces into "Masked" and "Unmasked" categories.
- Utilizes two different CNN architectures (ResNet50V2 and MobileNetV2) for classification.
- Designed to be part of the HumaneX Masked Face Recognition Pipeline.
- Can be used with static images, video files, or live camera feeds.

  ## Getting Started
To get started with this project, follow these steps:

```bach
git clone https://github.com/Jawabreh0/Decision_Algoritm.git
cd Decision_Algoritm
```
Set up your Python environment. We recommend using a virtual environment:

```bach
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```
Install the required dependencies:

```bach
pip install -r requirements.txt
```

You are now ready to use the project. Refer to the Usage section for more details on how to classify faces.

## Model Architecture
This project utilizes two CNN architectures:

- ResNet50V2: Residual Network with 50 layers, version 2.
- MobileNetV2: Lightweight CNN architecture optimized for mobile and embedded devices.
Both models have been pretrained on a large dataset and fine-tuned for the specific task of masked face classification.

## Dataset
The dataset used to train and evaluate the models is not included in this repository due to size constraints.

## Contributing
We welcome contributions to this project.

## License
This project is licensed under the MIT License.

## Note: This project is part of the HumaneX Ecosystem and is intended for specific use cases related to masked face classification. Please respect ethical and legal considerations when using this technology.
