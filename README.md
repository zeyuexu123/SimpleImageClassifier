# Simple Image Classifier

## Overview
This repository contains a Simple Image Classifier built to help me better understand the fundamentals of deep learning and image classification. The model is configured for training on CIFAR-10. 

## Goal
The goal of this project is to explore the fundamentals of CNN:
- How convolutional layers extract features from images
- How models are trained using backpropagation
- How accuracy is measured across classes
- Practical skills in PyTorch

## Project Structure
```
.
├── config.py       # Hyperparameters and global settings
├── data.py         # Dataset loading and preprocessing
├── model.py        # CNN architecture
├── train.py        # Training loop
├── evaluate.py     # Model evaluation
├── main.py         # Entry point
└── cifar_net.pth   # Saved model weights (generated after training)
```

## Model Architecture
The network consists of:
  3 convolutional blocks
    Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU
    Followed by MaxPooling
  Adaptive Average Pooling
  Fully connected layer (256 → 10)

## Dataset
 - CIFAR-10
 - 60,000 color images (32×32)
 - 10 classes:
```
airplane, automobile, bird, cat, deer,
dog, frog, horse, ship, truck
```

## Training the Model
To start training&evaluate after trained:
```
python -c "import torch, torchvision" || python -m pip install torch torchvision
python main.py
```

## Acknowledgements
Training a Classifier - https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
