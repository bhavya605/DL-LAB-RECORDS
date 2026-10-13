Deep Learning Lab 
Author: Harshita Bhatnagar

This repository contains all experiments performed as part of the Deep Learning Lab.
The experiments cover the full pipeline of modern deep learning—from basic neural networks to advanced generative models such as VAEs and GANs.

Each experiment includes:
Aim/Objectives
Theory/Concepts
Implementation (TensorFlow / PyTorch)
Graphs & Visualizations
Results & Outputs
Conclusion

List of Experiments -

Exp-2: Building Neural Networks from Scratch
Implement perceptron AND gate
XOR/AND using Feedforward Neural Network

Exp-3: Activation Functions & Loss Functions
Sigmoid, ReLU, Softmax, Tanh
Cross-entropy, MSE visualization

Exp-4: Multi-Layer Perceptron (MLP)
Forward/backprop
Hidden layers, accuracy curves

Exp-5: Image Classification with Pre-trained CNNs (ResNet)
Transfer Learning
Training, testing, confusion matrix, ROC

Exp-6: Convolutional Neural Network (CNN) from Scratch
Filters, pooling, feature maps
Training & performance evaluation

Exp-7: Model Evaluation Metrics
Accuracy, Precision, Recall, F1-score
Sensitivity, Specificity, ROC

Exp-8: CNN Tutorial Implementation
Using Kaggle flower dataset
Feature extraction, visualization

Exp-9: Transfer Learning & Fine-tuning
Using ResNet/VGG models
Detailed training logs

Exp-10: Image Segmentation using U-Net
Build U-Net
Dice loss, IoU metrics
Segmentation masks visualization

Exp-11: Object Detection using R-CNN
Selective search region proposals
ROI classification + bounding-box regression
Prediction visualization

Exp-12: Autoencoder for Image Reconstruction
Dimensionality reduction
Reconstruction quality comparison

Exp-13: Variational Autoencoder (VAE)
Sampling in latent spac
Average class images
Interpolation between flowers

Exp-14: GAN for Synthetic Image Generation
Generator + Discriminator
Train GAN on flower images
Generate synthetic samples
Compare with VAE output

Environment Setup

Install dependencies:
pip install -r requirements.txt

Running the Experiments

Run any experiment using:
python Exp-<number>.py
Example:
python Exp-10.py

Outputs & Visualizations

Each experiment includes:
Accuracy/Loss curves
Confusion matrices
Sample predictions
Segmentation masks
Reconstructed images
Synthetic generated images (GAN & VAE)
All figures are displayed automatically when the script runs.

Technologies Used

TensorFlow / Keras
PyTorch / Torchvision
OpenCV / Selective Search
Albumentations (augmentations)
UMAP (dimensionality reduction)
Matplotlib & Seaborn
Scikit-learn metrics

Highlights of This Repository

✔ All models fully implemented
✔ Clean structure
✔ Fast and optimized code
✔ Plots included (NOT removed)
✔ Real dataset
✔ Covers entire deep learning pipeline
✔ Includes both discriminative and generative modeling

Acknowledgements

Kaggle Flowers Dataset
TensorFlow, PyTorch
University Deep Learning Lab guidelines
