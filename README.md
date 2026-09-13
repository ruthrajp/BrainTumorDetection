# 🧠 Brain Tumor Detection Using Transfer Learning

A deep learning-based web application that classifies brain MRI images into four categories using **ShuffleNet V2** and **Transfer Learning**.

## 📌 Project Overview

Brain tumors can be difficult to identify from MRI images through manual analysis alone. This project uses a deep learning approach to automatically classify brain MRI images into different categories.

The system uses a **pre-trained ShuffleNet V2 model** and applies **Transfer Learning** to classify MRI images into:

- Glioma Tumor
- Meningioma Tumor
- Pituitary Tumor
- No Tumor

The trained model is integrated with a **Flask web application**, allowing users to upload an MRI image and receive a prediction through a web interface.

## 🚀 Features

- Brain MRI image classification
- Four-class tumor classification
- Transfer Learning using ShuffleNet V2
- Image preprocessing
- Web-based MRI image upload
- Real-time prediction through Flask
- Simple and user-friendly interface

## 🛠️ Technologies Used

### Programming Language
- Python

### Deep Learning
- PyTorch
- Torchvision
- ShuffleNet V2
- Transfer Learning

### Image Processing
- Pillow (PIL)

### Web Development
- Flask
- HTML
- CSS
- Jinja2

### Development Environment
- Google Colab
- Jupyter Notebook

## 🧠 Model Architecture

The project uses **ShuffleNet V2**, a lightweight and computationally efficient convolutional neural network.

A pre-trained ShuffleNet V2 model is used as the base model. The final fully connected layer is modified to support four output classes.

```python
model.fc = nn.Linear(model.fc.in_features, 4)
