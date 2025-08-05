# Image Classification with MobileNetV3 and TensorFlow

This repository contains solutions and practical experimentation for an **image classification pipeline** using deep learning with **MobileNetV3** in TensorFlow. The project focuses on training a lightweight and efficient CNN to recognize visual characters, applying best practices in preprocessing, model architecture, resource monitoring, and evaluation using a confusion matrix.

---

## Project Overview

This project demonstrates a complete image classification workflow focused on computational efficiency and accuracy. It handles data loading from Google Drive, transfer learning with MobileNetV3, and includes tools for performance monitoring and validation.

1.  **Data Loading & Preparation**  
    The dataset is extracted from a ZIP file stored in Google Drive. Images are organized into training, validation, and testing folders. They are then converted to RGB format and resized to fit the input shape required by the model.

2.  **Model Architecture (MobileNetV3 Small + CNN Layers)**  
    Convolutional Neural Networks (CNNs) are neural networks designed to process grid-like data, such as images. They use convolutional filters to automatically extract local features (edges, shapes, textures), making them ideal for visual classification tasks.

    In this project, we use **MobileNetV3 Small**, a lightweight and efficient CNN architecture designed for mobile devices. It was built using neural architecture search (NAS) techniques and includes optimized blocks such as **depthwise separable convolutions** and **SE (Squeeze-and-Excitation) modules**, reducing the number of parameters without sacrificing performance.

    The final architecture includes:
    
    - A **pretrained MobileNetV3Small** base (frozen convolutional layers)
    - `GlobalAveragePooling2D` to reduce dimensionality
    - A final `Dense` layer with `softmax` activation for multi-class classification

3.  **Training Process**  
    The model is trained for up to 30 epochs with continuous monitoring:

    - `EarlyStopping`: stops training if validation accuracy does not improve after 5 consecutive epochs
    - `ModelCheckpoint`: saves the best-performing model
    - `LambdaCallback`: for custom logging each epoch

    The model is compiled with `sparse_categorical_crossentropy` as the loss function and `accuracy` as the evaluation metric, using the `Adam` optimizer.

4.  **Performance Monitoring**  
    During training, CPU and memory usage are monitored using `psutil` to ensure the code runs efficiently in restricted environments such as Google Colab.

5.  **Evaluation with Confusion Matrix**  
    After training, the model is evaluated on the test dataset. Predictions are compared to the true labels using `sklearn`'s `confusion_matrix`, and the result is visualized with `seaborn.heatmap`. This analysis shows where the model performs well and where it tends to misclassify.

---

## Key Learning Outcomes & Features

1.  **Image Data Handling for Deep Learning**  
    Hands-on experience with preprocessing, directory structuring, and batch loading using `ImageDataGenerator`.

2.  **Transfer Learning with MobileNetV3**  
    Efficient application of a pretrained architecture to accelerate model development and improve accuracy with limited data.

3.  **Optimized Training**  
    Use of callbacks like EarlyStopping and automatic checkpointing to prevent overfitting and reduce training time.

4.  **Computational Efficiency**  
    Active monitoring of system resources to ensure the notebook runs smoothly in environments with limited capacity.

5.  **In-Depth Evaluation with Confusion Matrix**  
    Detailed understanding of the model's performance on each individual class, identifying common misclassification patterns.

---


## Libraries Used

-   `tensorflow` – for building and training deep learning models  
-   `pandas` – for data structure handling  
-   `numpy` – for numerical processing  
-   `matplotlib` – for visualization and plotting  
-   `seaborn` – for advanced visualizations (confusion matrix)  
-   `psutil` – for real-time system resource monitoring  
-   `sklearn` – for evaluation metrics such as confusion matrix  
-   `google.colab` – for file system integration and Google Drive access  

---
