# Image Classification with MobileNetV3 and TensorFlow

This repository contains solutions and practical experimentation for an **image classification pipeline** using deep learning with **MobileNetV3** in TensorFlow. The project focuses on training a lightweight convolutional neural network on a character recognition dataset, implementing best practices in data preprocessing, model fine-tuning, resource monitoring, and evaluation through confusion matrices.

---

## Project Overview

This notebook-based project showcases an end-to-end workflow for character image classification, emphasizing model efficiency and accuracy in handling image data. The steps involve structured data handling from Google Drive, deep learning model building with transfer learning, training optimization strategies, and performance evaluation.

1.  **Data Loading & Preparation**
    The dataset is extracted from a ZIP archive stored on Google Drive. Image files are automatically organized into training, validation, and test directories. All images are loaded in RGB format and resized to meet the input requirements of the MobileNetV3 architecture.

2.  **Model Architecture (MobileNetV3 Small)**
    A pre-trained MobileNetV3 Small network is used as the base feature extractor. On top of this, custom dense layers are added, with a softmax output layer matching the number of target classes. The model is built using `Sequential()` from `tensorflow.keras.models`.

3.  **Training Process**
    Training is conducted with optimizations including:
    
    - `EarlyStopping`: to halt training when validation loss stops improving.
    - `ModelCheckpoint`: to save the best-performing model weights.
    - `LambdaCallback`: for custom console logging per epoch.

    The model is compiled with `categorical_crossentropy` and trained using `Adam` optimizer, evaluating accuracy on the validation dataset after each epoch.

4.  **Performance Monitoring**
    System resources (CPU, RAM) are monitored during execution using the `psutil` and `subprocess` libraries to ensure training is efficient and does not exceed platform constraints (Google Colab runtime).

5.  **Evaluation with Confusion Matrix**
    Predictions on the test dataset are compared to true labels to generate a confusion matrix. This matrix is visualized using `seaborn.heatmap`, giving detailed insights into model accuracy and common misclassifications across classes.

---

## Key Learning Outcomes & Features

1.  **Image Data Handling for Deep Learning**
    Techniques for structuring and preprocessing image datasets, including ZIP file extraction, RGB conversion, and batch loading via `ImageDataGenerator`.

2.  **Transfer Learning with MobileNetV3**
    Integration of `MobileNetV3Small` as a base model for efficient and accurate image classification, leveraging pre-trained weights for improved generalization.

3.  **Training Optimization**
    Use of callbacks (`EarlyStopping`, `ModelCheckpoint`) to automate and improve the training process, preventing overfitting and reducing computation time.

4.  **Resource Efficiency & Monitoring**
    Practical implementation of runtime monitoring tools to manage memory and CPU usage during model training, particularly in cloud environments like Colab.

5.  **Post-Training Evaluation**
    Application of confusion matrix analysis to evaluate classification performance and identify weaknesses in the model’s predictive ability.

---

## Libraries Used

-   `tensorflow` – for building and training deep learning models
-   `pandas` – for data structure handling
-   `numpy` – for numerical processing
-   `matplotlib` – for plotting evaluation metrics
-   `seaborn` – for advanced visualization (confusion matrix)
-   `psutil` – for real-time system resource monitoring
-   `sklearn` – for metrics such as confusion matrix
-   `google.colab` – for managing cloud file systems and mounting Google Drive

---
