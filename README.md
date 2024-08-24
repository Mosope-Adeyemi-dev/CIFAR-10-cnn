# CIFAR-10 Convolutional Neural Network

This project involves training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset, a widely-used benchmark for image classification tasks.

## CIFAR-10 Dataset

The CIFAR-10 dataset is a collection of 60,000 32x32 color images across 10 different classes, with 6,000 images per class. The dataset is divided into two parts:

- **Training Set:** 50,000 images
- **Test Set:** 10,000 images

### Classes

The CIFAR-10 dataset contains the following 10 classes:

- **0:** Airplane
- **1:** Automobile
- **2:** Bird
- **3:** Cat
- **4:** Deer
- **5:** Dog
- **6:** Frog
- **7:** Horse
- **8:** Ship
- **9:** Truck

Each class represents a real-world object, making CIFAR-10 a versatile dataset for evaluating image classification models. The images are relatively small, with a resolution of 32x32 pixels, and contain a variety of complex features, making it a challenging dataset for neural networks.

### Challenges

Some of the challenges associated with the CIFAR-10 dataset include:

- **Low Resolution:** The 32x32 resolution can make it difficult for models to capture fine details.
- **Class Overlap:** Some classes, like automobiles and trucks, or cats and dogs, have visually similar features, which can make classification more challenging.
- **Noise:** The dataset includes a variety of backgrounds and image conditions, adding to the complexity of the task.

### Use Cases

CIFAR-10 is often used to:

- Benchmark the performance of new machine learning and deep learning algorithms.
- Explore the effectiveness of data augmentation, regularization techniques, and other improvements in image classification models.
- Provide a foundation for more complex image recognition tasks, such as object detection and segmentation.

## Model Architecture

The CNN model employs the following architecture:

- **Convolutional Layers:** 4 layers with 3x3 filters
- **Max Pooling Layers:** 2 layers
- **Fully Connected Layers:** 2 hidden layers

## Model Performance

After training for 20 epochs, the model achieved the following results:

- **Training Set:**
  - Loss: **0.1208**
  - Accuracy: **95%**
  
- **Test Set:**
  - Loss: **0.96**
  - Accuracy: **77%**

### Observations

- The model performs well on the training set but shows a noticeable drop in performance on the test set, indicating potential overfitting.
- Increasing the number of training epochs may further improve model performance.
- Additional techniques such as **data augmentation**, **regularization** (e.g., Dropout, L2 regularization), and **hyperparameter tuning** could enhance the model's generalization ability.

## Model Saving

The trained model is saved in the `saved_models` directory.
