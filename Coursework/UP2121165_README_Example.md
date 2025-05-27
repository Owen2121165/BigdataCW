# Fashion MNIST Image Classification with CNNs

## Introduction
This project implements a machine learning pipeline for classifying fashion images using Convolutional Neural Networks. It uses the Fashion MNIST dataset, which contains grayscale images of clothing items across 10 categories. The aim is to compare different Convolutional Neural Network architectures and determine which performs best in terms of accuracy and generalisation.

## Business Objectives
The goal is to build the CNN that can classify unseen fashion items into 10 predefined categories. A high validation and test accuracy is targeted to achieve above 90% to demonstrate the effectiveness of the model.

## ML Pipeline

### 1. Data Collection
- **Source**: `tensorflow.keras.datasets.fashion_mnist`
- The data was loaded and reshaped to add a channel dimension.
- Pixel values were scaled from [0-255] to [0, 1] for better training of the model.
- Labels were converted to one-hot vectors for multi-class classification. 

### 2. EDA
- Seaborn was used to visualise Class distribution.
- Sample images plotted to understand the image content and diversity.
- No major class imbalance was discovered.

### 3. Model Building

#### Model 1: Baseline
- Conv2D (32, 3x3) using 32 filters to extract features
- MaxPooling (2x2): Downsmaple feature maps
- Flatten: converts 2D output to 1D
- Dense layers (64: connect layers with 64 neurons) (10: output layers for 10 classes)
- Achieved ~90.87% validation accuracy

#### Model 2: Improved Hyperparameters
- Increased filters and dense layer capacity
- reduced learning rate for finer tuning.
- Achieved ~91.53% validation accuracy

#### Model 3: Dropout + BatchNorm
- BatchNormalization improves stability for training
- Dropout reduces overfitting
- Achieved ~90.87% validation accuracy

### 4. Model Evaluation
- Validation accuracy and loss were shown.
- Confusion matrix was generated for the best model.
- Model 2 performed the best during final testing.

### 5. Prediction
- all 3 confusion matrix was displayed.
- Accuracy was computed and confusion matrix was analysed.
- Predictions were visualised with true labels and softmax scores.

## Jupyter Notebook Structure
The notebook includes:
1. Setup and data loading
2. EDA
3. Three CNN model definitions and training
4. Model evaluation and testing
5. Summary and final remarks

## Future Work
- Implement data augmentation to further increase training diversity
- Try transfer learning using pretrained networks
- Deploy model using web interface

## Libraries and Modules
- **TensorFlow**: Building and training CNN models
- **NumPy & Pandas**: Data manipulation
- **matplotlib & seaborn**: Visualization
- **scikit-learn**: Evaluation metrics and confusion matrix

## Unfixed Bugs
- No major bugs encountered.

## Acknowledgements and References
- ChatGPT was used to scaffold code and organize the notebook layout.
- Model structure adapted from TensorFlow documentation: https://www.tensorflow.org/tutorials/images/cnn

## Conclusions
CNNs are effective for image classification tasks. By iteratively improving model architecture, validation accuracy was increased. Model 2 achieved the best performance and generalization.
