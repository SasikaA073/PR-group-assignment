# Waste Classification Using Convolutional Neural Networks (CNN)

This project focuses on developing a **simple convolutional neural network (CNN)** to classify waste using the **RealWaste dataset**. The dataset includes nine classes of waste to train the model, enabling applications like automated waste sorting and environmental monitoring.

## Dataset
We utilized the **RealWaste dataset**, which includes the following classes:
- Cardboard
- Food Organics
- Glass
- Metal
- Miscellaneous Trash
- Paper
- Plastic
- Textile Trash
- Vegetation

The dataset was split into:
- Training set: 60%
- Validation set: 20%
- Testing set: 20%

Dataset link: [RealWaste Dataset](https://archive.ics.uci.edu/dataset/908/realwaste)

## Project Features

### 1. Custom CNN Model
We implemented a custom CNN architecture with the following specifications:
- **Convolutional Layers**: Four layers with ReLU activation and varying filter sizes (64, 128, 256, and 512).
- **Pooling Layers**: Max pooling for down-sampling.
- **Fully Connected Layers**: 
  - First layer: 512 neurons with ReLU and 50% dropout.
  - Output layer: 9 neurons with softmax activation.
- **Optimizer**: Adam (chosen over SGD for its faster convergence).
- **Loss Function**: Sparse categorical cross-entropy.

### 2. Fine-tuning Pretrained Model
We fine-tuned the **ResNet50** model using transfer learning. This pretrained model significantly outperformed the custom model in terms of accuracy, precision, and recall.

### 3. Hyperparameter Tuning
We experimented with different learning rates (0.1, 0.01, 0.001, and 0.0001). The best performance was achieved with a learning rate of 0.0001.

## Evaluation

### Custom Model Performance
- **Test Accuracy**: 0.1987
- **Precision**: 0.0395
- **Recall**: 0.1987

### Fine-tuned ResNet50 Performance
- **Test Accuracy**: 0.8875
- **Precision**: 0.8913
- **Recall**: 0.8875

## Advantages and Limitations

### Custom Model
- **Advantages**:
  - Tailored to the dataset.
  - Lightweight and flexible.
- **Limitations**:
  - Requires more data for effective training.
  - Longer training time.
  - Limited generalization.

### Pretrained Model (ResNet50)
- **Advantages**:
  - Leverages knowledge from large-scale datasets.
  - Performs well with limited data.
  - Faster convergence.
- **Limitations**:
  - Higher computational resource requirements.
  - Potential overfitting if not fine-tuned carefully.

## Results
Fine-tuning the ResNet50 model resulted in significantly better performance than the custom CNN model, highlighting the advantages of transfer learning in image classification tasks.


