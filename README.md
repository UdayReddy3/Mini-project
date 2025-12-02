# 🌿 Plant Disease Detection System

A complete, end-to-end deep learning project for detecting plant diseases using the **PlantVillage dataset** and **ResNet50** transfer learning. The system includes a web interface for dynamic image upload and real-time disease prediction.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Training the Model](#training-the-model)
- [Running the Web Application](#running-the-web-application)
- [Project Components](#project-components)
- [Usage Examples](#usage-examples)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)

## ✨ Features

- **Transfer Learning**: ResNet50 pre-trained on ImageNet, fine-tuned on PlantVillage dataset
- **Data Augmentation**: Rotation, zoom, flip, and shift augmentation for robust training
- **Real-time Inference**: Fast prediction on single images with confidence scores
- **Web Interface**: Streamlit-based interactive UI for easy image uploads and predictions
- **Batch Processing**: Process multiple images at once with summary statistics
- **Comprehensive Metrics**: Accuracy, loss, classification reports, and confusion matrices
- **Optimized Training**: Early stopping, model checkpointing, and learning rate reduction callbacks
- **38 Disease Classes**: Recognizes ~38 plant disease categories from PlantVillage dataset

## 📁 Project Structure

```
plant_disease_detection/
├── data/
│   └── plantvillage/          # Place PlantVillage dataset here
│       ├── disease_class_1/
│       ├── disease_class_2/
│       └── ...
├── models/
│   ├── plant_disease_model.h5  # Trained model (generated after training)
│   ├── best_model.h5           # Best model checkpoint
│   ├── class_names.json        # Class names mapping
│   ├── metrics.json            # Training metrics
│   └── training_history.png    # Training curves plot
├── uploads/                    # Temporary uploads directory
├── model.py                    # Model definition, training, and evaluation
├── predict.py                  # Single image inference function
├── app.py                      # Streamlit web application
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation (this file)
```

## 🚀 Installation

### 1. Clone or Create the Project

```bash
cd c:\Users\J.KARTHIK REDDY\Desktop\kk
cd plant_disease_detection
```

### 2. Create a Virtual Environment (Recommended)

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: If you encounter issues with TensorFlow installation on Windows, consider using:
```bash
pip install tensorflow-cpu  # For CPU only
# or
pip install tensorflow-gpu  # For GPU support
```

## 📊 Dataset Setup

### Downloading the PlantVillage Dataset

The PlantVillage dataset is publicly available and contains images of diseased and healthy plant leaves.

#### Option 1: Download from Kaggle (Recommended)

1. Visit: https://www.kaggle.com/abdallahalidev/plantvillage-dataset
2. Download the dataset (compressed file ~1GB)
3. Extract it to the `data/plantvillage/` directory

```bash
# After downloading and extracting
# Ensure the directory structure is:
# data/plantvillage/Apple___Apple_scab/
# data/plantvillage/Apple___Black_rot/
# ... etc
```

#### Option 2: Download from GitHub

```bash
# Clone the PlantVillage dataset repository
git clone https://github.com/spMohanty/PlantVillage-Dataset.git
# Copy raw/color directory to data/plantvillage
cp -r PlantVillage-Dataset/raw/color/* data/plantvillage/
```

#### Option 3: Direct Download

Download from: https://github.com/spMohanty/PlantVillage-Dataset/releases/download/v1.2/plantvillage_dataset_color.tar.xz

### Dataset Structure

The dataset should have the following structure:

```
data/plantvillage/
├── Apple___Apple_scab/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Apple___Black_rot/
│   ├── image1.jpg
│   └── ...
├── Corn___Cercospora_leaf_spot Gray_leaf_spot/
│   └── ...
└── ... (38 total disease classes)
```

Each subdirectory name represents a disease class and contains corresponding images.

## 🏋️ Training the Model

### Basic Training

```bash
python model.py
```

This will:
1. Load the PlantVillage dataset from `data/plantvillage/`
2. Build a ResNet50 transfer learning model
3. Train for up to 50 epochs with early stopping
4. Save the trained model to `models/plant_disease_model.h5`
5. Generate training history plots and metrics

### Training with Custom Parameters

Edit `model.py` and modify the `main()` function:

```python
# In model.py, modify the main() function:

model_handler = PlantDiseaseModel(img_size=224, num_classes=38)
# ... load dataset ...
model_handler.build_model(freeze_base=True, num_frozen_layers=150)
model_handler.compile_model(learning_rate=1e-4)

# Train with custom epochs
history = model_handler.train(
    train_dataset,
    val_dataset,
    epochs=100,  # Increase epochs
    model_save_path='models/best_model.h5'
)
```

### Expected Training Time

- **GPU**: ~20-40 minutes (depends on GPU power)
- **CPU**: ~2-4 hours
- **First Run**: Add ~5 minutes for model download (ResNet50)

### Training Output

After training, you'll see:
- Training/validation accuracy and loss curves in `models/training_history.png`
- Classification metrics in `models/metrics.json`
- Trained model saved as `models/plant_disease_model.h5`

## 🌐 Running the Web Application

### Start the Streamlit App

```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

### Web Interface Features

#### Tab 1: Single Image Prediction
- Upload a single plant leaf image
- Get real-time disease prediction
- View confidence scores for all disease classes
- Interactive bar chart showing top 10 predictions
- Detailed predictions table

#### Tab 2: Batch Analysis
- Upload multiple images at once
- Process all images in one batch
- View summary statistics
- Pie chart showing disease distribution
- Confidence statistics

#### Tab 3: Information
- View supported disease classes
- Model and dataset information
- Usage tips and best practices

## 📝 Project Components

### 1. `model.py` - Model Development and Training

**Main Class**: `PlantDiseaseModel`

Key methods:
- `__init__()`: Initialize model parameters
- `load_plantvillage_dataset()`: Load dataset with augmentation
- `load_tf_dataset()`: Alternative dataset loading with tf.data API
- `build_model()`: Build ResNet50 transfer learning model
- `compile_model()`: Compile with optimizer and loss function
- `train()`: Train with callbacks (early stopping, checkpointing, LR reduction)
- `evaluate()`: Evaluate model and generate classification report
- `plot_training_history()`: Visualize training curves
- `save_model()`: Save trained model
- `load_model()`: Load pre-trained model

**Features**:
- Transfer learning with ResNet50
- Layer freezing/unfreezing for fine-tuning
- Data augmentation (rotation, zoom, flip, shift)
- Early stopping and model checkpointing
- Learning rate reduction on plateau
- Comprehensive evaluation metrics

### 2. `predict.py` - Inference Module

**Main Class**: `DiseasePredictor`

Key methods:
- `load_model_and_classes()`: Load trained model and class names
- `preprocess_image()`: Load and preprocess single image
- `predict_disease()`: Predict disease from image path
- `batch_predict()`: Predict for all images in directory
- `predict_from_array()`: Predict from numpy array (useful for web apps)

**Standalone Function**:
```python
from predict import predict_disease

result = predict_disease('path/to/image.jpg')
print(result['disease_class'])
print(result['confidence'])
```

**Return Structure**:
```python
{
    'disease_class': 'Apple___Apple_scab',
    'confidence': 95.42,
    'all_predictions': {
        'Apple___Apple_scab': 95.42,
        'Apple___Black_rot': 2.13,
        ...
    },
    'timestamp': '2024-11-30T14:30:45.123456',
    'image_path': 'path/to/image.jpg',
    'top_3_predictions': [
        ('Apple___Apple_scab', 95.42),
        ('Apple___Black_rot', 2.13),
        ('Apple___Cedar_apple_rust', 1.45)
    ]
}
```

### 3. `app.py` - Streamlit Web Application

**Features**:
- Responsive web interface with custom CSS
- File upload (JPG, PNG, BMP, TIFF)
- Real-time predictions with confidence scores
- Interactive charts using Plotly
- Batch image processing
- Disease class information
- Model and dataset details

**Tabs**:
1. **Single Image Prediction**: Upload and predict for one image
2. **Batch Analysis**: Process multiple images and get statistics
3. **Information**: View system details and usage guide

## 💡 Usage Examples

### Example 1: Train the Model

```bash
python model.py
```

### Example 2: Predict on a Single Image

```python
from predict import predict_disease

result = predict_disease('test_leaf.jpg')
print(f"Disease: {result['disease_class']}")
print(f"Confidence: {result['confidence']}%")

# View top 3 predictions
for i, (disease, conf) in enumerate(result['top_3_predictions'], 1):
    print(f"{i}. {disease}: {conf:.2f}%")
```

### Example 3: Batch Prediction

```python
from predict import DiseasePredictor

predictor = DiseasePredictor()
results = predictor.batch_predict('path/to/images/')

for result in results:
    print(f"{result['image_path']}: {result['disease_class']} ({result['confidence']}%)")
```

### Example 4: Integrate with Custom Application

```python
from predict import DiseasePredictor
import cv2
import numpy as np

# Initialize predictor
predictor = DiseasePredictor()

# Load image with OpenCV
image = cv2.imread('image.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize to model input size
image_resized = cv2.resize(image_rgb, (224, 224))

# Predict
result = predictor.predict_from_array(image_resized)
print(f"Predicted: {result['disease_class']}")
```

## 🧠 Model Architecture

### Transfer Learning Approach

```
ResNet50 (Pre-trained on ImageNet)
    ↓
Frozen Base Layers (150 layers)
    ↓
Global Average Pooling
    ↓
Dense(256, ReLU) + BatchNorm + Dropout(0.5)
    ↓
Dense(128, ReLU) + BatchNorm + Dropout(0.3)
    ↓
Dense(38, Softmax) ← Number of disease classes
```

### Key Architecture Decisions

1. **ResNet50 Base**: Powerful feature extractor, trained on ImageNet
2. **Frozen Base Layers**: Retain learned features from ImageNet
3. **Global Average Pooling**: Reduces spatial dimensions efficiently
4. **L2 Regularization**: Prevent overfitting
5. **Batch Normalization**: Improve training stability
6. **Dropout**: Additional regularization
7. **Categorical Crossentropy**: Multi-class classification loss

### Training Parameters

- **Optimizer**: Adam (learning rate: 1e-4)
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)
- **Validation Split**: 20%
- **Data Augmentation**: Rotation (40°), Zoom (0.2), Shift (0.2), Flip

## 📊 Performance Metrics

### Expected Performance (PlantVillage Dataset)

After training for ~30-50 epochs:

- **Validation Accuracy**: ~95-98%
- **Validation Loss**: ~0.1-0.2
- **Precision (avg)**: ~0.96
- **Recall (avg)**: ~0.96
- **F1-Score (avg)**: ~0.96

**Note**: Actual performance depends on:
- Quality of training data
- Number of epochs trained
- GPU/CPU capabilities
- Data augmentation effectiveness

### Viewing Metrics

Metrics are saved in `models/metrics.json`:

```json
{
  "accuracy": 0.964,
  "loss": 0.125,
  "classification_report": {
    "Apple___Apple_scab": {
      "precision": 0.95,
      "recall": 0.94,
      "f1-score": 0.945
    },
    ...
  }
}
```

## 🐛 Troubleshooting

### Issue 1: Model files not found

**Error**: `FileNotFoundError: Model file not found: models/plant_disease_model.h5`

**Solution**:
1. Ensure you've run `python model.py` to train the model
2. Check that files exist in the `models/` directory
3. Verify the file paths in `predict.py` match your setup

### Issue 2: Dataset not found

**Error**: `No image files found in data/plantvillage/`

**Solution**:
1. Download PlantVillage dataset from Kaggle or GitHub
2. Extract to `data/plantvillage/` directory
3. Verify subdirectory structure (disease class folders)
4. Check that images are in supported formats (JPG, PNG)

### Issue 3: Out of Memory (OOM) during training

**Solutions**:
1. Reduce batch size: Edit `model.py`, change `batch_size=32` to `batch_size=16`
2. Use CPU instead of GPU: `pip install tensorflow-cpu`
3. Train on GPU with more VRAM
4. Reduce image size: Change `img_size=224` to `img_size=160`

### Issue 4: Streamlit app not loading predictions

**Error**: `AttributeError: 'NoneType' object has no attribute '...'`

**Solution**:
1. Ensure model files are in `models/` directory
2. Run `streamlit run app.py` from the project root directory
3. Check that `models/plant_disease_model.h5` exists
4. Verify `models/class_names.json` exists

### Issue 5: Slow predictions

**Solutions**:
1. Use GPU for inference: Install CUDA and `tensorflow-gpu`
2. Reduce image size for inference
3. Use model quantization or distillation for faster inference
4. Pre-load model outside the prediction loop

### Issue 6: Poor prediction accuracy

**Solutions**:
1. Train for more epochs
2. Increase data augmentation parameters
3. Use a larger learning rate or learning rate schedule
4. Verify dataset quality and labeling
5. Fine-tune more ResNet50 layers (reduce `num_frozen_layers`)

## 📚 Additional Resources

### PlantVillage Dataset
- Kaggle: https://www.kaggle.com/abdallahalidev/plantvillage-dataset
- GitHub: https://github.com/spMohanty/PlantVillage-Dataset
- Paper: https://arxiv.org/abs/1604.03169

### ResNet50 Transfer Learning
- Original Paper: https://arxiv.org/abs/1512.03385
- TensorFlow Docs: https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50

### Streamlit Documentation
- Official Docs: https://docs.streamlit.io/
- Components Gallery: https://streamlit.io/components

## 🔧 Advanced Customization

### Fine-tune More Layers

```python
# In model.py, modify build_model():
model_handler.build_model(freeze_base=True, num_frozen_layers=100)  # Freeze fewer layers
```

### Use Different Base Model

```python
# Replace ResNet50 with other architectures:
from tensorflow.keras.applications import EfficientNetB0, MobileNetV2

base_model = EfficientNetB0(weights='imagenet', include_top=False)
# or
base_model = MobileNetV2(weights='imagenet', include_top=False)
```

### Deploy to Cloud

```bash
# Deploy to Streamlit Cloud
# 1. Push to GitHub
# 2. Visit: https://share.streamlit.io
# 3. Connect your repository
# 4. Select app.py as main file
```

## 📄 License and Attribution

- PlantVillage Dataset: Hughes et al. (2016)
- ResNet50: He et al. (2015)
- TensorFlow/Keras: Google Research

## 📧 Support and Contribution

For issues or suggestions:
1. Check the Troubleshooting section
2. Review the code comments
3. Consult official documentation links

---

**Happy Plant Disease Detection! 🌱✨**
