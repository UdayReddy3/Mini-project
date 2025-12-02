"""
Plant Disease Detection Model
This module contains the CNN model architecture using transfer learning (ResNet50)
for plant disease classification on the PlantVillage dataset.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import json


class PlantDiseaseModel:
    """
    A class to handle model creation, training, and evaluation for plant disease detection.
    Uses transfer learning with ResNet50 pre-trained on ImageNet.
    """
    
    def __init__(self, img_size=224, num_classes=38):
        """
        Initialize the model parameters.
        
        Args:
            img_size (int): Target image size for ResNet50 (default: 224)
            num_classes (int): Number of disease classes (PlantVillage has ~38 classes)
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.class_names = None
        self.label_encoder = LabelEncoder()
        
    def load_plantvillage_dataset(self, data_path, validation_split=0.2, test_split=0.1):
        """
        Load the PlantVillage dataset from local directory structure.
        Expected structure: data_path/disease_class/images/
        
        Args:
            data_path (str): Path to the PlantVillage dataset
            validation_split (float): Fraction of data for validation
            test_split (float): Fraction of data for testing
            
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
        print(f"Loading PlantVillage dataset from {data_path}...")
        
        # Create image data generators with augmentation
        train_datagen = ImageDataGenerator(
            rescale=1.0/255.0,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=validation_split
        )
        
        val_datagen = ImageDataGenerator(rescale=1.0/255.0)
        
        # Load training data
        train_dataset = train_datagen.flow_from_directory(
            data_path,
            target_size=(self.img_size, self.img_size),
            batch_size=32,
            class_mode='categorical',
            subset='training'
        )
        
        # Load validation data
        val_dataset = train_datagen.flow_from_directory(
            data_path,
            target_size=(self.img_size, self.img_size),
            batch_size=32,
            class_mode='categorical',
            subset='validation'
        )
        
        # Store class names
        self.class_names = list(train_dataset.class_indices.keys())
        self.num_classes = len(self.class_names)
        
        print(f"Found {self.num_classes} disease classes")
        print(f"Classes: {self.class_names}")
        
        return train_dataset, val_dataset
    
    def load_tf_dataset(self, data_path, validation_split=0.2):
        """
        Alternative method to load dataset using tf.data for better performance.
        
        Args:
            data_path (str): Path to the PlantVillage dataset
            validation_split (float): Fraction of data for validation
            
        Returns:
            tuple: (train_dataset, val_dataset)
        """
        print(f"Loading dataset using tf.data API from {data_path}...")
        
        # Create dataset from directory
        dataset = tf.keras.utils.image_dataset_from_directory(
            data_path,
            seed=42,
            image_size=(self.img_size, self.img_size),
            batch_size=32,
            validation_split=validation_split,
            subset=None,
            label_mode='categorical'
        )
        
        # Split into train and validation
        train_size = int(len(dataset) * (1 - validation_split))
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size)
        
        # Extract class names
        self.class_names = dataset.class_names
        self.num_classes = len(self.class_names)
        
        print(f"Found {self.num_classes} disease classes")
        print(f"Classes: {self.class_names}")
        
        # Normalize pixel values
        normalization_layer = layers.Rescaling(1.0/255)
        train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
        val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
        
        # Prefetch for performance
        train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
        val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset
    
    def build_model(self, freeze_base=True, num_frozen_layers=150):
        """
        Build the model using transfer learning with ResNet50.
        
        Args:
            freeze_base (bool): Whether to freeze base model layers
            num_frozen_layers (int): Number of layers to freeze from the base model
            
        Returns:
            model: Compiled Keras model
        """
        print("Building transfer learning model with ResNet50...")
        
        # Load pre-trained ResNet50 model
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Freeze base model layers for transfer learning
        if freeze_base:
            for layer in base_model.layers[:num_frozen_layers]:
                layer.trainable = False
            print(f"Froze first {num_frozen_layers} layers of ResNet50")
        
        # Build custom top layers
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        self.model = model
        return model
    
    def compile_model(self, learning_rate=1e-4):
        """
        Compile the model with optimizer, loss, and metrics.
        
        Args:
            learning_rate (float): Learning rate for Adam optimizer
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model compiled successfully")
        print(self.model.summary())
    
    def train(self, train_dataset, val_dataset, epochs=50, model_save_path='models/best_model.h5'):
        """
        Train the model with callbacks for early stopping and checkpointing.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            epochs (int): Number of training epochs
            model_save_path (str): Path to save the best model
            
        Returns:
            history: Training history object
        """
        print("Starting model training...")
        
        # Create callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train the model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate(self, val_dataset):
        """
        Evaluate the model on validation data and generate detailed metrics.
        
        Args:
            val_dataset: Validation dataset
            
        Returns:
            dict: Dictionary containing accuracy, loss, and classification report
        """
        print("Evaluating model on validation set...")
        
        loss, accuracy = self.model.evaluate(val_dataset, verbose=1)
        print(f"\nValidation Accuracy: {accuracy:.4f}")
        print(f"Validation Loss: {loss:.4f}")
        
        # Get predictions and true labels
        y_true = []
        y_pred = []
        
        for images, labels in val_dataset:
            predictions = self.model.predict(images, verbose=0)
            y_true.extend(np.argmax(labels.numpy(), axis=1))
            y_pred.extend(np.argmax(predictions, axis=1))
        
        # Generate classification report
        report = classification_report(
            y_true, y_pred,
            target_names=self.class_names,
            output_dict=True
        )
        
        print("\nClassification Report:")
        print(classification_report(
            y_true, y_pred,
            target_names=self.class_names
        ))
        
        metrics = {
            'accuracy': float(accuracy),
            'loss': float(loss),
            'classification_report': report
        }
        
        return metrics
    
    def plot_training_history(self, history, save_path='models/training_history.png'):
        """
        Plot training and validation accuracy/loss curves.
        
        Args:
            history: Training history object
            save_path (str): Path to save the plot
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot accuracy
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot loss
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
        plt.close()
    
    def save_model(self, model_path='models/plant_disease_model.h5'):
        """Save the trained model."""
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def save_class_names(self, class_names_path='models/class_names.json'):
        """Save class names for inference."""
        with open(class_names_path, 'w') as f:
            json.dump(self.class_names, f)
        print(f"Class names saved to {class_names_path}")
    
    def load_model(self, model_path='models/plant_disease_model.h5'):
        """Load a pre-trained model."""
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")
    
    def load_class_names(self, class_names_path='models/class_names.json'):
        """Load class names."""
        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)
        self.num_classes = len(self.class_names)
        print(f"Loaded {self.num_classes} class names")


def main():
    """
    Main function to demonstrate the complete training pipeline.
    """
    # Initialize model
    model_handler = PlantDiseaseModel(img_size=224, num_classes=38)
    
    # Load dataset
    # Update this path to point to your PlantVillage dataset
    data_path = 'data/PlantVillage/raw/color'  # Directory with subdirectories for each disease class
    
    try:
        train_dataset, val_dataset = model_handler.load_plantvillage_dataset(
            data_path,
            validation_split=0.2
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please ensure the PlantVillage dataset is in the 'data/plantvillage' directory")
        print("with subdirectories for each disease class containing images.")
        return
    
    # Build model
    model_handler.build_model(freeze_base=True, num_frozen_layers=150)
    model_handler.compile_model(learning_rate=1e-4)
    
    # Train model
    history = model_handler.train(
        train_dataset,
        val_dataset,
        epochs=50,
        model_save_path='models/best_model.h5'
    )
    
    # Plot training history
    model_handler.plot_training_history(history)
    
    # Evaluate model
    metrics = model_handler.evaluate(val_dataset)
    
    # Save model and class names
    model_handler.save_model('models/plant_disease_model.h5')
    model_handler.save_class_names('models/class_names.json')
    
    # Save metrics
    with open('models/metrics.json', 'w') as f:
        # Convert classification report to serializable format
        report = metrics['classification_report']
        json.dump({
            'accuracy': metrics['accuracy'],
            'loss': metrics['loss'],
            'classification_report': report
        }, f, indent=4)
    
    print("\nTraining completed successfully!")
    print(f"Best Model Accuracy: {metrics['accuracy']:.4f}")


if __name__ == '__main__':
    main()
