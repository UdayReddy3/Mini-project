"""
Plant Disease Prediction Module
This module provides inference functions for single image prediction
using the trained plant disease detection model.
"""

import os
import json
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from datetime import datetime


class DiseasePredictor:
    """
    A class to handle predictions on single plant disease images.
    """
    
    def __init__(self, model_path='models/plant_disease_model_finetuned.h5', 
                 class_names_path='models/class_names.json', img_size=192):
        """
        Initialize the predictor with a trained model.
        
        Args:
            model_path (str): Path to the trained model
            class_names_path (str): Path to the class names JSON file
            img_size (int): Target image size (default: 192 for MobileNetV2)
        """
        self.img_size = img_size
        self.model = None
        self.class_names = []
        self.model_path = model_path
        self.class_names_path = class_names_path
        
        self.load_model_and_classes()
    
    def load_model_and_classes(self):
        """Load the pre-trained model and class names."""
        try:
            self.model = keras.models.load_model(self.model_path)
            print(f"Model loaded from {self.model_path}")
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            with open(self.class_names_path, 'r') as f:
                self.class_names = json.load(f)
            print(f"Loaded {len(self.class_names)} disease classes")
        except FileNotFoundError:
            raise FileNotFoundError(f"Class names file not found: {self.class_names_path}")
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess a single image for prediction.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: Preprocessed image array
            PIL.Image: Original image for display
            
        Raises:
            ValueError: If image cannot be loaded or is invalid
        """
        try:
            # Load image using PIL
            image = Image.open(image_path).convert('RGB')
            
            # Resize image
            image_resized = image.resize((self.img_size, self.img_size))
            
            # Convert to numpy array
            image_array = np.array(image_resized, dtype=np.float32)
            
            # Normalize pixel values to [0, 1]
            image_array = image_array / 255.0
            
            # Add batch dimension
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array, image
        
        except FileNotFoundError:
            raise ValueError(f"Image file not found: {image_path}")
        except Exception as e:
            raise ValueError(f"Error loading image: {str(e)}")
    
    def predict_disease(self, image_path):
        """
        Predict the disease from a single image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Dictionary containing:
                - 'disease_class': Predicted disease class name
                - 'confidence': Confidence score (0-100)
                - 'all_predictions': Dictionary of all classes with confidence scores
                - 'timestamp': Timestamp of prediction
                - 'image_path': Original image path
        """
        try:
            # Preprocess image
            image_array, original_image = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(image_array, verbose=0)
            
            # Get the predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx]) * 100
            
            # Create dictionary of all predictions
            all_predictions = {
                self.class_names[i]: float(predictions[0][i]) * 100 
                for i in range(len(self.class_names))
            }
            
            # Sort by confidence
            all_predictions = dict(sorted(
                all_predictions.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            result = {
                'disease_class': predicted_class,
                'confidence': round(confidence, 2),
                'all_predictions': all_predictions,
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'top_3_predictions': list(all_predictions.items())[:3]
            }
            
            return result
        
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def batch_predict(self, image_dir):
        """
        Predict diseases for all images in a directory.
        
        Args:
            image_dir (str): Directory containing images
            
        Returns:
            list: List of prediction results
        """
        results = []
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        try:
            for filename in os.listdir(image_dir):
                if os.path.splitext(filename)[1].lower() in supported_extensions:
                    image_path = os.path.join(image_dir, filename)
                    try:
                        result = self.predict_disease(image_path)
                        results.append(result)
                        print(f"✓ Predicted: {filename} -> {result['disease_class']} ({result['confidence']}%)")
                    except Exception as e:
                        print(f"✗ Failed to predict {filename}: {str(e)}")
            
            return results
        
        except FileNotFoundError:
            raise ValueError(f"Directory not found: {image_dir}")
    
    def predict_from_array(self, image_array):
        """
        Predict disease from a numpy array (useful for web apps).
        
        Args:
            image_array (np.ndarray): Image array (H, W, 3) with values in [0, 1] or [0, 255]
            
        Returns:
            dict: Prediction result
        """
        try:
            # Check if values are in [0, 255] range and normalize if needed
            if image_array.max() > 1.0:
                image_array = image_array / 255.0
            
            # Resize if necessary
            if image_array.shape[:2] != (self.img_size, self.img_size):
                image_pil = Image.fromarray((image_array * 255).astype('uint8'))
                image_pil = image_pil.resize((self.img_size, self.img_size))
                image_array = np.array(image_pil, dtype=np.float32) / 255.0
            
            # Add batch dimension
            image_batch = np.expand_dims(image_array, axis=0)
            
            # Make prediction
            predictions = self.model.predict(image_batch, verbose=0)
            
            # Get the predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx]) * 100
            
            # Create dictionary of all predictions
            all_predictions = {
                self.class_names[i]: float(predictions[0][i]) * 100 
                for i in range(len(self.class_names))
            }
            
            # Sort by confidence
            all_predictions = dict(sorted(
                all_predictions.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            
            result = {
                'disease_class': predicted_class,
                'confidence': round(confidence, 2),
                'all_predictions': all_predictions,
                'timestamp': datetime.now().isoformat(),
                'top_3_predictions': list(all_predictions.items())[:3]
            }
            
            return result
        
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")


def predict_disease(image_path, model_path='models/plant_disease_model_finetuned.h5',
                    class_names_path='models/class_names.json'):
    """
    Standalone function to predict disease from a single image.
    This is the main inference function that can be used directly.
    
    Args:
        image_path (str): Path to the image file
        model_path (str): Path to the trained model
        class_names_path (str): Path to the class names JSON file
        
    Returns:
        dict: Prediction result containing disease class, confidence score, etc.
        
    Example:
        >>> result = predict_disease('path/to/image.jpg')
        >>> print(f"Disease: {result['disease_class']}")
        >>> print(f"Confidence: {result['confidence']}%")
    """
    predictor = DiseasePredictor(model_path, class_names_path)
    return predictor.predict_disease(image_path)


def main():
    """
    Demo function showing how to use the predictor.
    """
    print("=" * 60)
    print("Plant Disease Detection - Inference Module")
    print("=" * 60)
    
    # Initialize predictor
    try:
        predictor = DiseasePredictor(
            model_path='models/plant_disease_model_finetuned.h5',
            class_names_path='models/class_names.json'
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the model and class names files exist in the 'models' directory.")
        return
    
    # Example: Predict from a single image
    # Replace with your test image path
    test_image = 'test_image.jpg'
    
    if os.path.exists(test_image):
        try:
            result = predictor.predict_disease(test_image)
            
            print(f"\nPrediction Result:")
            print(f"  Disease Class: {result['disease_class']}")
            print(f"  Confidence: {result['confidence']}%")
            print(f"  Timestamp: {result['timestamp']}")
            print(f"\nTop 3 Predictions:")
            for i, (disease, confidence) in enumerate(result['top_3_predictions'], 1):
                print(f"  {i}. {disease}: {confidence:.2f}%")
        
        except Exception as e:
            print(f"Prediction error: {e}")
    else:
        print(f"Test image not found: {test_image}")
        print("To test predictions, provide a plant disease image in JPG or PNG format.")
        print("\nUsage:")
        print("  from predict import predict_disease")
        print("  result = predict_disease('path/to/image.jpg')")
        print("  print(result['disease_class'], result['confidence'])")


if __name__ == '__main__':
    main()
