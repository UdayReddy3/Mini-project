"""
Fine-tune script: Load the existing trained model and fine-tune on new Paddy dataset classes.

Usage:
    python fine_tune.py \
        --data-dir "data/PlantVillage/raw/color" \
        --model-path "models/plant_disease_model.h5" \
        --class-names-path "models/class_names.json" \
        --output-path "models/plant_disease_model_finetuned.h5" \
        [--epochs 10] [--batch-size 32] [--learning-rate 0.0001]

What this does:
1. Loads existing model and class_names.json
2. Rebuilds final Dense layer to match total number of classes (PlantVillage + Paddy)
3. Recompiles with new output shape
4. Fine-tunes on combined dataset with low learning rate to preserve learned features
5. Saves new model to output path
"""

import argparse
import os
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split


def load_class_names(class_names_path):
    """Load class names from JSON file."""
    with open(class_names_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def count_images_in_classes(data_dir, class_names):
    """Count images per class folder."""
    counts = {}
    for cls in class_names:
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            counts[cls] = len(imgs)
        else:
            counts[cls] = 0
    return counts


def rebuild_model_for_new_classes(model, num_new_classes):
    """
    Rebuild model's final Dense layer to output num_new_classes.
    Preserves all earlier layers and their weights.
    """
    # Clone model and modify last layer
    new_model = keras.models.clone_model(model)
    
    # Remove the last (output) layer and rebuild
    new_model.layers[-1]._name = 'predictions_old'
    
    # Build Sequential model by stacking all layers except the last
    model_config = model.get_config()
    layers_config = model_config['layers'][:-1]  # Remove last layer
    
    # Add new Dense output layer
    layers_config.append({
        'class_name': 'Dense',
        'config': {
            'name': 'predictions',
            'dtype': 'float32',
            'trainable': True,
            'units': num_new_classes,
            'activation': 'softmax'
        }
    })
    
    # Create new Sequential model with updated config
    model_config['layers'] = layers_config
    new_model = keras.Sequential.from_config(model_config)
    
    # Copy weights from old model to new model (except last layer)
    for i, old_layer in enumerate(model.layers[:-1]):
        if i < len(new_model.layers) - 1:
            try:
                new_model.layers[i].set_weights(old_layer.get_weights())
            except Exception:
                pass
    
    return new_model


def create_data_generators(image_size=192, batch_size=32):
    """Create ImageDataGenerators for training and validation."""
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_gen = ImageDataGenerator(rescale=1./255)
    
    return train_gen, val_gen


def fine_tune_model(
    model,
    data_dir,
    class_names,
    epochs=10,
    batch_size=32,
    learning_rate=0.0001,
    image_size=192,
    validation_split=0.2
):
    """
    Fine-tune model on dataset in data_dir.
    
    Args:
        model: Keras model to fine-tune
        data_dir: Root directory containing class subdirectories
        class_names: List of class folder names
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        image_size: Target image size (assumes square)
        validation_split: Fraction of data to use for validation
    
    Returns:
        history: Training history object
    """
    train_gen, val_gen = create_data_generators(image_size, batch_size)
    
    # Use flow_from_directory to load images
    train_flow = train_gen.flow_from_directory(
        data_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        classes=class_names,
        shuffle=True
    )
    
    val_flow = val_gen.flow_from_directory(
        data_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode='categorical',
        classes=class_names,
        shuffle=False
    )
    
    # Compile model with low learning rate for fine-tuning
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Fine-tuning model with {len(class_names)} classes for {epochs} epochs...")
    
    history = model.fit(
        train_flow,
        steps_per_epoch=train_flow.samples // batch_size,
        validation_data=val_flow,
        validation_steps=val_flow.samples // batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True
            )
        ]
    )
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Fine-tune model for new disease classes (e.g., Paddy)')
    parser.add_argument('--data-dir', default='data/PlantVillage/raw/color', help='Root dataset directory')
    parser.add_argument('--model-path', default='models/plant_disease_model.h5', help='Path to existing model')
    parser.add_argument('--class-names-path', default='models/class_names.json', help='Path to class_names.json')
    parser.add_argument('--output-path', default='models/plant_disease_model_finetuned.h5', help='Output model path')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.0001, help='Fine-tune learning rate')
    parser.add_argument('--image-size', type=int, default=192, help='Target image size')
    args = parser.parse_args()

    # Load class names
    print(f"Loading class names from {args.class_names_path}...")
    class_names = load_class_names(args.class_names_path)
    print(f"Total classes: {len(class_names)}")
    
    # Count images per class
    counts = count_images_in_classes(args.data_dir, class_names)
    print("Images per class:")
    for cls, count in sorted(counts.items()):
        print(f"  {cls}: {count}")
    
    # Load existing model
    print(f"Loading model from {args.model_path}...")
    old_model = keras.models.load_model(args.model_path)
    old_num_classes = old_model.output_shape[1]
    new_num_classes = len(class_names)
    
    print(f"Old model outputs: {old_num_classes} classes")
    print(f"New model outputs: {new_num_classes} classes")
    
    if old_num_classes == new_num_classes:
        print("Same number of classes; no rebuild needed. Using original model.")
        model = old_model
    else:
        print(f"Rebuilding model to output {new_num_classes} classes...")
        model = rebuild_model_for_new_classes(old_model, new_num_classes)
    
    # Fine-tune
    print(f"Starting fine-tuning on data in {args.data_dir}...")
    history = fine_tune_model(
        model,
        args.data_dir,
        class_names,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        image_size=args.image_size
    )
    
    # Save model
    print(f"Saving fine-tuned model to {args.output_path}...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    model.save(args.output_path)
    print(f"✓ Model saved: {args.output_path}")
    print("\nFine-tuning complete!")
    print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")


if __name__ == '__main__':
    main()
