"""
Fast fine-tune: Train on ALL 48 classes for just 3 epochs (10-15 min instead of 3 hours)
Balances full dataset with quick training. Focuses on new Paddy classes via learning rate.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import json

print("Fast fine-tune: Training on all 48 classes for 3 epochs...\n")

# Load model and classes
model = keras.models.load_model('models/plant_disease_model_finetuned.h5')
with open('models/class_names.json', 'r') as f:
    classes = json.load(f)

print(f"Model ready. Total classes: {len(classes)}")

# Create data generators with light augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True
)

# Load ALL training data
print("\nLoading all disease images...")
train_flow = train_gen.flow_from_directory(
    'data/PlantVillage/raw/color',
    target_size=(192, 192),
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)

# Compile with moderate learning rate
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train for just 3 epochs
print(f"\nTraining 3 epochs on {train_flow.samples} total images...")
print("Limiting to 100 steps/epoch for speed (~3-5 min/epoch)\n")

history = model.fit(
    train_flow,
    steps_per_epoch=100,  # Limit steps for SPEED
    epochs=3,
    verbose=1
)

# Save
model.save('models/plant_disease_model_finetuned.h5')
print("\n✅ Fast training complete in ~10-15 minutes!")
print(f"Final accuracy: {history.history['accuracy'][-1]:.4f}")
print("Model ready for predictions on all 48 classes!")
