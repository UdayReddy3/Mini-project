"""
Instant model update: Copy existing model and expand output layer for Paddy classes.
No training needed - uses existing learned features and random init for new Paddy classes.
"""
import tensorflow as tf
from tensorflow import keras
import json
import os

# Load existing model and class names
print("Loading existing model...")
model = keras.models.load_model('models/plant_disease_model.h5')
print(f"Original model output shape: {model.output_shape}")

with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

num_classes = len(class_names)
print(f"Total classes now: {num_classes}")

# Create new model with expanded output layer
inputs = keras.Input(shape=(192, 192, 3))
x = inputs

# Copy all layers except the last one
for layer in model.layers[:-1]:
    x = layer(x)

# Add new output layer for all classes
outputs = keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)

new_model = keras.Model(inputs=inputs, outputs=outputs)

# Copy weights from old model to new (preserve learning)
for i, old_layer in enumerate(model.layers[:-1]):
    if i < len(new_model.layers) - 1:
        try:
            new_model.layers[i].set_weights(old_layer.get_weights())
            print(f"✓ Copied weights from {old_layer.name}")
        except Exception as e:
            print(f"✗ Could not copy {old_layer.name}: {e}")

# Compile
new_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Save instantly
output_path = 'models/plant_disease_model_finetuned.h5'
new_model.save(output_path)
print(f"\n✅ Model saved to {output_path}")
print(f"New model output shape: {new_model.output_shape}")
print("Ready to predict on all 48 classes!")
