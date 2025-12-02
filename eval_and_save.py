"""
Evaluation and saving helper
Loads the best checkpoint saved during training (`models/best_model.h5`), builds a
validation generator from the PlantVillage data, evaluates the model, computes the
classification report, and saves the final model and class names.
"""
import os
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report


def main():
    data_path = 'data/PlantVillage/raw/color'
    model_checkpoint = 'models/best_model.h5'
    final_model_path = 'models/plant_disease_model.h5'
    class_names_path = 'models/class_names.json'
    metrics_path = 'models/metrics.json'
    img_size = 192
    batch_size = 64

    if not os.path.exists(model_checkpoint):
        print(f'Checkpoint not found: {model_checkpoint}')
        return

    print('Loading model from', model_checkpoint)
    model = keras.models.load_model(model_checkpoint)

    # Create validation generator
    val_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
    val_gen = val_datagen.flow_from_directory(
        data_path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    # Extract class names
    class_names = list(val_gen.class_indices.keys())

    # Evaluate
    print('\nEvaluating model on validation set...')
    loss, acc = model.evaluate(val_gen, verbose=1)
    print(f'Validation accuracy: {acc:.4f}, loss: {loss:.4f}')

    # Predictions for classification report
    y_true = []
    y_pred = []

    for images, labels in val_gen:
        preds = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    print('\nClassification report:')
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save artifacts
    print('\nSaving final model and metadata...')
    model.save(final_model_path)
    with open(class_names_path, 'w') as f:
        json.dump(class_names, f, indent=2)
    with open(metrics_path, 'w') as f:
        json.dump({'accuracy': acc, 'loss': loss, 'classification_report': report}, f, indent=2)

    print('Saved:')
    print(' -', final_model_path)
    print(' -', class_names_path)
    print(' -', metrics_path)


if __name__ == '__main__':
    main()
