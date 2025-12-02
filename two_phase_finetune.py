"""
Two-phase fine-tune script
Phase A: replace classifier head, freeze backbone, train head-only for N epochs
Phase B: unfreeze top K layers of backbone, fine-tune for M epochs

Usage:
python two_phase_finetune.py --data-dir data/PlantVillage/raw/color --model models/plant_disease_model_finetuned.h5 \
  --class-names models/class_names.json --phaseA-epochs 5 --phaseB-epochs 8 --steps 150 --batch 32 --unfreeze-layers 30

Saves updated model to the same model path.
"""
import argparse
import json
import os
import math
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


def compute_class_counts(data_dir):
    counts = {}
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        imgs = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        counts[cls] = len(imgs)
    return counts


def build_class_weight_map(class_names, counts):
    total = sum(counts.get(c, 0) for c in class_names)
    num_classes = len(class_names)
    cw = {}
    for idx, cls in enumerate(class_names):
        cnt = counts.get(cls, 0)
        if cnt == 0:
            cw[idx] = 1.0
        else:
            cw[idx] = float(total) / (num_classes * cnt)
    return cw


def replace_head(model, num_classes):
    # Assume model ends with a Dense softmax layer. Replace it keeping preceding layers.
    x = model.layers[-2].output if len(model.layers) >= 2 else model.output
    new_out = keras.layers.Dense(num_classes, activation='softmax', name='predictions')(x)
    new_model = keras.Model(inputs=model.input, outputs=new_out)
    return new_model


def main(args):
    print('Loading model:', args.model)
    model = keras.models.load_model(args.model)

    with open(args.class_names, 'r') as f:
        class_names = json.load(f)
    num_classes = len(class_names)

    counts = compute_class_counts(args.data_dir)
    print('Sample counts for some classes:')
    sample_items = list(counts.items())[:5]
    for k, v in sample_items:
        print(' ', k, v)

    class_weight = build_class_weight_map(class_names, counts)
    # Increase Paddy emphasis
    for idx, cls in enumerate(class_names):
        if cls.startswith('Paddy_'):
            class_weight[idx] *= 2.0

    print('\nPhase A: Replacing head and training head-only')
    # Create new head
    # If model already has correct output shape, we still re-create the head to ensure clean training
    base_layers = model.layers[:-1]
    # Build new model by reusing everything except final layer
    penultimate = model.layers[-2].output if len(model.layers) >= 2 else model.output
    new_logits = keras.layers.Dense(num_classes, activation='softmax', name='predictions')(penultimate)
    new_model = keras.Model(inputs=model.input, outputs=new_logits)

    # Freeze backbone (all layers except the new head)
    for layer in new_model.layers[:-1]:
        layer.trainable = False
    new_model.layers[-1].trainable = True

    new_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    train_flow = train_gen.flow_from_directory(
        args.data_dir,
        target_size=(192, 192),
        batch_size=args.batch,
        class_mode='categorical'
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(args.model, save_best_only=True, monitor='loss'),
        keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    ]

    print('Training head-only for', args.phaseA_epochs, 'epochs, steps_per_epoch=', args.steps)
    new_model.fit(train_flow, epochs=args.phaseA_epochs, steps_per_epoch=args.steps, class_weight=class_weight, callbacks=callbacks)

    print('\nPhase B: Unfreeze top layers and fine-tune')
    # Unfreeze top N layers of the backbone
    unfreeze = args.unfreeze_layers
    if unfreeze > 0:
        total_layers = len(new_model.layers)
        # We will unfreeze last `unfreeze` layers (but keep input and early backbone frozen)
        for layer in new_model.layers[-unfreeze:]:
            layer.trainable = True

    # Recompile with low LR
    new_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

    # New callbacks
    callbacks2 = [
        keras.callbacks.ModelCheckpoint(args.model, save_best_only=True, monitor='loss'),
        keras.callbacks.EarlyStopping(monitor='loss', patience=4, restore_best_weights=True)
    ]

    print('Fine-tuning for', args.phaseB_epochs, 'epochs, steps_per_epoch=', args.steps)
    new_model.fit(train_flow, epochs=args.phaseB_epochs, steps_per_epoch=args.steps, class_weight=class_weight, callbacks=callbacks2)

    print('Saving final model to', args.model)
    new_model.save(args.model)
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/PlantVillage/raw/color')
    parser.add_argument('--model', default='models/plant_disease_model_finetuned.h5')
    parser.add_argument('--class-names', default='models/class_names.json')
    parser.add_argument('--phaseA-epochs', type=int, default=5)
    parser.add_argument('--phaseB-epochs', type=int, default=8)
    parser.add_argument('--steps', type=int, default=150)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--unfreeze-layers', type=int, default=30)
    parser.add_argument('--img-size', type=int, default=192)
    args = parser.parse_args()
    main(args)
