"""
Targeted fine-tune to improve Paddy class recognition.
- Loads `models/plant_disease_model_finetuned.h5`
- Computes per-class image counts and class weights
- Uses ImageDataGenerator with augmentation
- Trains with `class_weight` to boost Paddy classes
- Saves model back to same path

Usage: python targeted_finetune_paddy.py --epochs 8 --steps 200 --batch 32
"""
import argparse
import json
import os
import shutil
import tempfile
import time
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    # Use inverse frequency weighting: weight_c = total_samples / (num_classes * count_c)
    total = sum(counts.get(c, 0) for c in class_names)
    num_classes = len(class_names)
    class_weight = {}
    for idx, cls in enumerate(class_names):
        cnt = counts.get(cls, 0)
        if cnt == 0:
            class_weight[idx] = 1.0
        else:
            class_weight[idx] = float(total) / (num_classes * cnt)
    return class_weight


def main(args):
    # paths
    model_path = args.model_path
    class_names_path = args.class_names
    data_dir = args.data_dir

    print('Loading model:', model_path)
    model = keras.models.load_model(model_path)

    with open(class_names_path, 'r') as f:
        class_names = json.load(f)

    print('Counting images per class...')
    counts = compute_class_counts(data_dir)
    for c in sorted(counts.keys()):
        print(f'  {c}: {counts[c]}')

    class_weight = build_class_weight_map(class_names, counts)

    # Boost Paddy classes slightly
    for idx, cls in enumerate(class_names):
        if cls.startswith('Paddy_'):
            class_weight[idx] *= 2.0  # give extra emphasis to Paddy

    print('\nSample class_weight (first 10):')
    for k in list(class_weight.keys())[:10]:
        print(k, class_weight[k])

    # Generators
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Optionally create an oversampled temporary dataset to emphasize Paddy classes
    temp_dir = None
    train_data_dir = data_dir
    if args.oversample_factor and args.oversample_factor > 1:
        temp_dir = os.path.join(tempfile.gettempdir(), f"pv_oversample_{int(time.time())}")
        print('Creating oversampled dataset at', temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        for cls in os.listdir(data_dir):
            src_cls = os.path.join(data_dir, cls)
            if not os.path.isdir(src_cls):
                continue
            dst_cls = os.path.join(temp_dir, cls)
            os.makedirs(dst_cls, exist_ok=True)
            imgs = [f for f in os.listdir(src_cls) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img in imgs:
                src_path = os.path.join(src_cls, img)
                # copy original once
                shutil.copy2(src_path, os.path.join(dst_cls, img))
                # if Paddy class, duplicate extra copies
                if cls.startswith('Paddy_'):
                    for i in range(args.oversample_factor - 1):
                        name, ext = os.path.splitext(img)
                        dst_name = f"{name}_dup{i}{ext}"
                        shutil.copy2(src_path, os.path.join(dst_cls, dst_name))
        train_data_dir = temp_dir

    train_flow = train_gen.flow_from_directory(
        train_data_dir,
        target_size=(192, 192),
        batch_size=args.batch_size,
        class_mode='categorical',
        shuffle=True
    )

    # Compile: lower LR, small steps
    opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    ckpt = keras.callbacks.ModelCheckpoint(args.model_path, save_best_only=True, monitor='loss')
    early = keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

    print(f"\nStarting training: epochs={args.epochs}, steps_per_epoch={args.steps}")
    history = model.fit(
        train_flow,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        class_weight=class_weight,
        callbacks=[ckpt, early]
    )

    print('Training complete. Saving model to', args.model_path)
    model.save(args.model_path)
    print('Done.')

    # Cleanup temporary oversampled data
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print('Removed temporary oversampled dataset', temp_dir)
        except Exception:
            print('Warning: failed to remove temp dir', temp_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', default='data/PlantVillage/raw/color')
    parser.add_argument('--model-path', default='models/plant_disease_model_finetuned.h5')
    parser.add_argument('--class-names', default='models/class_names.json')
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--oversample-factor', type=int, default=1,
                        help='How many times to duplicate Paddy class images (1 = no oversample)')
    args = parser.parse_args()
    main(args)
