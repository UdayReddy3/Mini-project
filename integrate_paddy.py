"""
Script to integrate a Paddy dataset into the existing PlantVillage-style dataset.

Usage:
    python integrate_paddy.py --source "C:/path/to/paddy_dataset" \
        [--target "data/PlantVillage/raw/color"] [--prefix "Paddy_"]

Expectations for source folder:
- Either a directory with subfolders per-class (recommended) OR a single folder of images that should be placed under a single class.

What the script does:
- Copies folders (or files) from source into target directory, creating new class folders.
- Updates `models/class_names.json` by adding any new class folder names (avoids duplicates).
- Optionally creates a placeholder JSON in `precautions_extra/` for each new class so you can add disease-specific precautions.

Note: This script does not retrain the model. After integrating classes you must retrain or fine-tune the model (see README for training instructions).
"""

import argparse
import os
import shutil
import json


def safe_makedirs(path):
    os.makedirs(path, exist_ok=True)


def copy_class_folder(src_folder, target_root, prefix=None):
    """Copy a folder of images into target_root with optional prefix for classname."""
    if not os.path.isdir(src_folder):
        return None
    class_name = os.path.basename(src_folder)
    if prefix:
        new_name = f"{prefix}{class_name}"
    else:
        new_name = class_name
    target_dir = os.path.join(target_root, new_name)
    safe_makedirs(target_dir)
    # Copy files
    for fname in os.listdir(src_folder):
        s = os.path.join(src_folder, fname)
        d = os.path.join(target_dir, fname)
        if os.path.isfile(s):
            try:
                shutil.copy2(s, d)
            except Exception:
                continue
    return new_name


def copy_single_folder_of_images(src_folder, target_root, class_name):
    """If source is a single folder of images, copy into target_root/class_name"""
    target_dir = os.path.join(target_root, class_name)
    safe_makedirs(target_dir)
    for fname in os.listdir(src_folder):
        s = os.path.join(src_folder, fname)
        d = os.path.join(target_dir, fname)
        if os.path.isfile(s):
            try:
                shutil.copy2(s, d)
            except Exception:
                continue
    return class_name


def update_class_names(models_dir, new_classes):
    """Update models/class_names.json adding any new classes."""
    class_file = os.path.join(models_dir, 'class_names.json')
    classes = []
    if os.path.exists(class_file):
        try:
            with open(class_file, 'r', encoding='utf-8') as f:
                classes = json.load(f)
        except Exception:
            classes = []
    # Ensure list
    if not isinstance(classes, list):
        classes = list(classes)
    added = []
    for cls in new_classes:
        if cls not in classes:
            classes.append(cls)
            added.append(cls)
    # Save
    safe_makedirs(models_dir)
    with open(class_file, 'w', encoding='utf-8') as f:
        json.dump(classes, f, indent=2, ensure_ascii=False)
    return added


def create_precautions_placeholders(new_classes, out_dir='precautions_extra'):
    """Create placeholder precaution JSON entries for each new class to be filled later by user or extension."""
    safe_makedirs(out_dir)
    payload = {}
    for cls in new_classes:
        # create a minimal structure
        payload[cls] = {
            "disease_name": cls.replace('_', ' '),
            "severity": "Unknown",
            "description": "Placeholder entry for this disease. Please update with local recommendations.",
            "symptoms": [],
            "precautions": [
                "Monitor crop and consult local extension services",
                "Keep records and isolate infected plants"
            ],
            "chemical_treatment": [],
            "natural_treatment": [],
            "time_to_recovery": "Varies",
            "yield_impact": "Unknown",
            "cost_effectiveness": "Unknown"
        }
    if payload:
        out_path = os.path.join(out_dir, 'paddy_placeholders.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return out_path
    return None


def main():
    parser = argparse.ArgumentParser(description='Integrate Paddy dataset into PlantVillage structure')
    parser.add_argument('--source', '-s', required=True, help='Path to Paddy dataset folder')
    parser.add_argument('--target', '-t', default='data/PlantVillage/raw/color', help='Target dataset root')
    parser.add_argument('--prefix', '-p', default='Paddy_', help='Prefix to add to new class folder names')
    parser.add_argument('--models-dir', default='models', help='Models directory containing class_names.json')
    args = parser.parse_args()

    src = os.path.abspath(args.source)
    tgt = os.path.abspath(args.target)

    if not os.path.exists(src):
        print(f"Source path not found: {src}")
        return

    safe_makedirs(tgt)
    new_classes = []

    # If source has subfolders, treat each as a class folder
    subfolders = [os.path.join(src, d) for d in os.listdir(src) if os.path.isdir(os.path.join(src, d))]
    if subfolders:
        print("Detected subfolders in source; importing each as a class folder...")
        for sf in subfolders:
            added = copy_class_folder(sf, tgt, prefix=args.prefix)
            if added:
                new_classes.append(added)
    else:
        # Single folder of images: create a single class named prefix+basename
        class_name = f"{args.prefix}{os.path.basename(src)}"
        copy_single_folder_of_images(src, tgt, class_name)
        new_classes.append(class_name)

    # Update class_names.json
    added = update_class_names(args.models_dir, new_classes)
    print(f"Added classes to class_names.json: {added}")

    # Create placeholder precautions
    placeholders = create_precautions_placeholders(added)
    if placeholders:
        print(f"Created precaution placeholders: {placeholders}")

    print("Integration complete. Next: retrain or fine-tune the model to include new classes.")


if __name__ == '__main__':
    main()
