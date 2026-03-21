"""
Preprocess aPY (Attribute Pascal & Yahoo) dataset for DDECCS pipeline.

Converts the raw Farhadi et al. (2009) annotation format into the same
structure used by AwA2:
  - JPEGImages/<class_name>/image.jpg
  - classes.txt
  - predicate-matrix-continuous.txt  (per-class mean attributes)
  - predicates.txt
  - labels.txt

The aPY dataset has PER-OBJECT annotations (multiple objects per image,
each with its own bounding box and 64 binary attributes).
We treat each object crop as a separate instance, matching DDC's setup.

DDC paper: "64 semantic attributes and 5274 instances, 15 classes"
DDC uses 15 classes for clustering. We keep all 32 and let --n_clusters control K.

Usage:
    python3 setup_apy.py \
        --annotation_dir data/aPY-data/attribute_data \
        --pascal_images /path/to/VOCdevkit/VOC2008/JPEGImages \
        --yahoo_images /path/to/ayahoo_test_images \
        --output_dir data/aPY-data/aPY
"""

import argparse
import logging
import os
import shutil
from collections import defaultdict

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_annotation_file(filepath):
    """Parse aPY annotation file. Returns list of (img_name, class_name, bbox, attributes)."""
    entries = []
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 70:  # img + class + 4 bbox + 64 attrs = 70
                continue
            img_name = parts[0]
            class_name = parts[1]
            bbox = [int(x) for x in parts[2:6]]
            attrs = [int(x) for x in parts[6:70]]
            entries.append((img_name, class_name, bbox, attrs))
    return entries


def find_image(img_name, pascal_dir, yahoo_dir):
    """Find image in Pascal or Yahoo directories."""
    # Try Pascal first
    pascal_path = os.path.join(pascal_dir, img_name)
    if os.path.exists(pascal_path):
        return pascal_path

    # Try Yahoo (different naming)
    yahoo_path = os.path.join(yahoo_dir, img_name)
    if os.path.exists(yahoo_path):
        return yahoo_path

    # Try without extension variations
    for d in [pascal_dir, yahoo_dir]:
        if d and os.path.exists(d):
            for f in os.listdir(d):
                if f.startswith(img_name.split('.')[0]):
                    return os.path.join(d, f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Preprocess aPY dataset")
    parser.add_argument("--annotation_dir", required=True,
                        help="Path to attribute_data/ directory")
    parser.add_argument("--pascal_images", required=True,
                        help="Path to VOC2008/JPEGImages/")
    parser.add_argument("--yahoo_images", default=None,
                        help="Path to ayahoo_test_images/ (optional)")
    parser.add_argument("--output_dir", required=True,
                        help="Output directory (e.g., data/aPY-data/aPY)")
    parser.add_argument("--crop_objects", action="store_true", default=True,
                        help="Crop objects using bounding boxes (default: True)")
    args = parser.parse_args()

    ann_dir = args.annotation_dir
    out_dir = args.output_dir

    # Read class and attribute names
    with open(os.path.join(ann_dir, "class_names.txt")) as f:
        class_names = f.read().strip().split()
    logging.info(f"Classes ({len(class_names)}): {class_names}")

    with open(os.path.join(ann_dir, "attribute_names.txt")) as f:
        attr_names = [line.strip() for line in f if line.strip()]
    logging.info(f"Attributes ({len(attr_names)}): {attr_names[:10]}...")

    # Parse all annotation files
    all_entries = []
    for split_file in ["apascal_train.txt", "apascal_test.txt", "ayahoo_test.txt"]:
        filepath = os.path.join(ann_dir, split_file)
        if os.path.exists(filepath):
            entries = parse_annotation_file(filepath)
            logging.info(f"{split_file}: {len(entries)} objects")
            all_entries.extend(entries)

    logging.info(f"Total objects: {len(all_entries)}")

    # Count per class
    class_counts = defaultdict(int)
    for _, cls, _, _ in all_entries:
        class_counts[cls] += 1
    for cls in sorted(class_counts):
        logging.info(f"  {cls}: {class_counts[cls]} objects")

    # Create output structure
    img_out_dir = os.path.join(out_dir, "JPEGImages")
    os.makedirs(img_out_dir, exist_ok=True)

    # Process each object
    labels_lines = []
    class_attrs = defaultdict(list)  # for computing per-class mean attributes
    obj_count = 0
    missing = 0

    for img_name, cls, bbox, attrs in all_entries:
        src_path = find_image(img_name, args.pascal_images, args.yahoo_images)
        if src_path is None:
            missing += 1
            continue

        # Create class directory
        cls_dir = os.path.join(img_out_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)

        # Crop object from image using bbox
        if args.crop_objects:
            try:
                img = Image.open(src_path).convert("RGB")
                x1, y1, x2, y2 = bbox
                # Clamp to image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.width, x2), min(img.height, y2)
                if x2 > x1 and y2 > y1:
                    crop = img.crop((x1, y1, x2, y2))
                else:
                    crop = img
            except Exception as e:
                logging.warning(f"Error processing {src_path}: {e}")
                continue
        else:
            crop = None

        # Save with unique name: class_imgname_objN.jpg
        out_name = f"{os.path.splitext(img_name)[0]}_obj{obj_count:05d}.jpg"
        out_path = os.path.join(cls_dir, out_name)

        if args.crop_objects and crop is not None:
            crop.save(out_path)
        else:
            shutil.copy(src_path, out_path)

        labels_lines.append(f"{cls}/{out_name}")
        class_attrs[cls].append(attrs)
        obj_count += 1

    logging.info(f"Processed {obj_count} objects ({missing} images not found)")

    # Write labels.txt
    with open(os.path.join(out_dir, "labels.txt"), "w") as f:
        for line in labels_lines:
            f.write(line + "\n")

    # Write classes.txt (1-indexed to match AwA2 format)
    with open(os.path.join(out_dir, "classes.txt"), "w") as f:
        for i, cls in enumerate(class_names):
            f.write(f"{i + 1} {cls}\n")

    # Write predicates.txt (attribute names)
    with open(os.path.join(out_dir, "predicates.txt"), "w") as f:
        for i, name in enumerate(attr_names):
            f.write(f"{i + 1}\t{name}\n")

    # Write predicate-matrix-continuous.txt (per-class mean attributes)
    pred_matrix = np.zeros((len(class_names), len(attr_names)))
    for i, cls in enumerate(class_names):
        if cls in class_attrs:
            pred_matrix[i] = np.mean(class_attrs[cls], axis=0)
    np.savetxt(os.path.join(out_dir, "predicate-matrix-continuous.txt"), pred_matrix, fmt="%.6f")

    logging.info(f"Predicate matrix: {pred_matrix.shape}")
    logging.info(f"Output directory: {out_dir}")
    logging.info(f"Files created: labels.txt, classes.txt, predicates.txt, "
                 f"predicate-matrix-continuous.txt, JPEGImages/")
    logging.info("Done")


if __name__ == "__main__":
    main()