"""
Data Utilities Module
=====================

Functions for loading, validating, and preprocessing the candlestick pattern dataset.
Updated for 6-class dataset with train/valid/test splits.
"""

import os
import yaml
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

from src.config import (
    RAW_DATA_DIR,
    DATASET_YAML,
    CLASS_NAMES,
    CLASS_ID_TO_NAME
)


# =============================================================================
# DATASET VALIDATION
# =============================================================================

def verify_dataset_structure() -> bool:
    """
    Verify that the dataset is properly structured.
    
    Expected structure:
        data/raw/
        ├── train/images/, train/labels/
        ├── valid/images/, valid/labels/
        ├── test/images/, test/labels/
        └── data.yaml
    """
    required_paths = [
        RAW_DATA_DIR / "train" / "images",
        RAW_DATA_DIR / "train" / "labels",
        RAW_DATA_DIR / "valid" / "images",
        RAW_DATA_DIR / "valid" / "labels",
        DATASET_YAML
    ]
    
    # Optional test folder
    optional_paths = [
        RAW_DATA_DIR / "test" / "images",
        RAW_DATA_DIR / "test" / "labels",
    ]
    
    all_valid = True
    print("Checking dataset structure...")
    print("-" * 40)
    
    for path in required_paths:
        exists = path.exists()
        status = "OK" if exists else "MISSING"
        print(f"  [{status}] {path.relative_to(RAW_DATA_DIR.parent)}")
        if not exists:
            all_valid = False
    
    for path in optional_paths:
        exists = path.exists()
        status = "OK" if exists else "OPTIONAL"
        print(f"  [{status}] {path.relative_to(RAW_DATA_DIR.parent)}")
    
    print("-" * 40)
    
    if all_valid:
        print("Dataset structure verified successfully!")
    else:
        print("ERROR: Dataset incomplete. Please check data/raw/ folder.")
    
    return all_valid


def count_dataset_statistics() -> Dict:
    """Calculate dataset statistics including image counts and class distribution."""
    stats = {
        "train_images": 0,
        "valid_images": 0,
        "test_images": 0,
        "train_instances": Counter(),
        "valid_instances": Counter(),
        "test_instances": Counter(),
        "total_instances": 0
    }
    
    splits = ["train", "valid", "test"]
    
    for split in splits:
        images_dir = RAW_DATA_DIR / split / "images"
        labels_dir = RAW_DATA_DIR / split / "labels"
        
        if images_dir.exists():
            image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
            stats[f"{split}_images"] = len(image_files)
            
            if labels_dir.exists():
                for label_file in labels_dir.glob("*.txt"):
                    with open(label_file, "r") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                stats[f"{split}_instances"][class_id] += 1
    
    stats["total_instances"] = (
        sum(stats["train_instances"].values()) +
        sum(stats["valid_instances"].values()) +
        sum(stats["test_instances"].values())
    )
    
    return stats


def print_dataset_summary():
    """Print a formatted summary of the dataset."""
    stats = count_dataset_statistics()
    
    print("\n" + "=" * 60)
    print("DATASET SUMMARY (6-Class Candlestick Patterns)")
    print("=" * 60)
    print(f"\nImages:")
    print(f"  Training:   {stats['train_images']:,}")
    print(f"  Validation: {stats['valid_images']:,}")
    print(f"  Test:       {stats['test_images']:,}")
    print(f"  Total:      {stats['train_images'] + stats['valid_images'] + stats['test_images']:,}")
    
    print(f"\nAnnotated Instances: {stats['total_instances']:,}")
    
    print("\nClass Distribution (Training Set):")
    print("-" * 45)
    
    if stats["train_instances"]:
        sorted_counts = stats["train_instances"].most_common()
        max_count = max(stats["train_instances"].values()) if stats["train_instances"] else 1
        
        for class_id, count in sorted_counts:
            class_name = CLASS_ID_TO_NAME.get(class_id, f"Unknown_{class_id}")
            bar_length = int(count / max_count * 20)
            bar = "█" * bar_length
            print(f"  {class_name:<20} {count:>5} {bar}")
    else:
        print("  No training data found.")
    
    print("=" * 60)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_yaml_config() -> Optional[Dict]:
    """Load the dataset YAML configuration file."""
    if not DATASET_YAML.exists():
        print(f"ERROR: data.yaml not found at {DATASET_YAML}")
        return None
    
    with open(DATASET_YAML, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def load_image_with_boxes(image_path: Path, labels_dir: Path) -> Tuple[np.ndarray, List[Dict]]:
    """Load an image and its corresponding bounding box annotations."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    label_file = labels_dir / f"{image_path.stem}.txt"
    annotations = []
    
    if label_file.exists():
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center = float(parts[1]), float(parts[2])
                    box_width, box_height = float(parts[3]), float(parts[4])
                    
                    x1 = int((x_center - box_width / 2) * width)
                    y1 = int((y_center - box_height / 2) * height)
                    x2 = int((x_center + box_width / 2) * width)
                    y2 = int((y_center + box_height / 2) * height)
                    
                    annotations.append({
                        "class_id": class_id,
                        "class_name": CLASS_ID_TO_NAME.get(class_id, "Unknown"),
                        "bbox": [x_center, y_center, box_width, box_height],
                        "bbox_pixels": [x1, y1, x2, y2]
                    })
    
    return image, annotations


def get_sample_images(split: str = "train", n_samples: int = 5) -> List[Path]:
    """Get paths to sample images from the dataset."""
    images_dir = RAW_DATA_DIR / split / "images"
    
    if not images_dir.exists():
        return []
    
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    
    if len(image_files) <= n_samples:
        return image_files
    
    indices = np.random.choice(len(image_files), n_samples, replace=False)
    return [image_files[i] for i in indices]


# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def create_data_yaml(output_path: Path = None) -> Path:
    """Create or update the data.yaml file with correct absolute paths."""
    if output_path is None:
        output_path = DATASET_YAML
    
    yaml_content = {
        "path": str(RAW_DATA_DIR.absolute()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES
    }
    
    with open(output_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)
    
    print(f"Created data.yaml at: {output_path}")
    return output_path


def get_class_weights() -> Dict[int, float]:
    """Calculate class weights for handling imbalanced data."""
    stats = count_dataset_statistics()
    total_counts = stats["train_instances"]
    
    if not total_counts:
        return {}
    
    total = sum(total_counts.values())
    n_classes = len(total_counts)
    
    weights = {}
    for class_id, count in total_counts.items():
        weights[class_id] = total / (n_classes * count) if count > 0 else 1.0
    
    return weights


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing data_utils module...")
    print()
    verify_dataset_structure()
    print()
    print_dataset_summary()
