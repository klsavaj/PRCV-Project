"""
Project Configuration
=====================

Centralized configuration for Candlestick Pattern Detection project.
Updated for 6-class dataset from Roboflow (duokan/ahihi-m5uvy).

Classes: Dragonfly Doji, Gravestone Doji, Hammer, Hanging Man, Marubozu, Spinning Top
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model artifacts
MODELS_DIR = PROJECT_ROOT / "models"
WEIGHTS_DIR = MODELS_DIR / "weights"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESULTS_DIR = OUTPUTS_DIR / "results"
LOGS_DIR = OUTPUTS_DIR / "logs"

# Dataset YAML
DATASET_YAML = RAW_DATA_DIR / "data.yaml"


# =============================================================================
# CANDLESTICK PATTERN CLASSES (6 Classes - New Dataset)
# =============================================================================

CLASS_NAMES = [
    "Dragonfly Doji",    # 0 - Bullish reversal
    "Gravestone Doji",   # 1 - Bearish reversal
    "Hammer",            # 2 - Bullish reversal
    "Hanging Man",       # 3 - Bearish reversal
    "Marubozu",          # 4 - Trend continuation
    "Spinning Top"       # 5 - Indecision/Neutral
]

NUM_CLASSES = len(CLASS_NAMES)  # 6 classes

# Mappings
CLASS_ID_TO_NAME = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_NAME_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


# =============================================================================
# PATTERN SENTIMENT CLASSIFICATION
# =============================================================================

BULLISH_PATTERNS = {
    "Hammer",
    "Dragonfly Doji"
}

BEARISH_PATTERNS = {
    "Hanging Man",
    "Gravestone Doji"
}

# Marubozu and Spinning Top are NEUTRAL (context-dependent)


# =============================================================================
# YOLOV8 TRAINING CONFIGURATION
# =============================================================================

MODEL_CONFIG = {
    "architecture": "yolov8s.pt",  # Small model for better accuracy
    "epochs": 50,
    "patience": 15,
    "batch_size": 16,
    "img_size": 640,
    "optimizer": "AdamW",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "fliplr": 0.5,
    "mosaic": 1.0
}


# =============================================================================
# ALPHA VANTAGE API CONFIGURATION
# =============================================================================

ALPHA_VANTAGE_API_KEY = os.environ.get("ALPHA_VANTAGE_API_KEY", "")
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

API_RATE_LIMIT = {
    "calls_per_minute": 5,
    "calls_per_day": 25
}


# =============================================================================
# INFERENCE CONFIGURATION
# =============================================================================

INFERENCE_CONFIG = {
    "confidence_threshold": 0.25,
    "iou_threshold": 0.45,
    "max_detections": 100
}


# =============================================================================
# VISUALIZATION CONFIGURATION
# =============================================================================

CHART_CONFIG = {
    "default_candles": 50,
    "figure_size": (12, 8),
    "dpi": 100
}

COLORS = {
    "bullish": "#22c55e",
    "bearish": "#ef4444",
    "neutral": "#6b7280",
    "background": "#ffffff"
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_directories():
    """Create all project directories if they don't exist."""
    dirs = [
        RAW_DATA_DIR / "train" / "images",
        RAW_DATA_DIR / "train" / "labels",
        RAW_DATA_DIR / "valid" / "images",
        RAW_DATA_DIR / "valid" / "labels",
        RAW_DATA_DIR / "test" / "images",
        RAW_DATA_DIR / "test" / "labels",
        PROCESSED_DATA_DIR,
        WEIGHTS_DIR,
        FIGURES_DIR,
        RESULTS_DIR,
        LOGS_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_pattern_sentiment(pattern_name):
    """Get market sentiment for a candlestick pattern."""
    if pattern_name in BULLISH_PATTERNS:
        return "BULLISH", COLORS["bullish"]
    elif pattern_name in BEARISH_PATTERNS:
        return "BEARISH", COLORS["bearish"]
    else:
        return "NEUTRAL", COLORS["neutral"]


def print_config_summary():
    """Print configuration summary."""
    print("=" * 50)
    print("PROJECT CONFIGURATION SUMMARY")
    print("=" * 50)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Number of Classes: {NUM_CLASSES}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print(f"Model Architecture: {MODEL_CONFIG['architecture']}")
    print(f"Training Epochs: {MODEL_CONFIG['epochs']}")
    print(f"Batch Size: {MODEL_CONFIG['batch_size']}")
    print(f"Image Size: {MODEL_CONFIG['img_size']}")
    print(f"API Key Set: {'Yes' if ALPHA_VANTAGE_API_KEY else 'No'}")
    print("=" * 50)


# Create directories on import
create_directories()