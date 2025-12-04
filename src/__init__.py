"""
Candlestick Pattern Detection - Source Module
==============================================

This package contains reusable modules for the candlestick pattern
detection project. Organizing code into modules promotes:

    1. Reusability - Functions can be imported across notebooks
    2. Maintainability - Changes in one place propagate everywhere
    3. Testing - Modules can be tested independently
    4. Clarity - Separates concerns (data, model, visualization)

Module Overview:
    config.py           - Centralized configuration and constants
    data_utils.py       - Dataset loading and preprocessing utilities
    alpha_vantage_api.py - Stock market data fetching
    chart_generator.py  - Candlestick chart image generation
    pattern_detector.py - YOLO model wrapper for inference
    visualization.py    - Result plotting and visualization

Usage:
    from src.config import MODEL_CONFIG, CLASS_NAMES
    from src.alpha_vantage_api import AlphaVantageAPI
    from src.pattern_detector import PatternDetector
"""

# Version info
__version__ = "1.0.0"
__author__ = "Krunal && Axay"

# Import key classes for convenient access
# This allows: from src import PatternDetector
# Instead of: from src.pattern_detector import PatternDetector

from src.config import (
    CLASS_NAMES,
    NUM_CLASSES,
    MODEL_CONFIG,
    BULLISH_PATTERNS,
    BEARISH_PATTERNS
)