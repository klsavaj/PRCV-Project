"""
Pattern Detector Module
=======================

This module wraps the YOLOv8 model for candlestick pattern detection.
It provides a clean interface for running inference on chart images
and processing the detection results.

The detector handles:
    1. Loading the trained model weights
    2. Running inference with configurable confidence thresholds
    3. Parsing detection results into a usable format
    4. Adding sentiment labels (bullish/bearish/neutral)
    5. Generating annotated output images

YOLOv8 Output Format:
    Each detection contains:
    - Bounding box coordinates (xyxy format)
    - Confidence score (0-1)
    - Class ID
    
We augment this with class names and trading sentiments.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass

from ultralytics import YOLO

from src.config import (
    WEIGHTS_DIR,
    CLASS_NAMES,
    CLASS_ID_TO_NAME,
    INFERENCE_CONFIG,
    BULLISH_PATTERNS,
    BEARISH_PATTERNS,
    COLORS
)


@dataclass
class Detection:
    """
    Data class representing a single pattern detection.
    
    Using dataclass provides:
    - Clean attribute access (det.pattern instead of det['pattern'])
    - Automatic __repr__ for debugging
    - Type hints for IDE support
    """
    pattern: str            # Pattern name
    class_id: int          # Numeric class ID
    confidence: float      # Detection confidence (0-1)
    sentiment: str         # BULLISH, BEARISH, or NEUTRAL
    color: str             # Hex color for visualization
    bbox: List[float]      # Bounding box [x1, y1, x2, y2]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pattern": self.pattern,
            "class_id": self.class_id,
            "confidence": round(self.confidence * 100, 1),
            "sentiment": self.sentiment,
            "color": self.color,
            "bbox": self.bbox
        }


class PatternDetector:
    """
    YOLOv8-based candlestick pattern detector.
    
    This class loads a trained YOLOv8 model and provides methods for
    detecting patterns in candlestick chart images.
    
    Attributes:
        model: The loaded YOLO model
        confidence_threshold: Minimum confidence for valid detections
        class_names: List of pattern names
        
    Usage:
        detector = PatternDetector("models/weights/best.pt")
        
        # Detect patterns in an image
        detections, annotated = detector.detect("chart.png")
        
        for det in detections:
            print(f"{det.pattern}: {det.confidence:.1%} ({det.sentiment})")
    """
    
    def __init__(
        self,
        model_path: Union[str, Path] = None,
        confidence_threshold: float = None
    ):
        """
        Initialize the pattern detector.
        
        Args:
            model_path: Path to trained .pt weights file.
                       Defaults to models/weights/best.pt
            confidence_threshold: Minimum confidence for detections.
                                 Defaults to value in config.py
        """
        # Set model path
        if model_path is None:
            model_path = WEIGHTS_DIR / "best.pt"
        self.model_path = Path(model_path)
        
        # Set confidence threshold
        self.confidence_threshold = (
            confidence_threshold or
            INFERENCE_CONFIG["confidence_threshold"]
        )
        
        # Load model
        self.model = self._load_model()
        self.class_names = CLASS_NAMES
    
    def _load_model(self) -> YOLO:
        """
        Load the YOLO model from weights file.
        
        Returns:
            Loaded YOLO model
            
        Raises:
            FileNotFoundError: If model weights don't exist
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found: {self.model_path}\n"
                f"Make sure you've trained the model or downloaded weights."
            )
        
        model = YOLO(str(self.model_path))
        print(f"Model loaded: {self.model_path.name}")
        return model
    
    def _get_sentiment(self, pattern_name: str) -> Tuple[str, str]:
        """
        Determine the market sentiment for a pattern.
        
        Args:
            pattern_name: Name of the detected pattern
            
        Returns:
            tuple: (sentiment_label, hex_color)
        """
        if pattern_name in BULLISH_PATTERNS:
            return "BULLISH", COLORS["bullish"]
        elif pattern_name in BEARISH_PATTERNS:
            return "BEARISH", COLORS["bearish"]
        else:
            return "NEUTRAL", COLORS["neutral"]
    
    def detect(
        self,
        image: Union[str, Path, np.ndarray],
        confidence_threshold: float = None,
        return_annotated: bool = True
    ) -> Tuple[List[Detection], Optional[np.ndarray]]:
        """
        Detect candlestick patterns in an image.
        
        Args:
            image: Image path or numpy array (BGR or RGB)
            confidence_threshold: Override default threshold for this call
            return_annotated: Whether to return annotated image
            
        Returns:
            tuple: (list of Detection objects, annotated image or None)
        """
        conf = confidence_threshold or self.confidence_threshold
        
        # Run inference
        results = self.model(
            image,
            conf=conf,
            iou=INFERENCE_CONFIG["iou_threshold"],
            max_det=INFERENCE_CONFIG["max_detections"],
            verbose=False
        )
        
        # Parse detections
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                
                # Get pattern name
                pattern_name = CLASS_ID_TO_NAME.get(
                    class_id,
                    f"Unknown_{class_id}"
                )
                
                # Get sentiment
                sentiment, color = self._get_sentiment(pattern_name)
                
                # Create detection object
                detection = Detection(
                    pattern=pattern_name,
                    class_id=class_id,
                    confidence=confidence,
                    sentiment=sentiment,
                    color=color,
                    bbox=bbox
                )
                detections.append(detection)
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        # Get annotated image if requested
        annotated = None
        if return_annotated and len(results) > 0:
            annotated = results[0].plot()
            # Convert BGR to RGB
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        
        return detections, annotated
    
    def detect_from_dataframe(
        self,
        df,
        symbol: str = "",
        num_candles: int = 50,
        confidence_threshold: float = None
    ) -> Tuple[List[Detection], np.ndarray, np.ndarray]:
        """
        Generate chart from DataFrame and detect patterns.
        
        This is a convenience method that combines chart generation
        and pattern detection in one call.
        
        Args:
            df: OHLCV DataFrame
            symbol: Stock symbol for chart title
            num_candles: Number of candles to display
            confidence_threshold: Detection threshold
            
        Returns:
            tuple: (detections, original_chart, annotated_chart)
        """
        from src.chart_generator import ChartGenerator
        
        # Generate chart as numpy array
        generator = ChartGenerator()
        chart_image = generator.to_numpy(df, symbol, num_candles)
        
        # Detect patterns
        detections, annotated = self.detect(
            chart_image,
            confidence_threshold,
            return_annotated=True
        )
        
        return detections, chart_image, annotated
    
    def print_detections(self, detections: List[Detection]):
        """
        Print detection results in a formatted manner.
        
        Args:
            detections: List of Detection objects
        """
        print("\n" + "=" * 50)
        print(f"DETECTED PATTERNS: {len(detections)}")
        print("=" * 50)
        
        if not detections:
            print("  No patterns detected.")
            print("  Try lowering the confidence threshold.")
        else:
            for i, det in enumerate(detections, 1):
                emoji = "🟢" if det.sentiment == "BULLISH" else \
                       "🔴" if det.sentiment == "BEARISH" else "⚪"
                
                print(f"\n  {i}. {det.pattern}")
                print(f"     Confidence: {det.confidence:.1%}")
                print(f"     Signal: {emoji} {det.sentiment}")
        
        print("\n" + "=" * 50)


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def detect_patterns(
    image_path: Union[str, Path],
    model_path: Union[str, Path] = None,
    confidence: float = 0.25
) -> List[Dict]:
    """
    Simple function to detect patterns without instantiating class.
    
    Args:
        image_path: Path to chart image
        model_path: Path to model weights (optional)
        confidence: Detection threshold
        
    Returns:
        List of detection dictionaries
    """
    detector = PatternDetector(model_path, confidence)
    detections, _ = detector.detect(image_path, return_annotated=False)
    return [d.to_dict() for d in detections]


# =============================================================================
# DEMO / TESTING
# =============================================================================

if __name__ == "__main__":
    print("Pattern Detector Module Test")
    print("=" * 40)
    
    # Check if model exists
    default_path = WEIGHTS_DIR / "best.pt"
    
    if default_path.exists():
        detector = PatternDetector()
        print(f"\nModel loaded successfully!")
        print(f"Confidence threshold: {detector.confidence_threshold}")
        print(f"Number of classes: {len(detector.class_names)}")
    else:
        print(f"\nModel not found at: {default_path}")
        print("Train the model first using notebook 03_model_training.ipynb")