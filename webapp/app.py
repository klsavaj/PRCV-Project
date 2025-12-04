"""
Flask Backend API for Candlestick Pattern Detection
====================================================
"""

import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import base64
from io import BytesIO
import cv2
from PIL import Image
import tempfile

# Project imports
from src.config import (
    WEIGHTS_DIR, CLASS_NAMES, CLASS_ID_TO_NAME,
    BULLISH_PATTERNS, BEARISH_PATTERNS, COLORS
)
from src.alpha_vantage_api import AlphaVantageAPI
from src.chart_generator import ChartGenerator

# Import YOLO
from ultralytics import YOLO

# =============================================================================
# FLASK APP INITIALIZATION
# =============================================================================

# CHANGED: Configure Flask to serve static files from current directory
app = Flask(__name__, static_url_path='', static_folder='.')

# Enable CORS
CORS(app)


# =============================================================================
# GLOBAL VARIABLES
# =============================================================================

model = None
chart_generator = None


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_model():
    global model
    model_path = WEIGHTS_DIR / "best.pt"
    if model_path.exists():
        model = YOLO(str(model_path))
        print(f"[OK] Model loaded: {model_path}")
    else:
        print(f"[WARNING] Model not found: {model_path}")
        model = None

def initialize_components():
    global chart_generator
    print("\n" + "="*50)
    print("Initializing Candlestick Pattern Detection API")
    print("="*50)
    load_model()
    chart_generator = ChartGenerator()
    print("[OK] Chart generator initialized")
    print("="*50 + "\n")


# =============================================================================
# HELPER FUNCTIONS (Condensed for brevity - copy from original if needed)
# =============================================================================

def get_sentiment(pattern_name):
    if pattern_name in BULLISH_PATTERNS: return "BULLISH", COLORS["bullish"]
    elif pattern_name in BEARISH_PATTERNS: return "BEARISH", COLORS["bearish"]
    return "NEUTRAL", COLORS["neutral"]

def image_to_base64(image_array):
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        pil_image = Image.fromarray(image_array.astype('uint8'))
    else:
        pil_image = Image.fromarray(image_array)
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def detect_patterns(image_path, confidence_threshold=0.25):
    global model
    if model is None: return [], None
    results = model(str(image_path), conf=confidence_threshold, verbose=False)
    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            pattern_name = CLASS_ID_TO_NAME.get(class_id, f"Pattern_{class_id}")
            sentiment, color = get_sentiment(pattern_name)
            detections.append({
                "pattern": pattern_name,
                "confidence": round(confidence * 100, 1),
                "sentiment": sentiment,
                "color": color,
                "bbox": bbox
            })
    detections.sort(key=lambda x: x["confidence"], reverse=True)
    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return detections, annotated_rgb


# =============================================================================
# API ROUTES
# =============================================================================

# ADDED: Root route to serve index.html
@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "message": "Candlestick Pattern Detection API"
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    data = request.json
    symbol = data.get("symbol", "AAPL").upper().strip()
    api_key = data.get("apiKey", "")
    num_candles = data.get("numCandles", 50)
    confidence = data.get("confidence", 0.25)
    
    if not api_key: return jsonify({"error": "API key is required"}), 400
    if model is None: return jsonify({"error": "Model not loaded."}), 500
    
    try:
        api = AlphaVantageAPI(api_key=api_key)
        df, error = api.get_daily_data(symbol)
        if error: return jsonify({"error": error}), 400
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            chart_path = tmp.name
            chart_generator.save_chart(df, symbol, Path(chart_path), num_candles)
        
        original_chart = chart_generator.to_base64(df, symbol, num_candles)
        detections, annotated_img = detect_patterns(chart_path, confidence)
        annotated_chart = image_to_base64(annotated_img) if annotated_img is not None else ""
        os.unlink(chart_path)
        
        recent_df = df.tail(5).round(2)
        recent_data = {
            date.strftime("%Y-%m-%d"): {
                "Open": row["Open"], "High": row["High"],
                "Low": row["Low"], "Close": row["Close"], "Volume": int(row["Volume"])
            } for date, row in recent_df.iterrows()
        }
        
        return jsonify({
            "symbol": symbol,
            "originalChart": original_chart,
            "annotatedChart": annotated_chart,
            "detections": detections,
            "recentData": recent_data
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_symbol():
    query = request.args.get("q", "")
    api_key = request.args.get("apiKey", "")
    if not query or not api_key: return jsonify({"results": []})
    try:
        api = AlphaVantageAPI(api_key=api_key)
        results = api.search_symbol(query)
        return jsonify({"results": results[:5]})
    except Exception as e:
        return jsonify({"results": [], "error": str(e)})

@app.route('/api/classes', methods=['GET'])
def get_classes():
    classes = []
    for i, name in enumerate(CLASS_NAMES):
        sentiment, color = get_sentiment(name)
        classes.append({"id": i, "name": name, "sentiment": sentiment, "color": color})
    return jsonify({"classes": classes})


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    initialize_components()
    print("Starting Flask server...")
    print("API available at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)