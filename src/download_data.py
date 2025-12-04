import os
from roboflow import Roboflow
from dotenv import load_dotenv
from pathlib import Path
import sys

# Add project root to path to import config
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import RAW_DATA_DIR

def download_dataset():
    """
    Downloads the candlestick pattern recognition dataset from Roboflow.
    Requires ROBOFLOW_API_KEY to be set in .env file.
    """
    load_dotenv()
    
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        print("❌ Error: ROBOFLOW_API_KEY not found in .env file.")
        print("Please add your Roboflow API key to the .env file:")
        print("ROBOFLOW_API_KEY=your_key_here")
        return False

    try:
        print(f"Downloading dataset to {RAW_DATA_DIR}...")
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("ranyas-workspace").project("candlestick-pattern-recognition")
        version = project.version(2)
        dataset = version.download("yolov8", location=str(RAW_DATA_DIR))
        print("✅ Dataset downloaded successfully.")
        return True
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return False

if __name__ == "__main__":
    download_dataset()
