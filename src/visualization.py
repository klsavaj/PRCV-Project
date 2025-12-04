"""
Visualization Module
====================

This module provides visualization utilities for:
    1. Training results (loss curves, metrics)
    2. Dataset analysis (class distributions, sample images)
    3. Detection results (annotated charts, comparison views)
    4. Evaluation metrics (confusion matrices, PR curves)

All plots follow a consistent style for professional presentation
in reports and presentations.

Design Philosophy:
    - Consistent color palette across all visualizations
    - Publication-quality figures (high DPI, clean labels)
    - Figures saved automatically to outputs/figures/
    - Functions return figure objects for further customization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from src.config import (
    FIGURES_DIR,
    CLASS_NAMES,
    CLASS_ID_TO_NAME,
    COLORS
)

# Set style defaults
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10


# =============================================================================
# TRAINING VISUALIZATION
# =============================================================================

def plot_training_history(
    results_csv: Path,
    save_path: Path = None
) -> plt.Figure:
    """
    Plot training metrics from YOLOv8 results.csv file.
    
    Creates a 2x3 grid showing:
        - Box loss (train and val)
        - Classification loss (train and val)
        - DFL loss (train and val)
        - Precision
        - Recall
        - mAP metrics
    
    Args:
        results_csv: Path to YOLO training results.csv
        save_path: Where to save figure (optional)
        
    Returns:
        matplotlib Figure object
    """
    # Load results
    df = pd.read_csv(results_csv)
    df.columns = df.columns.str.strip()  # Clean column names
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")
    
    # Row 1: Losses
    if 'train/box_loss' in df.columns:
        axes[0, 0].plot(df['train/box_loss'], label='Train', color='#2563eb')
        axes[0, 0].plot(df['val/box_loss'], label='Val', color='#dc2626')
        axes[0, 0].set_title('Box Loss (CIoU)')
        axes[0, 0].legend()
        axes[0, 0].set_xlabel('Epoch')
    
    if 'train/cls_loss' in df.columns:
        axes[0, 1].plot(df['train/cls_loss'], label='Train', color='#2563eb')
        axes[0, 1].plot(df['val/cls_loss'], label='Val', color='#dc2626')
        axes[0, 1].set_title('Classification Loss')
        axes[0, 1].legend()
        axes[0, 1].set_xlabel('Epoch')
    
    if 'train/dfl_loss' in df.columns:
        axes[0, 2].plot(df['train/dfl_loss'], label='Train', color='#2563eb')
        axes[0, 2].plot(df['val/dfl_loss'], label='Val', color='#dc2626')
        axes[0, 2].set_title('Distribution Focal Loss')
        axes[0, 2].legend()
        axes[0, 2].set_xlabel('Epoch')
    
    # Row 2: Metrics
    if 'metrics/precision(B)' in df.columns:
        axes[1, 0].plot(df['metrics/precision(B)'], color='#059669')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylim(0, 1)
    
    if 'metrics/recall(B)' in df.columns:
        axes[1, 1].plot(df['metrics/recall(B)'], color='#7c3aed')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylim(0, 1)
    
    if 'metrics/mAP50(B)' in df.columns:
        axes[1, 2].plot(df['metrics/mAP50(B)'], label='mAP@0.5', color='#2563eb')
        if 'metrics/mAP50-95(B)' in df.columns:
            axes[1, 2].plot(df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', color='#dc2626')
        axes[1, 2].set_title('Mean Average Precision')
        axes[1, 2].legend()
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        save_path = FIGURES_DIR / "training_history.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# DATASET VISUALIZATION
# =============================================================================

def plot_class_distribution(
    class_counts: Dict[int, int],
    title: str = "Class Distribution",
    save_path: Path = None
) -> plt.Figure:
    """
    Plot bar chart of class distribution.
    
    Args:
        class_counts: Dictionary mapping class_id to count
        title: Plot title
        save_path: Where to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Sort by class ID
    sorted_ids = sorted(class_counts.keys())
    counts = [class_counts[i] for i in sorted_ids]
    names = [CLASS_ID_TO_NAME.get(i, f"Class_{i}")[:20] for i in sorted_ids]
    
    # Create bar chart
    bars = ax.bar(range(len(counts)), counts, color='#3b82f6', edgecolor='white')
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(counts) * 0.01,
            f'{count}',
            ha='center',
            va='bottom',
            fontsize=8
        )
    
    # Styling
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax.set_xlabel('Pattern Class')
    ax.set_ylabel('Number of Instances')
    ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    if save_path is None:
        save_path = FIGURES_DIR / "class_distribution.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


def plot_sample_images(
    images: List[np.ndarray],
    titles: List[str] = None,
    save_path: Path = None,
    ncols: int = 3
) -> plt.Figure:
    """
    Display a grid of sample images.
    
    Args:
        images: List of image arrays (RGB format)
        titles: Optional titles for each image
        save_path: Where to save figure
        ncols: Number of columns in grid
        
    Returns:
        matplotlib Figure object
    """
    n_images = len(images)
    nrows = (n_images + ncols - 1) // ncols
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for idx, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        ax.axis('off')
        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save
    if save_path is None:
        save_path = FIGURES_DIR / "sample_images.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# DETECTION VISUALIZATION
# =============================================================================

def plot_detection_results(
    original: np.ndarray,
    annotated: np.ndarray,
    detections: List,
    symbol: str = "",
    save_path: Path = None
) -> plt.Figure:
    """
    Create side-by-side comparison of original and detected patterns.
    
    Args:
        original: Original chart image (RGB)
        annotated: Annotated chart with bounding boxes (RGB)
        detections: List of Detection objects
        symbol: Stock symbol for title
        save_path: Where to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original chart
    axes[0].imshow(original)
    axes[0].set_title(f"Original Chart - {symbol}", fontweight='bold')
    axes[0].axis('off')
    
    # Annotated chart
    axes[1].imshow(annotated)
    axes[1].set_title(f"Detected Patterns ({len(detections)} found)", fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    # Save
    if save_path is None:
        save_path = FIGURES_DIR / f"detection_{symbol}.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


def create_detection_summary(
    detections: List,
    save_path: Path = None
) -> plt.Figure:
    """
    Create a visual summary card of detected patterns.
    
    Args:
        detections: List of Detection objects
        save_path: Where to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, max(3, len(detections) * 0.6)))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, "DETECTED PATTERNS", 
            transform=ax.transAxes, fontsize=14, fontweight='bold',
            ha='center', va='top')
    
    if not detections:
        ax.text(0.5, 0.5, "No patterns detected", 
                transform=ax.transAxes, fontsize=12, ha='center', va='center',
                color='gray')
    else:
        y_pos = 0.85
        for det in detections:
            # Pattern name
            ax.text(0.1, y_pos, det.pattern, 
                    transform=ax.transAxes, fontsize=11, fontweight='bold')
            
            # Confidence bar
            bar_width = det.confidence * 0.4
            ax.barh(y_pos - 0.03, bar_width, height=0.04, left=0.45,
                   color=det.color, transform=ax.transAxes)
            
            # Confidence text
            ax.text(0.87, y_pos - 0.02, f"{det.confidence:.1%}",
                   transform=ax.transAxes, fontsize=10)
            
            # Sentiment
            emoji = "🟢" if det.sentiment == "BULLISH" else \
                   "🔴" if det.sentiment == "BEARISH" else "⚪"
            ax.text(0.92, y_pos, emoji, transform=ax.transAxes, fontsize=10)
            
            y_pos -= 0.12
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# EVALUATION VISUALIZATION
# =============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = None,
    save_path: Path = None
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        class_names: List of class names for labels
        save_path: Where to save figure
        
    Returns:
        matplotlib Figure object
    """
    if class_names is None:
        class_names = [f"Class_{i}" for i in range(len(cm))]
    
    # Truncate class names for readability
    short_names = [name[:15] for name in class_names]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=short_names,
        yticklabels=short_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path is None:
        save_path = FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


def plot_pr_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    class_name: str = "All Classes",
    save_path: Path = None
) -> plt.Figure:
    """
    Plot Precision-Recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        class_name: Name for legend
        save_path: Where to save figure
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, color='#2563eb', linewidth=2, label=class_name)
    ax.fill_between(recall, precision, alpha=0.3, color='#2563eb')
    
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = FIGURES_DIR / "pr_curve.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    return fig


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Visualization Module Test")
    print("=" * 40)
    
    # Test class distribution plot with dummy data
    dummy_counts = {i: np.random.randint(50, 500) for i in range(10)}
    fig = plot_class_distribution(dummy_counts, "Test Distribution")
    plt.close(fig)
    
    print("\nVisualization functions ready!")
    print(f"Figures will be saved to: {FIGURES_DIR}")