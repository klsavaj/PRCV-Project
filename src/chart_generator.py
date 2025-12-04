"""
Candlestick Chart Generator
===========================

This module generates candlestick chart images from OHLCV data.
The generated images serve as input to the YOLOv8 model for pattern detection.

We use mplfinance, a matplotlib extension specifically designed for financial
charting. It handles the complexity of drawing candlesticks with proper
coloring (green for up days, red for down days) and includes volume bars.

Key Considerations:
    1. Chart appearance should match the training data style
    2. Resolution should be consistent (640x640 for YOLO)
    3. Charts should be clean without excessive annotations
    4. Volume subplot provides additional context

Reference: https://github.com/matplotlib/mplfinance
"""

import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from pathlib import Path
from io import BytesIO
import base64
from datetime import datetime
from typing import Optional, Tuple, Union

# Use non-interactive backend for server environments
import matplotlib
matplotlib.use('Agg')

from src.config import (
    CHART_CONFIG,
    COLORS,
    PROCESSED_DATA_DIR
)


class ChartGenerator:
    """
    Generator for candlestick chart images.
    
    Creates publication-quality candlestick charts suitable for:
    1. Model inference (pattern detection)
    2. Visualization in reports
    3. Web application display
    
    The charts match the style of the training dataset to ensure
    the model generalizes well to generated charts.
    
    Usage:
        generator = ChartGenerator()
        
        # Save to file
        path = generator.save_chart(df, "AAPL", num_candles=50)
        
        # Get as base64 (for web display)
        b64 = generator.to_base64(df, "AAPL")
    """
    
    def __init__(
        self,
        figsize: Tuple[int, int] = None,
        dpi: int = None,
        style: str = "yahoo"
    ):
        """
        Initialize the chart generator.
        
        Args:
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for output resolution
            style: mplfinance style preset ('yahoo', 'charles', 'mike', etc.)
        """
        self.figsize = figsize or CHART_CONFIG["figure_size"]
        self.dpi = dpi or CHART_CONFIG["dpi"]
        
        # Create custom market colors
        # Green for bullish (close > open), Red for bearish (close < open)
        self.market_colors = mpf.make_marketcolors(
            up=COLORS["bullish"],
            down=COLORS["bearish"],
            edge="inherit",
            wick="inherit",
            volume="in"
        )
        
        # Create style with our colors
        self.style = mpf.make_mpf_style(
            marketcolors=self.market_colors,
            gridstyle="",
            y_on_right=False,
            facecolor="white"
        )
    
    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and prepare DataFrame for charting.
        
        mplfinance expects specific column names and a DatetimeIndex.
        This method ensures the data is in the correct format.
        
        Args:
            df: Input DataFrame with OHLCV data
            
        Returns:
            DataFrame ready for mplfinance
            
        Raises:
            ValueError: If required columns are missing
        """
        required_cols = ["Open", "High", "Low", "Close", "Volume"]
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Check for required columns (case-insensitive)
        df.columns = [c.capitalize() for c in df.columns]
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Ensure numeric types
        for col in required_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Remove any NaN rows
        df = df.dropna()
        
        return df
    
    def generate(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        num_candles: int = None,
        show_volume: bool = True,
        title: str = None
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Generate a candlestick chart figure.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol for title
            num_candles: Number of candles to display (from end of data)
            show_volume: Whether to include volume subplot
            title: Custom title (overrides symbol-based title)
            
        Returns:
            tuple: (matplotlib Figure, Axes)
        """
        df = self._validate_dataframe(df)
        
        # Limit to specified number of candles
        num_candles = num_candles or CHART_CONFIG["default_candles"]
        if len(df) > num_candles:
            df = df.tail(num_candles)
        
        # Set title
        if title is None:
            title = f"{symbol} - {num_candles} Day Chart" if symbol else ""
        
        # Generate chart
        fig, axes = mpf.plot(
            df,
            type="candle",
            style=self.style,
            title=title,
            ylabel="Price ($)",
            volume=show_volume,
            figsize=self.figsize,
            returnfig=True
        )
        
        return fig, axes
    
    def save_chart(
        self,
        df: pd.DataFrame,
        symbol: str,
        output_path: Path = None,
        num_candles: int = None,
        show_volume: bool = True
    ) -> Path:
        """
        Generate and save a candlestick chart to file.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol (used for filename and title)
            output_path: Where to save (default: data/processed/)
            num_candles: Number of candles to display
            show_volume: Whether to include volume subplot
            
        Returns:
            Path to saved image file
        """
        fig, _ = self.generate(df, symbol, num_candles, show_volume)
        
        # Create output path
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{symbol}_{timestamp}.png"
            output_path = PROCESSED_DATA_DIR / filename
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save figure
        fig.savefig(
            output_path,
            dpi=self.dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none"
        )
        plt.close(fig)
        
        print(f"Chart saved: {output_path}")
        return output_path
    
    def to_base64(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        num_candles: int = None,
        show_volume: bool = True
    ) -> str:
        """
        Generate chart and return as base64-encoded string.
        
        This is useful for embedding charts in web applications
        without saving temporary files.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            num_candles: Number of candles to display
            show_volume: Whether to include volume subplot
            
        Returns:
            Base64-encoded PNG image string
        """
        fig, _ = self.generate(df, symbol, num_candles, show_volume)
        
        # Save to BytesIO buffer
        buffer = BytesIO()
        fig.savefig(
            buffer,
            format="png",
            dpi=self.dpi,
            bbox_inches="tight",
            facecolor="white"
        )
        buffer.seek(0)
        plt.close(fig)
        
        # Encode to base64
        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")
        buffer.close()
        
        return img_base64
    
    def to_numpy(
        self,
        df: pd.DataFrame,
        symbol: str = "",
        num_candles: int = None,
        show_volume: bool = True
    ) -> np.ndarray:
        """
        Generate chart and return as numpy array.
        
        Useful for direct input to image processing pipelines.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Stock symbol
            num_candles: Number of candles to display
            show_volume: Whether to include volume subplot
            
        Returns:
            NumPy array of shape (H, W, 3) in RGB format
        """
        fig, _ = self.generate(df, symbol, num_candles, show_volume)
        
        # Render to array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        return img_array


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def create_chart(
    df: pd.DataFrame,
    symbol: str,
    output_path: Path = None,
    num_candles: int = 50
) -> Path:
    """
    Simple function to create a candlestick chart without instantiating class.
    
    Args:
        df: OHLCV DataFrame
        symbol: Stock ticker
        output_path: Where to save (optional)
        num_candles: Number of candles
        
    Returns:
        Path to saved chart
    """
    generator = ChartGenerator()
    return generator.save_chart(df, symbol, output_path, num_candles)


# =============================================================================
# DEMO / TESTING
# =============================================================================

if __name__ == "__main__":
    print("Chart Generator Module Test")
    print("=" * 40)
    
    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=100, freq="D")
    
    # Generate realistic price data
    price = 150.0
    data = []
    for _ in range(100):
        open_price = price
        change = np.random.randn() * 2
        close_price = open_price + change
        high_price = max(open_price, close_price) + abs(np.random.randn())
        low_price = min(open_price, close_price) - abs(np.random.randn())
        volume = np.random.randint(1_000_000, 10_000_000)
        data.append([open_price, high_price, low_price, close_price, volume])
        price = close_price
    
    df = pd.DataFrame(
        data,
        index=dates,
        columns=["Open", "High", "Low", "Close", "Volume"]
    )
    
    print(f"Sample data shape: {df.shape}")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Generate chart
    generator = ChartGenerator()
    
    # Save to file
    output_path = PROCESSED_DATA_DIR / "test_chart.png"
    saved_path = generator.save_chart(df, "TEST", output_path, num_candles=50)
    print(f"\nChart saved to: {saved_path}")
    
    # Test base64 generation
    b64 = generator.to_base64(df, "TEST", num_candles=30)
    print(f"Base64 length: {len(b64)} characters")