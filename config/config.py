# Configuration Management Module for Flight Price Prediction Project.

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Directory paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, PLOTS_DIR, LOGS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    """Data-related configuration parameters."""
    
    # Dataset information
    dataset_name: str = "Flight_Price_Dataset_of_Bangladesh.csv"
    target_column: str = "Total Fare (BDT)"
    
    # Columns to drop (if present)
    # NOTE: Base Fare and Tax are dropped to prevent data leakage
    # (Total Fare = Base Fare + Tax, so including them would be cheating)
    columns_to_drop: List[str] = field(default_factory=lambda: [
        "Unnamed: 0", "Index", "Unnamed",
        "Source Name", "Destination Name", "Arrival Date & Time", 
        "Aircraft Type", "Stopovers",
        "Base Fare (BDT)", "Tax & Surcharge (BDT)"
    ])
    
    # Categorical columns for encoding
    categorical_columns: List[str] = field(default_factory=lambda: [
        "Airline", "Source", "Destination", "Class", "Booking Source", "Seasonality"
    ])
    
    # Numerical columns for scaling
    numerical_columns: List[str] = field(default_factory=lambda: [
        "Duration (hrs)", "Days Before Departure"
    ])
    
    # Date column for feature extraction
    date_column: str = "Departure Date & Time"
    
    # Missing value thresholds
    missing_threshold: float = 0.3  # Drop columns with >30% missing
    
    # City name normalization mapping
    city_normalization: Dict[str, str] = field(default_factory=lambda: {
        "Dacca": "Dhaka",
        "Chittagang": "Chittagong",
        "Chattogram": "Chittagong",
        "Sylhet City": "Sylhet",
    })


@dataclass
class ModelConfig:
    """Model training configuration parameters."""
    
    # Reproducibility
    random_state: int = 42
    
    # Train-test split
    test_size: float = 0.2
    
    # Cross-validation
    cv_folds: int = 5
    
    # Hyperparameter grids for tuning
    ridge_params: Dict[str, List[Any]] = field(default_factory=lambda: {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    })
    
    lasso_params: Dict[str, List[Any]] = field(default_factory=lambda: {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    })
    
    decision_tree_params: Dict[str, List[Any]] = field(default_factory=lambda: {
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    })
    
    random_forest_params: Dict[str, List[Any]] = field(default_factory=lambda: {
        "n_estimators": [50, 100],
        "max_depth": [10, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    })
    
    gradient_boosting_params: Dict[str, List[Any]] = field(default_factory=lambda: {
        "n_estimators": [100, 150], # number of trees built sequentially
        "max_depth": [4, 5], #How deep each tree is allowd to grow
        "learning_rate": [0.05, 0.1], #How much each new tree corrects previous errors
        "subsample": [0.8], #Each tree seees only 80% of the data randomly
        "min_samples_leaf": [3] #forces tres to generalize
    })


@dataclass
class VisualizationConfig:
    """Visualization configuration parameters."""
    
    # Plot style
    style: str = "seaborn-v0_8-whitegrid"
    
    # Figure sizes
    default_figsize: tuple = (10, 6)
    large_figsize: tuple = (14, 8)
    heatmap_figsize: tuple = (12, 10)
    
    # Color palette
    color_palette: str = "husl"
    
    # DPI for saved figures
    save_dpi: int = 300
    
    # Plot format
    save_format: str = "png"


@dataclass
class LoggingConfig:
    """Logging configuration parameters."""
    
    # Logging level
    level: str = "INFO"
    
    # Log format
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Date format
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Log file name
    log_file: str = "flight_price_prediction.log"


# Create default config instances
data_config = DataConfig()
model_config = ModelConfig()
viz_config = VisualizationConfig()
logging_config = LoggingConfig()
