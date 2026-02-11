"""
Feature Engineering Module.

This module handles all feature transformation, encoding, and scaling operations.
Implements Step 2B (Feature Engineering) of the project.

Industry Standard Justifications:
- Sklearn pipelines ensure consistent transformations across train/test
- Feature preprocessing is fitted on training data only to prevent data leakage
- All transformations are logged for reproducibility and debugging
- Transformers are saved for inference consistency
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pickle
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import data_config, model_config, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DateFeatureExtractor:
    """
    Extract temporal features from date column.
    
    Creates features like Month, Day, Weekday, and Season
    that can capture temporal patterns in flight pricing.
    """
    
    # Bangladesh season mapping (based on local climate)
    SEASON_MAP = {
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Summer', 4: 'Summer', 5: 'Summer',
        6: 'Monsoon', 7: 'Monsoon', 8: 'Monsoon', 9: 'Monsoon',
        10: 'Autumn', 11: 'Autumn'
    }
    
    def __init__(self, date_column: str = None):
        """
        Initialize DateFeatureExtractor.
        
        Args:
            date_column: Name of the date column
        """
        self.date_column = date_column or data_config.date_column
        logger.info(f"DateFeatureExtractor initialized for column: {self.date_column}")
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from the date column.
        
        Args:
            df: Input DataFrame with date column
        
        Returns:
            DataFrame with new temporal features
        """
        df = df.copy()
        
        if self.date_column not in df.columns:
            logger.warning(f"Date column '{self.date_column}' not found in DataFrame")
            return df
        
        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[self.date_column]):
            df[self.date_column] = pd.to_datetime(df[self.date_column], errors='coerce')
        
        # Extract temporal features
        df['Year'] = df[self.date_column].dt.year
        df['Month'] = df[self.date_column].dt.month
        df['Day'] = df[self.date_column].dt.day
        df['DayOfWeek'] = df[self.date_column].dt.dayofweek  # Monday=0, Sunday=6
        df['WeekOfYear'] = df[self.date_column].dt.isocalendar().week.astype(int)
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)
        
        # Cyclical encoding for periodic features (sin/cos)
        # This preserves the circular nature: Dec is close to Jan, Sunday close to Monday
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)
        df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
        
        # Extract season
        df['Season'] = df['Month'].map(self.SEASON_MAP)
        
        # Day name for interpretability (not used in model, but useful for analysis)
        df['DayName'] = df[self.date_column].dt.day_name()
        df['MonthName'] = df[self.date_column].dt.month_name()
        
        logger.info("Extracted temporal features: Year, Month, Day, DayOfWeek, "
                   "WeekOfYear, IsWeekend, Season, DayName, MonthName, "
                   "cyclical sin/cos for Month, DayOfWeek, Day")
        
        return df


class RouteFeatureExtractor:
    """
    Extract route-based features with smoothed target encoding.
    
    Uses Bayesian smoothed mean encoding for routes to avoid overfitting
    on rare routes while preserving target signal.
    """
    
    def __init__(self, target_column: str = None, smoothing: float = 10.0):
        """Initialize RouteFeatureExtractor."""
        self.target_column = target_column or data_config.target_column
        self.route_stats = None
        self.global_stats = {}  # Fallback values for unseen routes
        self.global_mean = 0.0
        self.smoothing = smoothing  # Smoothing factor for target encoding
        logger.info("RouteFeatureExtractor initialized (smoothed target encoding)")
    
    def fit(self, df: pd.DataFrame) -> 'RouteFeatureExtractor':
        """
        Fit the extractor by computing smoothed route statistics.
        
        Uses Bayesian smoothed mean: 
            encoded = (count * route_mean + smoothing * global_mean) / (count + smoothing)
        
        Args:
            df: Training DataFrame
        
        Returns:
            self
        """
        if 'Source' in df.columns and 'Destination' in df.columns:
            df = df.copy()
            df['Route'] = df['Source'] + '_to_' + df['Destination']
            
            if self.target_column in df.columns:
                self.global_mean = df[self.target_column].mean()
                
                route_agg = df.groupby('Route').agg({
                    self.target_column: ['mean', 'std', 'count']
                })
                route_agg.columns = ['Route_Mean_Fare', 'Route_Std_Fare', 'Route_Count']
                route_agg = route_agg.reset_index()
                
                # Apply Bayesian smoothing to prevent overfitting on rare routes
                route_agg['Route_Encoded_Fare'] = (
                    (route_agg['Route_Count'] * route_agg['Route_Mean_Fare'] + 
                     self.smoothing * self.global_mean) /
                    (route_agg['Route_Count'] + self.smoothing)
                ).round(2)
                
                self.route_stats = route_agg
                
                # Store global medians as fallback for unseen routes
                self.global_stats = {
                    'Route_Encoded_Fare': round(self.global_mean, 2),
                    'Route_Std_Fare': round(self.route_stats['Route_Std_Fare'].median(), 2),
                    'Route_Count': round(self.route_stats['Route_Count'].median(), 2)
                }
                
                logger.info(f"Fitted RouteFeatureExtractor on {len(self.route_stats)} unique routes "
                           f"(smoothing={self.smoothing})")
            else:
                self.route_stats = df.groupby('Route').size().reset_index(name='Route_Count')
                logger.info(f"Fitted RouteFeatureExtractor on {len(self.route_stats)} unique routes (no fare stats)")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform DataFrame by adding smoothed route features.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with route features
        """
        df = df.copy()
        
        if 'Source' in df.columns and 'Destination' in df.columns:
            df['Route'] = df['Source'] + '_to_' + df['Destination']
            
            if self.route_stats is not None:
                # Merge only the smoothed columns (not raw mean to prevent leakage)
                merge_cols = ['Route']
                for col in ['Route_Encoded_Fare', 'Route_Std_Fare', 'Route_Count']:
                    if col in self.route_stats.columns:
                        merge_cols.append(col)
                
                df = df.merge(self.route_stats[merge_cols], on='Route', how='left')
                
                # Fill missing route stats with global fallbacks
                for col in ['Route_Encoded_Fare', 'Route_Std_Fare', 'Route_Count']:
                    if col in df.columns:
                        fallback = getattr(self, 'global_stats', {}).get(col, 0)
                        df[col] = df[col].fillna(fallback)
            
            logger.info("Added smoothed route target-encoding features")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


class FeatureEncoder:
    """
    Handle categorical variable encoding.
    
    Supports both One-Hot Encoding and Label Encoding
    with proper handling of unseen categories during inference.
    """
    
    def __init__(
        self,
        encoding_type: str = 'onehot',
        categorical_columns: List[str] = None
    ):
        """
        Initialize FeatureEncoder.
        
        Args:
            encoding_type: 'onehot' or 'label'
            categorical_columns: List of columns to encode
        """
        self.encoding_type = encoding_type
        self.categorical_columns = categorical_columns or data_config.categorical_columns
        self.encoders = {}
        self.feature_names = None
        logger.info(f"FeatureEncoder initialized with {encoding_type} encoding")
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEncoder':
        """
        Fit encoders on the training data.
        
        Args:
            df: Training DataFrame
        
        Returns:
            self
        """
        for col in self.categorical_columns:
            if col not in df.columns:
                continue
            
            if self.encoding_type == 'onehot':
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoder.fit(df[[col]])
            else:
                encoder = LabelEncoder()
                encoder.fit(df[col].astype(str))
            
            self.encoders[col] = encoder
            logger.info(f"Fitted {self.encoding_type} encoder for {col}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical columns using fitted encoders.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with encoded categorical columns
        """
        df = df.copy()
        
        for col in self.categorical_columns:
            if col not in df.columns or col not in self.encoders:
                continue
            
            encoder = self.encoders[col]
            
            if self.encoding_type == 'onehot':
                encoded = encoder.transform(df[[col]])
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=df.index)
                df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
            else:
                # Handle unseen labels
                df[f"{col}_encoded"] = df[col].apply(
                    lambda x: encoder.transform([str(x)])[0] 
                    if str(x) in encoder.classes_ else -1
                )
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)


class FeatureScaler:
    """
    Handle numerical feature scaling.
    
    Supports StandardScaler and MinMaxScaler with proper
    fit/transform separation to prevent data leakage.
    """
    
    def __init__(
        self,
        scaling_type: str = 'standard',
        numerical_columns: List[str] = None
    ):
        """
        Initialize FeatureScaler.
        
        Args:
            scaling_type: 'standard' or 'minmax'
            numerical_columns: List of columns to scale
        """
        self.scaling_type = scaling_type
        self.numerical_columns = numerical_columns or data_config.numerical_columns
        
        if scaling_type == 'standard':
            self.scaler = StandardScaler()
        else:
            self.scaler = MinMaxScaler()
        
        self.is_fitted = False
        logger.info(f"FeatureScaler initialized with {scaling_type} scaling")
    
    def fit(self, df: pd.DataFrame) -> 'FeatureScaler':
        """
        Fit scaler on training data.
        
        Args:
            df: Training DataFrame
        
        Returns:
            self
        """
        cols_to_scale = [col for col in self.numerical_columns if col in df.columns]
        
        if cols_to_scale:
            self.scaler.fit(df[cols_to_scale])
            self.is_fitted = True
            logger.info(f"Fitted scaler on columns: {cols_to_scale}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical columns using fitted scaler.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with scaled numerical columns
        """
        if not self.is_fitted:
            raise ValueError("FeatureScaler must be fitted before transform")
        
        df = df.copy()
        cols_to_scale = [col for col in self.numerical_columns if col in df.columns]
        
        if cols_to_scale:
            df[cols_to_scale] = self.scaler.transform(df[cols_to_scale])
            logger.info(f"Scaled columns: {cols_to_scale}")
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform scaled columns back to original scale."""
        if not self.is_fitted:
            raise ValueError("FeatureScaler must be fitted before inverse_transform")
        
        df = df.copy()
        cols_to_scale = [col for col in self.numerical_columns if col in df.columns]
        
        if cols_to_scale:
            df[cols_to_scale] = self.scaler.inverse_transform(df[cols_to_scale])
        
        return df


class FeatureEngineer:
    """
    Main feature engineering orchestrator.
    
    Combines all feature extraction, encoding, and scaling
    operations into a coherent pipeline.
    """
    
    def __init__(
        self,
        encoding_type: str = 'onehot',
        scaling_type: str = 'standard',
        config=data_config
    ):
        """
        Initialize FeatureEngineer.
        
        Args:
            encoding_type: Type of categorical encoding
            scaling_type: Type of numerical scaling
            config: DataConfig instance
        """
        self.config = config
        self.date_extractor = DateFeatureExtractor()
        self.route_extractor = RouteFeatureExtractor(target_column=config.target_column)
        self.encoder = FeatureEncoder(encoding_type=encoding_type)
        self.scaler = FeatureScaler(scaling_type=scaling_type)
        
        self.feature_names = None
        self.is_fitted = False
        
        logger.info("FeatureEngineer initialized")
    
    def fit(self, df: pd.DataFrame, y: pd.Series = None) -> 'FeatureEngineer':
        """
        Fit all transformers on training data.
        
        Args:
            df: Training features DataFrame
            y: Training target (optional, used for route statistics)
        
        Returns:
            self
        """
        logger.info("Fitting FeatureEngineer on training data...")
        
        # Include target for route statistics if available
        df_with_target = df.copy()
        if y is not None and self.config.target_column not in df.columns:
            df_with_target[self.config.target_column] = y
        
        # Extract date features (stateless, no fitting needed)
        df_transformed = self.date_extractor.extract_features(df_with_target)
        
        # Fit route extractor (needs target for statistics)
        self.route_extractor.fit(df_transformed)
        df_transformed = self.route_extractor.transform(df_transformed)
        
        # Fit encoder
        self.encoder.fit(df_transformed)
        df_transformed = self.encoder.transform(df_transformed)
        
        # Fit scaler on numerical columns (excluding target)
        numerical_cols = [col for col in df_transformed.select_dtypes(
            include=[np.number]).columns if col != self.config.target_column]
        self.scaler.numerical_columns = numerical_cols
        self.scaler.fit(df_transformed)
        
        self.is_fitted = True
        logger.info("FeatureEngineer fitting complete")
        
        return self
    
    def transform(self, df: pd.DataFrame, scale_features: bool = True) -> pd.DataFrame:
        """
        Transform features using fitted transformers.
        
        Args:
            df: Input DataFrame
            scale_features: Whether to apply scaling
        
        Returns:
            Transformed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        logger.info("Transforming features...")
        
        # Extract date features
        df_transformed = self.date_extractor.extract_features(df)
        
        # Add route features
        df_transformed = self.route_extractor.transform(df_transformed)
        
        # Encode categorical variables
        df_transformed = self.encoder.transform(df_transformed)
        
        # Scale numerical features
        if scale_features:
            df_transformed = self.scaler.transform(df_transformed)
        
        # Add interaction features (after encoding & scaling)
        df_transformed = self._add_interaction_features(df_transformed)
        
        # Drop non-numeric columns that can't be used in modeling
        cols_to_drop = ['Date', 'DayName', 'MonthName', 'Route', 'Season']
        cols_to_drop = [col for col in cols_to_drop if col in df_transformed.columns]
        
        # Also drop original categorical columns if they still exist
        for col in self.config.categorical_columns:
            if col in df_transformed.columns:
                cols_to_drop.append(col)
        
        # Drop columns from config.columns_to_drop
        for col in self.config.columns_to_drop:
            if col in df_transformed.columns:
                cols_to_drop.append(col)
        
        # Drop any remaining non-numeric columns (object, category, datetime types)
        non_numeric_cols = df_transformed.select_dtypes(
            include=['object', 'category', 'datetime64']
        ).columns.tolist()
        cols_to_drop.extend([c for c in non_numeric_cols if c not in cols_to_drop])
        
        df_transformed = df_transformed.drop(columns=cols_to_drop, errors='ignore')
        
        self.feature_names = list(df_transformed.columns)
        
        return df_transformed
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features to capture combined effects.
        
        Creates multiplicative interactions between key feature pairs
        that have business meaning (e.g., duration * days_before_departure).
        """
        df = df.copy()
        
        # Duration × Days Before Departure (booking timing interacts with flight length)
        if 'Duration (hrs)' in df.columns and 'Days Before Departure' in df.columns:
            df['Duration_x_DaysBefore'] = df['Duration (hrs)'] * df['Days Before Departure']
        
        # Duration × IsWeekend (weekend long flights may be priced differently)
        if 'Duration (hrs)' in df.columns and 'IsWeekend' in df.columns:
            df['Duration_x_Weekend'] = df['Duration (hrs)'] * df['IsWeekend']
        
        # Route encoded fare × Days Before (route pricing changes with booking window)
        if 'Route_Encoded_Fare' in df.columns and 'Days Before Departure' in df.columns:
            df['RouteFare_x_DaysBefore'] = df['Route_Encoded_Fare'] * df['Days Before Departure']
        
        logger.info("Added interaction features")
        return df
    
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        y: pd.Series = None,
        scale_features: bool = True
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df, y).transform(df, scale_features)
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names after transformation."""
        return self.feature_names
    
    def save(self, path: Path = None) -> None:
        """
        Save the fitted FeatureEngineer to disk.
        
        Args:
            path: Path to save the transformer
        """
        if path is None:
            path = MODELS_DIR / "feature_engineer.pkl"
        
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Saved FeatureEngineer to {path}")
    
    @classmethod
    def load(cls, path: Path = None) -> 'FeatureEngineer':
        """
        Load a fitted FeatureEngineer from disk.
        
        Args:
            path: Path to the saved transformer
        
        Returns:
            Loaded FeatureEngineer instance
        """
        if path is None:
            path = MODELS_DIR / "feature_engineer.pkl"
        
        with open(path, 'rb') as f:
            instance = pickle.load(f)
        
        logger.info(f"Loaded FeatureEngineer from {path}")
        return instance


def prepare_train_test_data(
    df: pd.DataFrame,
    target_column: str = None,
    test_size: float = None,
    random_state: int = None,
    scale_features: bool = True,
    log_transform_target: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, 'FeatureEngineer']:
    """
    Prepare train/test split with feature engineering.
    
    This is the main entry point for feature engineering in the pipeline.
    It handles the entire process of splitting data, engineering features,
    and preparing data for model training.
    
    Args:
        df: Cleaned input DataFrame
        target_column: Name of target column
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        scale_features: Whether to scale numerical features
        log_transform_target: Whether to apply log1p transform to target
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_engineer)
    """
    target_column = target_column or data_config.target_column
    test_size = test_size or model_config.test_size
    random_state = random_state or model_config.random_state
    
    logger.info(f"Preparing train/test data with test_size={test_size}, "
               f"random_state={random_state}")
    
    # Separate features and target
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Apply log transform to reduce skew in target
    if log_transform_target:
        y = np.log1p(y)
        logger.info("Applied log1p transform to target variable (reduces right-skew)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")
    
    # Initialize and fit feature engineer on training data only
    feature_engineer = FeatureEngineer()
    feature_engineer.log_transform_target = log_transform_target  # Store flag for inference
    feature_engineer.fit(X_train, y_train)
    
    # Transform both train and test sets
    X_train_transformed = feature_engineer.transform(X_train, scale_features)
    X_test_transformed = feature_engineer.transform(X_test, scale_features)
    
    # Save feature engineer
    feature_engineer.save()
    
    logger.info(f"Final feature count: {len(feature_engineer.get_feature_names())}")
    
    return X_train_transformed, X_test_transformed, y_train, y_test, feature_engineer
