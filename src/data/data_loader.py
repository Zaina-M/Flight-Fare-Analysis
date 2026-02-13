# Data Loading and Cleaning Module.
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
from scipy.stats import mstats
import warnings
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import data_config, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataLoader:
    """
    Data loading and initial inspection class.
    
    Handles loading the flight price dataset and performing initial
    data quality assessments.
    """
    
    def __init__(self, config=data_config):
        
        self.config = config
        logger.info("DataLoader initialized")
    
    def load_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:

        #  the flight price dataset from CSV.
         
        if file_path is None:
            file_path = RAW_DATA_DIR / self.config.dataset_name
        
        logger.info(f"Loading dataset from: {file_path}")
        
        if not Path(file_path).exists():
            error_msg = f"Dataset not found at {file_path}. Please download from Kaggle."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            logger.info(f"Successfully loaded dataset with shape: {df.shape}")
            return df
        except UnicodeDecodeError:
            logger.warning("UTF-8 decoding failed, trying latin-1 encoding")
            df = pd.read_csv(file_path, encoding='latin-1')
            logger.info(f"Successfully loaded dataset with shape: {df.shape}")
            return df
    
    def inspect_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        # Perform initial data inspection and return summary statistics.
        
        logger.info("Performing data inspection...")
        
        inspection_result = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicates": df.duplicated().sum(),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2
        }
        
        # Log key findings
        logger.info(f"Dataset shape: {inspection_result['shape']}")
        logger.info(f"Number of columns: {len(inspection_result['columns'])}")
        logger.info(f"Duplicate rows: {inspection_result['duplicates']}")
        logger.info(f"Memory usage: {inspection_result['memory_usage_mb']:.2f} MB")
        
        # Log columns with missing values
        missing = {k: v for k, v in inspection_result['missing_values'].items() if v > 0}
        if missing:
            logger.warning(f"Columns with missing values: {missing}")
        
        return inspection_result
    
    def get_descriptive_stats(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        # Get descriptive statistics for numerical and categorical columns.
        
       
        logger.info("Computing descriptive statistics...")
        
        # Numerical statistics
        numerical_stats = df.describe()
        
        # Categorical statistics
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_stats = pd.DataFrame()
        
        for col in categorical_cols:
            col_stats = pd.DataFrame({
                'column': col,
                'unique_values': df[col].nunique(),
                'most_common': df[col].mode().iloc[0] if not df[col].mode().empty else None,
                'most_common_freq': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
            }, index=[0])
            categorical_stats = pd.concat([categorical_stats, col_stats], ignore_index=True)
        
        return numerical_stats, categorical_stats


class DataCleaner:
    # Handles missing value imputation, outlier detection, data type conversions, and data quality validation.
    
    
    def __init__(self, config=data_config):
        
        # Initialize DataCleaner with configuration.
  
        self.config = config
        self.cleaning_report = {}
        logger.info("DataCleaner initialized")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
       # Execute complete data cleaning pipeline.
    
        logger.info("Starting data cleaning pipeline...")
        original_shape = df.shape
        
        # Create copy to avoid modifying original
        df_cleaned = df.copy()
        
        # Step 1: Drop irrelevant columns
        df_cleaned = self._drop_irrelevant_columns(df_cleaned)
        
        # Step 2: Remove duplicate rows
        df_cleaned = self._remove_duplicates(df_cleaned)
        
        # Step 3: Normalize city names
        df_cleaned = self._normalize_city_names(df_cleaned)
        
        # Step 4: Handle missing values
        df_cleaned = self._handle_missing_values(df_cleaned)
        
        # Step 5: Validate and convert data types
        df_cleaned = self._convert_data_types(df_cleaned)
        
        # Step 6: Handle invalid entries
        df_cleaned = self._handle_invalid_entries(df_cleaned)
        
        # Step 7: Winsorize fare outliers
        df_cleaned = self._winsorize_outliers(df_cleaned)
        
        # Log cleaning summary
        logger.info(f"Cleaning complete. Shape: {original_shape} -> {df_cleaned.shape}")
        self.cleaning_report['original_shape'] = original_shape
        self.cleaning_report['final_shape'] = df_cleaned.shape
        self.cleaning_report['rows_removed'] = original_shape[0] - df_cleaned.shape[0]
        self.cleaning_report['columns_removed'] = original_shape[1] - df_cleaned.shape[1]
        
        return df_cleaned
    
    def _drop_irrelevant_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        #Drop columns that are irrelevant for analysis.
        columns_to_drop = []
        
        for col in df.columns:
            # Check if column matches any pattern in columns_to_drop config
            for pattern in self.config.columns_to_drop:
                if pattern.lower() in col.lower():
                    columns_to_drop.append(col)
                    break
        
        if columns_to_drop:
            logger.info(f"Dropping irrelevant columns: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop, errors='ignore')
        
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove duplicate rows.
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logger.info(f"Removing {duplicates} duplicate rows")
            df = df.drop_duplicates().reset_index(drop=True)
        return df
    
    def _normalize_city_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize inconsistent city names in Source and Destination columns."""
        city_cols = ['Source', 'Destination']
        
        for col in city_cols:
            if col in df.columns:
                # Apply normalization mapping
                for old_name, new_name in self.config.city_normalization.items():
                    mask = df[col].str.lower().str.strip() == old_name.lower()
                    if mask.any():
                        logger.info(f"Normalizing '{old_name}' to '{new_name}' in {col}")
                        df.loc[mask, col] = new_name
                
                # Strip whitespace and standardize capitalization
                df[col] = df[col].str.strip().str.title()
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values through imputation or removal."""
        missing_summary = df.isnull().sum()
        columns_with_missing = missing_summary[missing_summary > 0]
        
        if len(columns_with_missing) == 0:
            logger.info("No missing values found")
            return df
        
        logger.info(f"Handling missing values in: {list(columns_with_missing.index)}")
        
        for col in columns_with_missing.index:
            missing_pct = columns_with_missing[col] / len(df)
            
            # Drop columns with excessive missing values
            if missing_pct > self.config.missing_threshold:
                logger.warning(f"Dropping column {col} with {missing_pct:.1%} missing values")
                df = df.drop(columns=[col])
                continue
            
            # Impute based on data type
            if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                # Use median for numerical (more robust to outliers)
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Imputed {col} with median: {median_val}")
            else:
                # Use mode for categorical
                mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
                df[col] = df[col].fillna(mode_val)
                logger.info(f"Imputed {col} with mode: {mode_val}")
        
        return df
    
    def _convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert columns to appropriate data types."""
        
        # Convert numerical columns to float
        numerical_cols = self.config.numerical_columns + [self.config.target_column]
        for col in numerical_cols:
            if col in df.columns:
                try:
                    # Remove currency symbols and commas if present
                    if df[col].dtype == 'object':
                        df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.info(f"Converted {col} to numeric")
                except Exception as e:
                    logger.warning(f"Could not convert {col} to numeric: {e}")
        
        # Convert date column to datetime
        if self.config.date_column in df.columns:
            try:
                df[self.config.date_column] = pd.to_datetime(
                    df[self.config.date_column], 
                    errors='coerce',
                    infer_datetime_format=True
                )
                logger.info(f"Converted {self.config.date_column} to datetime")
            except Exception as e:
                logger.warning(f"Could not convert date column: {e}")
        
        return df
    
    def _handle_invalid_entries(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle invalid entries like negative fares."""
        fare_columns = ['Base Fare', 'Tax & Surcharge', 'Total Fare']
        
        for col in fare_columns:
            if col in df.columns:
                # Check for negative values
                negative_mask = df[col] < 0
                negative_count = negative_mask.sum()
                
                if negative_count > 0:
                    logger.warning(f"Found {negative_count} negative values in {col}")
                    # Remove rows with negative fare values
                    df = df[~negative_mask]
                    logger.info(f"Removed {negative_count} rows with negative {col}")
        
        # Validate Total Fare if components exist
        if all(col in df.columns for col in ['Base Fare', 'Tax & Surcharge', 'Total Fare']):
            # If Total Fare is missing, calculate it
            missing_total = df['Total Fare'].isnull()
            if missing_total.any():
                df.loc[missing_total, 'Total Fare'] = (
                    df.loc[missing_total, 'Base Fare'] + 
                    df.loc[missing_total, 'Tax & Surcharge']
                )
                logger.info(f"Calculated missing Total Fare for {missing_total.sum()} rows")
        
        return df.reset_index(drop=True)
    
    def _winsorize_outliers(
        self, df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99
    ) -> pd.DataFrame:
        # Winsorize extreme outliers in numerical fare/duration columns.
        cols_to_winsorize = [self.config.target_column, 'Duration (hrs)']
        
        for col in cols_to_winsorize:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                lo = df[col].quantile(lower)
                hi = df[col].quantile(upper)
                n_clipped = ((df[col] < lo) | (df[col] > hi)).sum()
                df[col] = df[col].clip(lower=lo, upper=hi)
                if n_clipped > 0:
                    logger.info(f"Winsorized {n_clipped} outliers in {col} "
                               f"(clipped to [{lo:.2f}, {hi:.2f}])")
        
        return df
    
    def get_cleaning_report(self) -> Dict[str, Any]:
        # Return the cleaning report with all transformations applied.
        return self.cleaning_report


def load_and_clean_data(
    file_path: Optional[Path] = None,
    save_processed: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    # Convenience function to load and clean data in one step.
    
    loader = DataLoader()
    df = loader.load_data(file_path)
    
    # Inspect data
    inspection_report = loader.inspect_data(df)
    
    # Clean data
    cleaner = DataCleaner()
    df_cleaned = cleaner.clean_data(df)
    
    # Add cleaning report to inspection report
    inspection_report['cleaning_report'] = cleaner.get_cleaning_report()
    
    # Save processed data
    if save_processed:
        processed_path = PROCESSED_DATA_DIR / "cleaned_flight_data.csv"
        df_cleaned.to_csv(processed_path, index=False)
        logger.info(f"Saved cleaned data to: {processed_path}")
    
    return df_cleaned, inspection_report
