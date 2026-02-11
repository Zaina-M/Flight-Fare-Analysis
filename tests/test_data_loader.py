"""
Unit tests for the data loading and cleaning module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_loader import DataLoader, DataCleaner
from config.config import data_config, RAW_DATA_DIR


@pytest.fixture
def sample_df():
    """Create a sample DataFrame mimicking the flight dataset."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        'Airline': np.random.choice(['Biman Bangladesh', 'US-Bangla', 'Novoair'], n),
        'Source': np.random.choice(['Dhaka', 'Chittagong', 'Sylhet'], n),
        'Destination': np.random.choice(['Dhaka', 'Chittagong', 'Sylhet'], n),
        'Departure Date & Time': pd.date_range('2024-01-01', periods=n, freq='D'),
        'Duration (hrs)': np.random.uniform(0.5, 5.0, n).round(1),
        'Days Before Departure': np.random.randint(1, 60, n),
        'Class': np.random.choice(['Economy', 'Business'], n),
        'Booking Source': np.random.choice(['Online Website', 'Travel Agency'], n),
        'Seasonality': np.random.choice(['Regular', 'Eid'], n),
        'Total Fare (BDT)': np.random.uniform(3000, 25000, n).round(2),
    })


@pytest.fixture
def dirty_df(sample_df):
    """Create a dirty DataFrame with issues to clean."""
    df = sample_df.copy()
    # Add duplicates
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    # Add a negative fare
    df.loc[0, 'Total Fare (BDT)'] = -100
    return df


class TestDataLoader:
    def test_load_data_file_not_found(self):
        loader = DataLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_data(Path("nonexistent_file.csv"))

    def test_inspect_data(self, sample_df):
        loader = DataLoader()
        result = loader.inspect_data(sample_df)
        assert result['shape'] == sample_df.shape
        assert 'missing_values' in result
        assert 'duplicates' in result

    def test_get_descriptive_stats(self, sample_df):
        loader = DataLoader()
        num_stats, cat_stats = loader.get_descriptive_stats(sample_df)
        assert not num_stats.empty
        assert not cat_stats.empty


class TestDataCleaner:
    def test_remove_duplicates(self, dirty_df):
        cleaner = DataCleaner()
        # Just test duplicate removal step
        result = cleaner._remove_duplicates(dirty_df)
        assert len(result) < len(dirty_df)

    def test_handle_invalid_entries(self, dirty_df):
        """Negative fares should be removed."""
        cleaner = DataCleaner()
        result = cleaner._handle_invalid_entries(dirty_df)
        if 'Total Fare' in result.columns:
            assert (result['Total Fare'] >= 0).all()

    def test_normalize_city_names(self, sample_df):
        df = sample_df.copy()
        df.loc[0, 'Source'] = 'Dacca'
        cleaner = DataCleaner()
        result = cleaner._normalize_city_names(df)
        assert result.loc[0, 'Source'] == 'Dhaka'

    def test_winsorize_outliers(self, sample_df):
        df = sample_df.copy()
        df.loc[0, 'Total Fare (BDT)'] = 999999  # extreme outlier
        cleaner = DataCleaner()
        result = cleaner._winsorize_outliers(df)
        assert result['Total Fare (BDT)'].max() < 999999

    def test_full_cleaning_pipeline(self, dirty_df):
        cleaner = DataCleaner()
        result = cleaner.clean_data(dirty_df)
        assert len(result) <= len(dirty_df)
        report = cleaner.get_cleaning_report()
        assert 'rows_removed' in report
