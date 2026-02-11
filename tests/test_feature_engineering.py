"""
Unit tests for the feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.feature_engineering import (
    DateFeatureExtractor,
    RouteFeatureExtractor,
    FeatureEncoder,
    FeatureScaler,
    FeatureEngineer,
)


@pytest.fixture
def sample_df():
    """Sample DataFrame for feature engineering tests."""
    np.random.seed(42)
    n = 80
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


class TestDateFeatureExtractor:
    def test_extract_features(self, sample_df):
        extractor = DateFeatureExtractor()
        result = extractor.extract_features(sample_df)
        assert 'Month' in result.columns
        assert 'DayOfWeek' in result.columns
        assert 'IsWeekend' in result.columns
        assert 'Season' in result.columns

    def test_cyclical_encoding(self, sample_df):
        extractor = DateFeatureExtractor()
        result = extractor.extract_features(sample_df)
        assert 'Month_sin' in result.columns
        assert 'Month_cos' in result.columns
        assert 'DayOfWeek_sin' in result.columns
        assert 'DayOfWeek_cos' in result.columns
        # Verify sin/cos are bounded [-1, 1]
        assert result['Month_sin'].between(-1, 1).all()
        assert result['Month_cos'].between(-1, 1).all()

    def test_missing_date_column(self, sample_df):
        df = sample_df.drop(columns=['Departure Date & Time'])
        extractor = DateFeatureExtractor()
        result = extractor.extract_features(df)
        assert 'Month' not in result.columns  # graceful no-op


class TestRouteFeatureExtractor:
    def test_fit_transform(self, sample_df):
        extractor = RouteFeatureExtractor()
        result = extractor.fit_transform(sample_df)
        assert 'Route' in result.columns
        assert 'Route_Encoded_Fare' in result.columns

    def test_unseen_route_fallback(self, sample_df):
        extractor = RouteFeatureExtractor()
        extractor.fit(sample_df)
        
        new_df = pd.DataFrame({
            'Source': ['NewCity'],
            'Destination': ['UnknownCity'],
            'Total Fare (BDT)': [5000],
        })
        result = extractor.transform(new_df)
        # Should not have NaN — should use global fallback
        assert not result['Route_Encoded_Fare'].isna().any()

    def test_smoothing_effect(self, sample_df):
        extractor = RouteFeatureExtractor(smoothing=10.0)
        result = extractor.fit_transform(sample_df)
        # Smoothed values should exist
        assert 'Route_Encoded_Fare' in result.columns


class TestFeatureEncoder:
    def test_onehot_encoding(self, sample_df):
        encoder = FeatureEncoder(encoding_type='onehot', categorical_columns=['Airline'])
        result = encoder.fit_transform(sample_df)
        assert 'Airline' not in result.columns
        assert any('Airline_' in col for col in result.columns)

    def test_unseen_category(self, sample_df):
        encoder = FeatureEncoder(encoding_type='onehot', categorical_columns=['Airline'])
        encoder.fit(sample_df)
        new_df = pd.DataFrame({'Airline': ['NewAirline']})
        result = encoder.transform(new_df)
        # Should handle gracefully with 0s for unseen categories
        assert 'Airline' not in result.columns


class TestFeatureScaler:
    def test_standard_scaling(self, sample_df):
        scaler = FeatureScaler(scaling_type='standard', numerical_columns=['Duration (hrs)'])
        scaler.fit(sample_df)
        result = scaler.transform(sample_df)
        assert abs(result['Duration (hrs)'].mean()) < 0.1  # near-zero mean

    def test_not_fitted_error(self, sample_df):
        scaler = FeatureScaler()
        with pytest.raises(ValueError):
            scaler.transform(sample_df)


class TestFeatureEngineer:
    def test_fit_transform(self, sample_df):
        engineer = FeatureEngineer()
        X = sample_df.drop(columns=['Total Fare (BDT)'])
        y = sample_df['Total Fare (BDT)']
        result = engineer.fit_transform(X, y)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(X)
        # Should have no object columns
        assert result.select_dtypes(include=['object']).shape[1] == 0

    def test_interaction_features(self, sample_df):
        engineer = FeatureEngineer()
        X = sample_df.drop(columns=['Total Fare (BDT)'])
        y = sample_df['Total Fare (BDT)']
        result = engineer.fit_transform(X, y)
        assert 'Duration_x_DaysBefore' in result.columns
