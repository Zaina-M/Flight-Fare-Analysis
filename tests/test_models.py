"""
Unit tests for the regression models module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.regressors import (
    LinearRegressionModel,
    RidgeRegressionModel,
    LassoRegressionModel,
    DecisionTreeRegressorModel,
    RandomForestRegressorModel,
    GradientBoostingRegressorModel,
    get_all_models,
    HAS_XGBOOST,
)
from src.models.base_model import ModelEvaluator


@pytest.fixture
def regression_data():
    """Generate simple regression data."""
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.uniform(0, 10, n),
    })
    y = 3 * X['feature_1'] + 2 * X['feature_2'] + np.random.randn(n) * 0.5
    return X, y


class TestLinearRegression:
    def test_fit_predict(self, regression_data):
        X, y = regression_data
        model = LinearRegressionModel()
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert model.is_fitted

    def test_evaluate(self, regression_data):
        X, y = regression_data
        model = LinearRegressionModel()
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert 'R2' in metrics
        assert metrics['R2'] > 0.5  # Should fit well

    def test_coefficients(self, regression_data):
        X, y = regression_data
        model = LinearRegressionModel()
        model.fit(X, y)
        coefs = model.get_coefficients()
        assert len(coefs) == 3


class TestRidgeRegression:
    def test_fit_predict(self, regression_data):
        X, y = regression_data
        model = RidgeRegressionModel(alpha=1.0)
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert metrics['R2'] > 0.5


class TestDecisionTree:
    def test_fit_predict(self, regression_data):
        X, y = regression_data
        model = DecisionTreeRegressorModel(max_depth=5)
        model.fit(X, y)
        assert model.is_fitted
        importances = model.get_feature_importances()
        assert len(importances) == 3


class TestRandomForest:
    def test_fit_predict(self, regression_data):
        X, y = regression_data
        model = RandomForestRegressorModel(n_estimators=10)
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert metrics['R2'] > 0.5
        assert model.get_oob_score() > 0


class TestGradientBoosting:
    def test_fit_predict(self, regression_data):
        X, y = regression_data
        model = GradientBoostingRegressorModel(n_estimators=50)
        model.fit(X, y)
        metrics = model.evaluate(X, y)
        assert metrics['R2'] > 0.5


class TestGetAllModels:
    def test_returns_all_models(self):
        models = get_all_models()
        assert 'Linear Regression' in models
        assert 'Ridge Regression' in models
        assert 'Random Forest' in models
        assert 'Gradient Boosting' in models
        assert 'Stacking Ensemble' in models
        if HAS_XGBOOST:
            assert 'XGBoost' in models


class TestModelEvaluator:
    def test_compute_all_metrics(self, regression_data):
        X, y = regression_data
        model = LinearRegressionModel()
        model.fit(X, y)
        y_pred = model.predict(X)
        evaluator = ModelEvaluator(save_plots=False)
        metrics = evaluator.compute_all_metrics(y.values, y_pred)
        assert 'R2' in metrics
        assert 'RMSE' in metrics
        assert 'MAE' in metrics

    def test_adjusted_r2(self):
        evaluator = ModelEvaluator(save_plots=False)
        adj_r2 = evaluator.compute_adjusted_r2(0.8, 100, 5)
        assert adj_r2 < 0.8  # Adjusted R2 should be less than R2
        assert adj_r2 > 0
