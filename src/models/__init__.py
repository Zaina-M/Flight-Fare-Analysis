"""
Models Package Initialization.

Provides access to all regression models, evaluation utilities,
and model comparison framework.
"""

from .base_model import BaseRegressor, ModelEvaluator
from .regressors import (
    LinearRegressionModel,
    RidgeRegressionModel,
    LassoRegressionModel,
    DecisionTreeRegressorModel,
    RandomForestRegressorModel,
    GradientBoostingRegressorModel,
    FeatureImportancePlotter,
    get_all_models
)
from .model_comparison import ModelComparison, compare_models

__all__ = [
    # Base classes
    'BaseRegressor',
    'ModelEvaluator',
    # Linear models
    'LinearRegressionModel',
    'RidgeRegressionModel',
    'LassoRegressionModel',
    # Tree-based models
    'DecisionTreeRegressorModel',
    'RandomForestRegressorModel',
    'GradientBoostingRegressorModel',
    # Utilities
    'FeatureImportancePlotter',
    'get_all_models',
    'ModelComparison',
    'compare_models'
]
