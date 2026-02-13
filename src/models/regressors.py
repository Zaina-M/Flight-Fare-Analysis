# Regression Models Module.

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.exceptions import ConvergenceWarning
import warnings

# Try to import XGBoost (optional dependency)
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import model_config, MODELS_DIR, PLOTS_DIR
from src.models.base_model import BaseRegressor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Suppress convergence warnings for Lasso during hyperparameter search
warnings.filterwarnings('ignore', category=ConvergenceWarning)

FIGURE_DPI = 150


class LinearRegressionModel(BaseRegressor):
    """
    Linear Regression baseline model.
    
    Serves as the baseline model for comparison with more
    complex methods. Simple, interpretable, and fast.
    """
    
    def __init__(self, random_state: int = None):
        """
        Initialize Linear Regression model.
        
        Args:
            random_state: Random seed (not used by LinearRegression, 
                         kept for consistency)
        """
        super().__init__(model_name="Linear Regression", random_state=random_state)
    
    def _create_model(self) -> LinearRegression:
        """Create the Linear Regression model."""
        return LinearRegression()
    
    def get_coefficients(self) -> pd.DataFrame:
        """
        Get feature coefficients from the fitted model.
        
        Returns:
            DataFrame with features and their coefficients
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing coefficients")
        
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_
        })
        coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
        
        return coef_df
    
    def get_intercept(self) -> float:
        """Get the intercept of the model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing intercept")
        return self.model.intercept_


class RidgeRegressionModel(BaseRegressor):
    """
    Ridge Regression model with L2 regularization.
    
    Helps prevent overfitting by penalizing large coefficients.
    Good for multicollinearity.
    """
    
    def __init__(self, alpha: float = 1.0, random_state: int = None):
        """
        Initialize Ridge Regression model.
        
        Args:
            alpha: Regularization strength
            random_state: Random seed for reproducibility
        """
        super().__init__(model_name="Ridge Regression", random_state=random_state)
        self.alpha = alpha
    
    def _create_model(self) -> Ridge:
        """Create the Ridge Regression model."""
        return Ridge(alpha=self.alpha, random_state=self.random_state)
    
    def get_coefficients(self) -> pd.DataFrame:
        """Get feature coefficients from the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing coefficients")
        
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_
        })
        coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
        
        return coef_df
    
    def tune_hyperparameters(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        param_grid: Dict = None,
        cv: int = None,
        scoring: str = 'neg_root_mean_squared_error'
    ) -> Dict[str, Any]:
        """
        Tune alpha hyperparameter using GridSearchCV.
        
        Args:
            X: Training features
            y: Training target
            param_grid: Parameter grid (defaults to config)
            cv: Cross-validation folds
            scoring: Scoring metric
        
        Returns:
            Dictionary with best parameters and results
        """
        param_grid = param_grid or model_config.ridge_params
        cv = cv or model_config.cv_folds
        
        logger.info(f"Tuning {self.model_name} hyperparameters...")
        
        grid_search = GridSearchCV(
            Ridge(random_state=self.random_state),
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True
        )
        
        grid_search.fit(X, y)
        
        self.alpha = grid_search.best_params_['alpha']
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,  # Negate since we used neg_rmse
            'cv_results': pd.DataFrame(grid_search.cv_results_)
        }
        
        logger.info(f"Best alpha: {self.alpha}, Best Score (RMSE): {results['best_score']:.4f}")
        
        return results


class LassoRegressionModel(BaseRegressor):
    """
    Lasso Regression model with L1 regularization.
    
    Performs feature selection by driving some coefficients to zero.
    Useful for sparse models.
    """
    
    def __init__(self, alpha: float = 1.0, random_state: int = None):
        """
        Initialize Lasso Regression model.
        
        Args:
            alpha: Regularization strength
            random_state: Random seed for reproducibility
        """
        super().__init__(model_name="Lasso Regression", random_state=random_state)
        self.alpha = alpha
    
    def _create_model(self) -> Lasso:
        """Create the Lasso Regression model."""
        return Lasso(alpha=self.alpha, random_state=self.random_state, max_iter=10000)
    
    def get_coefficients(self) -> pd.DataFrame:
        """Get feature coefficients from the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing coefficients")
        
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': self.model.coef_
        })
        coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
        coef_df['Is_Zero'] = coef_df['Coefficient'] == 0
        coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
        
        n_zero = coef_df['Is_Zero'].sum()
        logger.info(f"Lasso set {n_zero}/{len(coef_df)} coefficients to zero")
        
        return coef_df
    
    def get_selected_features(self) -> List[str]:
        """Get list of features with non-zero coefficients."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing features")
        
        coef_df = self.get_coefficients()
        return coef_df[~coef_df['Is_Zero']]['Feature'].tolist()
    
    def tune_hyperparameters(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        param_grid: Dict = None,
        cv: int = None,
        scoring: str = 'neg_root_mean_squared_error'
    ) -> Dict[str, Any]:
        """Tune alpha hyperparameter using GridSearchCV."""
        param_grid = param_grid or model_config.lasso_params
        cv = cv or model_config.cv_folds
        
        logger.info(f"Tuning {self.model_name} hyperparameters...")
        
        grid_search = GridSearchCV(
            Lasso(random_state=self.random_state, max_iter=10000),
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True
        )
        
        grid_search.fit(X, y)
        
        self.alpha = grid_search.best_params_['alpha']
        self.model = grid_search.best_estimator_
        self.is_fitted = True
        
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': -grid_search.best_score_,
            'cv_results': pd.DataFrame(grid_search.cv_results_)
        }
        
        logger.info(f"Best alpha: {self.alpha}, Best Score (RMSE): {results['best_score']:.4f}")
        
        return results


class DecisionTreeRegressorModel(BaseRegressor):
    """
    Decision Tree Regressor model.
    
    Non-linear model that partitions feature space based on
    decision rules. Highly interpretable but prone to overfitting.
    """
    
    def __init__(
        self,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = None
    ):
        """
        Initialize Decision Tree Regressor.
        
        Args:
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf
            random_state: Random seed for reproducibility
        """
        super().__init__(model_name="Decision Tree", random_state=random_state)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
    
    def _create_model(self) -> DecisionTreeRegressor:
        """Create the Decision Tree Regressor model."""
        return DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state
        )
    
    def get_feature_importances(self) -> pd.DataFrame:
        """Get feature importances from the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing feature importances")
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df
    
    def tune_hyperparameters(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        param_grid: Dict = None,
        cv: int = None,
        scoring: str = 'neg_root_mean_squared_error',
        use_randomized: bool = False,
        n_iter: int = 50
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using Grid or Randomized Search.
        
        Args:
            X: Training features
            y: Training target
            param_grid: Parameter grid (defaults to config)
            cv: Cross-validation folds
            scoring: Scoring metric
            use_randomized: Use RandomizedSearchCV instead of GridSearch
            n_iter: Number of iterations for randomized search
        
        Returns:
            Dictionary with best parameters and results
        """
        param_grid = param_grid or model_config.decision_tree_params
        cv = cv or model_config.cv_folds
        
        logger.info(f"Tuning {self.model_name} hyperparameters...")
        
        base_model = DecisionTreeRegressor(random_state=self.random_state)
        
        if use_randomized:
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=n_iter,
                cv=cv, scoring=scoring, n_jobs=-1,
                random_state=self.random_state,
                return_train_score=True
            )
        else:
            search = GridSearchCV(
                base_model, param_grid,
                cv=cv, scoring=scoring, n_jobs=-1,
                return_train_score=True
            )
        
        search.fit(X, y)
        
        self.max_depth = search.best_params_.get('max_depth', self.max_depth)
        self.min_samples_split = search.best_params_.get('min_samples_split', self.min_samples_split)
        self.min_samples_leaf = search.best_params_.get('min_samples_leaf', self.min_samples_leaf)
        self.model = search.best_estimator_
        self.is_fitted = True
        
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        results = {
            'best_params': search.best_params_,
            'best_score': -search.best_score_,
            'cv_results': pd.DataFrame(search.cv_results_)
        }
        
        logger.info(f"Best params: {search.best_params_}, Best Score (RMSE): {results['best_score']:.4f}")
        
        return results


class RandomForestRegressorModel(BaseRegressor):
    """
    Random Forest Regressor model.
    
    Ensemble of decision trees that reduces overfitting through
    bagging and feature randomization. Good balance of accuracy
    and interpretability.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        random_state: int = None,
        n_jobs: int = -1
    ):
        """
        Initialize Random Forest Regressor.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum tree depth
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        super().__init__(model_name="Random Forest", random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
    
    def _create_model(self) -> RandomForestRegressor:
        """Create the Random Forest Regressor model."""
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            oob_score=True  # Enable out-of-bag score for validation
        )
    
    def get_feature_importances(self) -> pd.DataFrame:
        """Get feature importances from the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing feature importances")
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum()
        
        return importance_df
    
    def get_oob_score(self) -> float:
        """Get out-of-bag R² score."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing OOB score")
        return self.model.oob_score_
    
    def tune_hyperparameters(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        param_grid: Dict = None,
        cv: int = None,
        scoring: str = 'neg_root_mean_squared_error',
        use_randomized: bool = True,
        n_iter: int = 30
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using Grid or Randomized Search.
        
        RandomizedSearchCV is recommended due to large parameter space.
        """
        param_grid = param_grid or model_config.random_forest_params
        cv = cv or model_config.cv_folds
        
        logger.info(f"Tuning {self.model_name} hyperparameters...")
        
        base_model = RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        
        if use_randomized:
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=n_iter,
                cv=cv, scoring=scoring, n_jobs=-1,
                random_state=self.random_state,
                return_train_score=True
            )
        else:
            search = GridSearchCV(
                base_model, param_grid,
                cv=cv, scoring=scoring, n_jobs=-1,
                return_train_score=True
            )
        
        search.fit(X, y)
        
        # Update parameters
        for param, value in search.best_params_.items():
            if hasattr(self, param):
                setattr(self, param, value)
        
        self.model = search.best_estimator_
        self.is_fitted = True
        
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        results = {
            'best_params': search.best_params_,
            'best_score': -search.best_score_,
            'cv_results': pd.DataFrame(search.cv_results_),
            'oob_score': self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None
        }
        
        logger.info(f"Best params: {search.best_params_}, Best Score (RMSE): {results['best_score']:.4f}")
        
        return results


class GradientBoostingRegressorModel(BaseRegressor):
    """
    Gradient Boosting Regressor model.
    
    Sequential ensemble method that builds trees to correct errors
    from previous iterations. Often achieves the best performance.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        subsample: float = 1.0,
        random_state: int = None
    ):
        """
        Initialize Gradient Boosting Regressor.
        
        Args:
            n_estimators: Number of boosting stages
            max_depth: Maximum tree depth
            learning_rate: Learning rate shrinks contribution of each tree
            subsample: Fraction of samples for fitting each tree
            random_state: Random seed for reproducibility
        """
        super().__init__(model_name="Gradient Boosting", random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
    
    def _create_model(self) -> GradientBoostingRegressor:
        """Create the Gradient Boosting Regressor model."""
        return GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=self.random_state
        )
    
    def get_feature_importances(self) -> pd.DataFrame:
        """Get feature importances from the fitted model."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing feature importances")
        
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum()
        
        return importance_df
    
    def get_training_deviance(self) -> np.ndarray:
        """Get training deviance at each boosting stage."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing training deviance")
        return self.model.train_score_
    
    def tune_hyperparameters(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        param_grid: Dict = None,
        cv: int = None,
        scoring: str = 'neg_root_mean_squared_error',
        use_randomized: bool = True,
        n_iter: int = 30
    ) -> Dict[str, Any]:
        """
        Tune hyperparameters using Grid or Randomized Search.
        
        RandomizedSearchCV is recommended due to computational cost.
        """
        param_grid = param_grid or model_config.gradient_boosting_params
        cv = cv or model_config.cv_folds
        
        logger.info(f"Tuning {self.model_name} hyperparameters...")
        
        base_model = GradientBoostingRegressor(random_state=self.random_state)
        
        if use_randomized:
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=n_iter,
                cv=cv, scoring=scoring, n_jobs=-1,
                random_state=self.random_state,
                return_train_score=True
            )
        else:
            search = GridSearchCV(
                base_model, param_grid,
                cv=cv, scoring=scoring, n_jobs=-1,
                return_train_score=True
            )
        
        search.fit(X, y)
        
        # Update parameters
        for param, value in search.best_params_.items():
            if hasattr(self, param):
                setattr(self, param, value)
        
        self.model = search.best_estimator_
        self.is_fitted = True
        
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        results = {
            'best_params': search.best_params_,
            'best_score': -search.best_score_,
            'cv_results': pd.DataFrame(search.cv_results_)
        }
        
        logger.info(f"Best params: {search.best_params_}, Best Score (RMSE): {results['best_score']:.4f}")
        
        return results


class XGBoostRegressorModel(BaseRegressor):
    """
    XGBoost Regressor model.
    
    Gradient boosted trees using the XGBoost library.
    Often achieves state-of-the-art results on tabular data.
    Requires: pip install xgboost
    """
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = None
    ):
        super().__init__(model_name="XGBoost", random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
    
    def _create_model(self):
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed. Install with: pip install xgboost")
        return XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
    
    def get_feature_importances(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before accessing feature importances")
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.model.feature_importances_
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        importance_df['Cumulative_Importance'] = importance_df['Importance'].cumsum()
        return importance_df
    
    def tune_hyperparameters(
        self,
        X, y,
        param_grid: Dict = None,
        cv: int = None,
        scoring: str = 'neg_root_mean_squared_error',
        use_randomized: bool = True,
        n_iter: int = 20
    ) -> Dict[str, Any]:
        if not HAS_XGBOOST:
            raise ImportError("xgboost is not installed.")
        
        param_grid = param_grid or {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0]
        }
        cv = cv or model_config.cv_folds
        
        logger.info(f"Tuning {self.model_name} hyperparameters...")
        
        base_model = XGBRegressor(random_state=self.random_state, n_jobs=-1, verbosity=0)
        
        if use_randomized:
            search = RandomizedSearchCV(
                base_model, param_grid, n_iter=n_iter,
                cv=cv, scoring=scoring, n_jobs=-1,
                random_state=self.random_state, return_train_score=True
            )
        else:
            search = GridSearchCV(
                base_model, param_grid, cv=cv, scoring=scoring,
                n_jobs=-1, return_train_score=True
            )
        
        search.fit(X, y)
        
        for param, value in search.best_params_.items():
            if hasattr(self, param):
                setattr(self, param, value)
        
        self.model = search.best_estimator_
        self.is_fitted = True
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        results = {
            'best_params': search.best_params_,
            'best_score': -search.best_score_,
            'cv_results': pd.DataFrame(search.cv_results_)
        }
        logger.info(f"Best params: {search.best_params_}, Best Score (RMSE): {results['best_score']:.4f}")
        return results


class StackingRegressorModel(BaseRegressor):
    """
    Stacking Ensemble Regressor.
    
    Uses Random Forest and Gradient Boosting as base estimators
    with Ridge Regression as the meta-learner.
    """
    
    def __init__(self, random_state: int = None):
        super().__init__(model_name="Stacking Ensemble", random_state=random_state)
    
    def _create_model(self):
        base_estimators = [
            ('rf', RandomForestRegressor(
                n_estimators=100, max_depth=10,
                random_state=self.random_state, n_jobs=-1
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=100, max_depth=5,
                learning_rate=0.1, random_state=self.random_state
            )),
        ]
        
        # Add XGBoost if available
        if HAS_XGBOOST:
            base_estimators.append(
                ('xgb', XGBRegressor(
                    n_estimators=100, max_depth=5,
                    learning_rate=0.1, random_state=self.random_state,
                    n_jobs=-1, verbosity=0
                ))
            )
        
        return StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=1.0, random_state=self.random_state),
            cv=3,
            n_jobs=-1
        )


class FeatureImportancePlotter:
    """
    Utility class for visualizing feature importance.
    
    Provides consistent feature importance visualizations
    across different model types.
    """
    
    def __init__(self, save_plots: bool = True, plot_dir: Path = None, run_id: str = None):
        """
        Initialize FeatureImportancePlotter.
        
        Args:
            save_plots: Whether to save plots to disk
            plot_dir: Directory for saving plots (should be run-specific folder)
            run_id: Run identifier for folder organization
        """
        self.save_plots = save_plots
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use run subfolder
        base_dir = plot_dir or PLOTS_DIR
        self.plot_dir = base_dir / f"run_{self.run_id}"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"FeatureImportancePlotter initialized, saving to: {self.plot_dir}")
    
    def _save_figure(self, fig: plt.Figure, name: str) -> Optional[Path]:
        """Save figure to disk (no timestamp in filename since folder has it)."""
        if self.save_plots:
            filename = f"{name}.png"
            filepath = self.plot_dir / filename
            fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            logger.info(f"Saved plot: {filepath}")
            plt.close(fig)
            return filepath
        plt.close(fig)
        return None
    
    def plot_linear_coefficients(
        self,
        model: Union[LinearRegressionModel, RidgeRegressionModel, LassoRegressionModel],
        top_n: int = 15
    ) -> Optional[Path]:
        """
        Plot coefficients for linear models.
        
        Args:
            model: Fitted linear model
            top_n: Number of top features to display
        
        Returns:
            Path to saved plot
        """
        logger.info(f"Plotting coefficients for {model.model_name}...")
        
        coef_df = model.get_coefficients()
        
        # Get top N by absolute value
        top_features = coef_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
        
        colors = ['coral' if c < 0 else 'steelblue' for c in top_features['Coefficient']]
        
        ax.barh(
            range(len(top_features)),
            top_features['Coefficient'],
            color=colors,
            edgecolor='white'
        )
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Coefficient Value', fontsize=12)
        ax.set_title(f'{model.model_name}: Feature Coefficients (Top {top_n})',
                    fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.invert_yaxis()
        
        # Add value labels
        for idx, (_, row) in enumerate(top_features.iterrows()):
            ax.text(
                row['Coefficient'] + (0.02 * max(abs(top_features['Coefficient']))),
                idx, f"{row['Coefficient']:.4f}",
                va='center', fontsize=9
            )
        
        plt.tight_layout()
        return self._save_figure(fig, f'coefficients_{model.model_name.lower().replace(" ", "_")}')
    
    def plot_tree_feature_importance(
        self,
        model: Union[DecisionTreeRegressorModel, RandomForestRegressorModel, GradientBoostingRegressorModel],
        top_n: int = 15
    ) -> Optional[Path]:
        """
        Plot feature importance for tree-based models.
        
        Args:
            model: Fitted tree-based model
            top_n: Number of top features to display
        
        Returns:
            Path to saved plot
        """
        logger.info(f"Plotting feature importance for {model.model_name}...")
        
        importance_df = model.get_feature_importances()
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
        
        bars = ax.barh(
            range(len(top_features)),
            top_features['Importance'],
            color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features))),
            edgecolor='white'
        )
        
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'{model.model_name}: Feature Importance (Top {top_n})',
                    fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        
        # Add value labels
        for idx, bar in enumerate(bars):
            width = bar.get_width()
            ax.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                   f'{width:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        return self._save_figure(fig, f'importance_{model.model_name.lower().replace(" ", "_")}')
    
    def plot_regularization_path(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        alphas: List[float] = None,
        model_type: str = 'ridge',
        feature_names: List[str] = None
    ) -> Optional[Path]:
        """
        Plot regularization path showing how coefficients change with alpha.
        
        Args:
            X: Training features
            y: Training target
            alphas: List of alpha values to evaluate
            model_type: 'ridge' or 'lasso'
            feature_names: Names of features
        
        Returns:
            Path to saved plot
        """
        logger.info(f"Plotting {model_type} regularization path...")
        
        if alphas is None:
            alphas = np.logspace(-4, 4, 50)
        
        if isinstance(X, pd.DataFrame):
            feature_names = feature_names or X.columns.tolist()
            X = X.values
        else:
            feature_names = feature_names or [f'Feature_{i}' for i in range(X.shape[1])]
        
        coefs = []
        for alpha in alphas:
            if model_type == 'ridge':
                model = Ridge(alpha=alpha)
            else:
                model = Lasso(alpha=alpha, max_iter=10000)
            
            model.fit(X, y)
            coefs.append(model.coef_)
        
        coefs = np.array(coefs)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i in range(coefs.shape[1]):
            ax.plot(alphas, coefs[:, i], label=feature_names[i])
        
        ax.set_xscale('log')
        ax.set_xlabel('Alpha (Regularization Strength)', fontsize=12)
        ax.set_ylabel('Coefficient Value', fontsize=12)
        ax.set_title(f'{model_type.capitalize()} Regularization Path',
                    fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
        
        # Only show legend if not too many features
        if len(feature_names) <= 10:
            ax.legend(loc='best', fontsize=8)
        
        plt.tight_layout()
        return self._save_figure(fig, f'regularization_path_{model_type}')


def get_all_models(random_state: int = None) -> Dict[str, BaseRegressor]:
    """
    Get dictionary of all available models.
    
    Args:
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary mapping model names to model instances
    """
    random_state = random_state or model_config.random_state
    
    models = {
        'Linear Regression': LinearRegressionModel(random_state=random_state),
        'Ridge Regression': RidgeRegressionModel(random_state=random_state),
        'Lasso Regression': LassoRegressionModel(random_state=random_state),
        'Decision Tree': DecisionTreeRegressorModel(random_state=random_state),
        'Random Forest': RandomForestRegressorModel(random_state=random_state),
        'Gradient Boosting': GradientBoostingRegressorModel(random_state=random_state),
    }
    
    # Add XGBoost if available
    if HAS_XGBOOST:
        models['XGBoost'] = XGBoostRegressorModel(random_state=random_state)
        logger.info("XGBoost is available and included in model comparison")
    else:
        logger.warning("XGBoost not installed — skipping. Install with: pip install xgboost")
    
    # Stacking Ensemble disabled for faster training
    # Uncomment if you want ensemble of all models (very slow)
    # models['Stacking Ensemble'] = StackingRegressorModel(random_state=random_state)
    
    return models
