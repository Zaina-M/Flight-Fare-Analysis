#Base Model and Evaluation Module.

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
from abc import ABC, abstractmethod
from datetime import datetime
import joblib
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error, 
    mean_squared_error, 
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score, cross_validate

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import model_config, MODELS_DIR, PLOTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Configure plot style
plt.style.use('seaborn-v0_8-whitegrid')
FIGURE_DPI = 150


class BaseRegressor(ABC):
    """
    Abstract base class for all regression models.
    
    Provides a consistent interface and common functionality
    for all regression models in the project.
    """
    
    def __init__(self, model_name: str = "BaseRegressor", random_state: int = None):
        
        # Initialize BaseRegressor.
        
        self.model_name = model_name
        self.random_state = random_state or model_config.random_state
        self.model = None
        self.is_fitted = False
        self.training_timestamp = None
        self.feature_names = None
        logger.info(f"Initialized {self.model_name}")
    
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying sklearn model. To be implemented by subclasses."""
        pass
    
    def fit(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray]
    ) -> 'BaseRegressor':
    
       #  Fit the model to training data.
        
        
        logger.info(f"Training {self.model_name}...")
        
        if self.model is None:
            self.model = self._create_model()
        
        # Store feature names if DataFrame
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Fit the model
        self.model.fit(X, y)
        self.is_fitted = True
        self.training_timestamp = datetime.now()
        
        logger.info(f"{self.model_name} training completed at {self.training_timestamp}")
        return self
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        
       # Make predictions on new data.
        
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted before prediction")
        
        return self.model.predict(X)
    
    def evaluate(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        
        # Evaluate model performance.
        
        y_pred = self.predict(X)
        
        metrics = {
            'R2': r2_score(y, y_pred),
            'MAE': mean_absolute_error(y, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
            'MSE': mean_squared_error(y, y_pred),
            'MAPE': mean_absolute_percentage_error(y, y_pred) * 100
        }
        
        logger.info(f"{self.model_name} Evaluation Metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics
    
    def cross_validate(
        self, 
        X: Union[pd.DataFrame, np.ndarray], 
        y: Union[pd.Series, np.ndarray],
        cv: int = None
    ) -> Dict[str, Any]:
        
       # Perform cross-validation.
        
        cv = cv or model_config.cv_folds
        
        logger.info(f"Running {cv}-fold cross-validation for {self.model_name}...")
        
        if self.model is None:
            self.model = self._create_model()
        
        scoring = {
            'R2': 'r2',
            'MAE': 'neg_mean_absolute_error',
            'RMSE': 'neg_root_mean_squared_error'
        }
        
        cv_results = cross_validate(
            self.model, X, y, cv=cv, 
            scoring=scoring,
            return_train_score=True
        )
        
        results = {
            'cv_folds': cv,
            'R2_mean': cv_results['test_R2'].mean(),
            'R2_std': cv_results['test_R2'].std(),
            'MAE_mean': -cv_results['test_MAE'].mean(),
            'MAE_std': cv_results['test_MAE'].std(),
            'RMSE_mean': -cv_results['test_RMSE'].mean(),
            'RMSE_std': cv_results['test_RMSE'].std(),
            'train_R2_mean': cv_results['train_R2'].mean(),
            'train_R2_std': cv_results['train_R2'].std()
        }
        
        logger.info(f"Cross-validation results for {self.model_name}:")
        logger.info(f"  R² (test): {results['R2_mean']:.4f} ± {results['R2_std']:.4f}")
        logger.info(f"  MAE (test): {results['MAE_mean']:.4f} ± {results['MAE_std']:.4f}")
        logger.info(f"  RMSE (test): {results['RMSE_mean']:.4f} ± {results['RMSE_std']:.4f}")
        
        return results
    
    def save_model(self, filepath: Path = None) -> Path:
        
       # Save the trained model to disk.
        
        if not self.is_fitted:
            raise ValueError(f"{self.model_name} must be fitted before saving")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.model_name.lower().replace(' ', '_')}_{timestamp}.joblib"
            filepath = MODELS_DIR / filename
        
        model_data = {
            'model': self.model,
            'model_name': self.model_name,
            'feature_names': self.feature_names,
            'training_timestamp': self.training_timestamp,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Saved model to: {filepath}")
        
        return filepath
    
    def load_model(self, filepath: Path) -> 'BaseRegressor':
        
        # Load a trained model from disk.
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.model_name = model_data.get('model_name', self.model_name)
        self.feature_names = model_data.get('feature_names')
        self.training_timestamp = model_data.get('training_timestamp')
        self.is_fitted = True
        
        logger.info(f"Loaded model from: {filepath}")
        return self
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        if self.model is not None:
            return self.model.get_params()
        return {}


class ModelEvaluator:
    """
    Comprehensive model evaluation and visualization class.
    
    Provides methods for evaluating model performance, analyzing
    residuals, and generating evaluation visualizations.
    """
    
    def __init__(self, save_plots: bool = True, plot_dir: Path = None):
        
       # Initialize ModelEvaluator.
       
        self.save_plots = save_plots
        self.plot_dir = plot_dir or PLOTS_DIR
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info("ModelEvaluator initialized")
    
    def _save_figure(self, fig: plt.Figure, name: str) -> Optional[Path]:
        """Save figure to disk with timestamp."""
        if self.save_plots:
            filename = f"{name}_{self.evaluation_timestamp}.png"
            filepath = self.plot_dir / filename
            fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            logger.info(f"Saved plot: {filepath}")
            plt.close(fig)
            return filepath
        plt.close(fig)
        return None
    
    def compute_all_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        
       # Compute comprehensive evaluation metrics.
        
        metrics = {
            'R2': r2_score(y_true, y_pred),
            'Adjusted_R2': None,  # Needs n_features, computed separately if needed
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'Max_Error': np.max(np.abs(y_true - y_pred)),
            'Median_AE': np.median(np.abs(y_true - y_pred))
        }
        
        # Additional custom metrics
        residuals = y_true - y_pred
        metrics['Residual_Std'] = np.std(residuals)
        metrics['Residual_Skew'] = float(pd.Series(residuals).skew())
        
        return metrics
    
    def compute_adjusted_r2(
        self, 
        r2: float, 
        n_samples: int, 
        n_features: int
    ) -> float:
        
        # Compute adjusted R² score.
        
       
        if n_samples - n_features - 1 <= 0:
            return r2
        return 1 - (1 - r2) * (n_samples - 1) / (n_samples - n_features - 1)
    
    def plot_actual_vs_predicted(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Optional[Path]:
        
       # Plot actual vs predicted values scatter plot.
       
        logger.info("Generating actual vs predicted plot...")
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='white', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
               linewidth=2, label='Perfect Prediction')
        
        # Add metrics annotation
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        textstr = f'$R^2$ = {r2:.4f}\nRMSE = {rmse:,.2f}\nMAE = {mae:,.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props)
        
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'{model_name}: Actual vs Predicted', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_figure(fig, f'actual_vs_predicted_{model_name.lower().replace(" ", "_")}')
    
    def plot_residuals(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Optional[Path]:
        
       # Generate residual diagnostic plots.
       
        logger.info("Generating residual diagnostic plots...")
        
        residuals = y_true - y_pred
        standardized_residuals = (residuals - residuals.mean()) / residuals.std()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.5, edgecolors='white', linewidth=0.3)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Predicted Values', fontsize=11)
        axes[0, 0].set_ylabel('Residuals', fontsize=11)
        axes[0, 0].set_title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
        
        # 2. Residual distribution
        axes[0, 1].hist(residuals, bins=50, density=True, alpha=0.7, 
                       color='steelblue', edgecolor='white')
        
        # Add normal curve overlay
        from scipy import stats
        x_range = np.linspace(residuals.min(), residuals.max(), 100)
        axes[0, 1].plot(x_range, stats.norm.pdf(x_range, residuals.mean(), residuals.std()),
                       'r-', linewidth=2, label='Normal Distribution')
        axes[0, 1].set_xlabel('Residuals', fontsize=11)
        axes[0, 1].set_ylabel('Density', fontsize=11)
        axes[0, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        
        # 3. Q-Q Plot
        stats.probplot(standardized_residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (Standardized Residuals)', fontsize=12, fontweight='bold')
        
        # 4. Scale-Location plot (sqrt of standardized residuals)
        sqrt_std_residuals = np.sqrt(np.abs(standardized_residuals))
        axes[1, 1].scatter(y_pred, sqrt_std_residuals, alpha=0.5, edgecolors='white', linewidth=0.3)
        
        # Add lowess trend line
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            sorted_indices = np.argsort(y_pred)
            lowess_result = lowess(sqrt_std_residuals[sorted_indices], 
                                   y_pred[sorted_indices], frac=0.3)
            axes[1, 1].plot(lowess_result[:, 0], lowess_result[:, 1], 
                           'r-', linewidth=2, label='LOWESS')
        except ImportError:
            pass
        
        axes[1, 1].set_xlabel('Predicted Values', fontsize=11)
        axes[1, 1].set_ylabel('√|Standardized Residuals|', fontsize=11)
        axes[1, 1].set_title('Scale-Location Plot', fontsize=12, fontweight='bold')
        
        fig.suptitle(f'{model_name}: Residual Diagnostics', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return self._save_figure(fig, f'residuals_{model_name.lower().replace(" ", "_")}')
    
    def plot_prediction_error(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        model_name: str = "Model"
    ) -> Optional[Path]:
        
        # Plot prediction error distribution.
       
        logger.info("Generating prediction error plot...")
        
        errors = y_true - y_pred
        percentage_errors = (errors / y_true) * 100
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Absolute error distribution
        axes[0].hist(np.abs(errors), bins=50, alpha=0.7, 
                    color='coral', edgecolor='white')
        axes[0].axvline(np.mean(np.abs(errors)), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(np.abs(errors)):,.2f}')
        axes[0].axvline(np.median(np.abs(errors)), color='green', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(np.abs(errors)):,.2f}')
        axes[0].set_xlabel('Absolute Error', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Distribution of Absolute Errors', fontsize=12, fontweight='bold')
        axes[0].legend()
        
        # Percentage error distribution (filtered to reasonable range)
        pct_errors_filtered = percentage_errors[(percentage_errors > -100) & (percentage_errors < 100)]
        axes[1].hist(pct_errors_filtered, bins=50, alpha=0.7, 
                    color='steelblue', edgecolor='white')
        axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1].set_xlabel('Percentage Error (%)', fontsize=11)
        axes[1].set_ylabel('Frequency', fontsize=11)
        axes[1].set_title('Distribution of Percentage Errors', fontsize=12, fontweight='bold')
        
        fig.suptitle(f'{model_name}: Prediction Errors', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return self._save_figure(fig, f'prediction_errors_{model_name.lower().replace(" ", "_")}')
    
    def generate_evaluation_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        n_features: int = None
    ) -> Dict[str, Any]:
        
       # Generate complete evaluation report with metrics and plots.
        
        logger.info(f"Generating evaluation report for {model_name}...")
        
        report = {
            'model_name': model_name,
            'timestamp': self.evaluation_timestamp,
            'n_samples': len(y_true),
            'metrics': self.compute_all_metrics(y_true, y_pred),
            'plots': []
        }
        
        # Compute adjusted R² if n_features provided
        if n_features is not None:
            report['n_features'] = n_features
            report['metrics']['Adjusted_R2'] = self.compute_adjusted_r2(
                report['metrics']['R2'], len(y_true), n_features
            )
        
        # Generate essential plots only (actual vs predicted + residual diagnostics)
        plot_methods = [
            (self.plot_actual_vs_predicted, (y_true, y_pred, model_name)),
            (self.plot_residuals, (y_true, y_pred, model_name)),
        ]
        
        for method, args in plot_methods:
            try:
                plot_path = method(*args)
                if plot_path:
                    report['plots'].append(plot_path)
            except Exception as e:
                logger.error(f"Error generating plot: {e}")
        
        logger.info(f"Evaluation report complete for {model_name}")
        return report
    
    def format_metrics_table(
        self, 
        metrics: Dict[str, float], 
        model_name: str = "Model"
    ) -> pd.DataFrame:
        
        # Format metrics as a clean DataFrame.
        
        df = pd.DataFrame({
            'Metric': list(metrics.keys()),
            'Value': list(metrics.values())
        })
        df['Model'] = model_name
        return df[['Model', 'Metric', 'Value']]
