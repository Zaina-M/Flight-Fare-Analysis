"""
Model Comparison and Evaluation Module.

This module provides utilities for comparing multiple models,
generating comparison reports, and visualizing performance differences.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import model_config, PLOTS_DIR, REPORTS_DIR, MODELS_DIR
from src.models.base_model import BaseRegressor, ModelEvaluator
from src.utils.logger import get_logger

logger = get_logger(__name__)

FIGURE_DPI = 150


class ModelComparison:
    """
    Comprehensive model comparison framework.
    
    Compares multiple models using consistent evaluation methodology
    and generates comparison reports and visualizations.
    """
    
    def __init__(
        self,
        models: Dict[str, BaseRegressor] = None,
        save_plots: bool = True,
        plot_dir: Path = None,
        run_id: str = None
    ):
        
       # Initialize ModelComparison.
      
        self.models = models or {}
        self.results = {}
        self.cv_results = {}
        self.save_plots = save_plots
        self.comparison_timestamp = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use run subfolder
        base_dir = plot_dir or PLOTS_DIR
        self.plot_dir = base_dir / f"run_{self.comparison_timestamp}"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"ModelComparison initialized with {len(self.models)} models")
    
    def add_model(self, name: str, model: BaseRegressor) -> None:
        """Add a model to the comparison."""
        self.models[name] = model
        logger.info(f"Added model: {name}")
    
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
    
    def train_all_models(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray]
    ) -> Dict[str, BaseRegressor]:
        
        # Train all models in the comparison.
        
        logger.info("=" * 60)
        logger.info("Training All Models")
        logger.info("=" * 60)
        
        for name, model in self.models.items():
            try:
                logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
                logger.info(f"Successfully trained {name}")
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
        
        return self.models
    
    def evaluate_all_models(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray]
    ) -> pd.DataFrame:
        
        # Evaluate all trained models on test data.
     
        logger.info("=" * 60)
        logger.info("Evaluating All Models")
        logger.info("=" * 60)
        
        results_list = []
        
        for name, model in self.models.items():
            if not model.is_fitted:
                logger.warning(f"Skipping {name} - model not fitted")
                continue
            
            try:
                metrics = model.evaluate(X_test, y_test)
                metrics['Model'] = name
                
                # Store predictions for later visualization
                self.results[name] = {
                    'predictions': model.predict(X_test),
                    'metrics': metrics
                }
                
                results_list.append(metrics)
                logger.info(f"Evaluated {name}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.2f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {name}: {e}")
        
        if not results_list:
            logger.warning("No models were successfully evaluated")
            return pd.DataFrame()
        
        # Create comparison DataFrame
        results_df = pd.DataFrame(results_list)
        cols = ['Model', 'R2', 'MAE', 'RMSE', 'MSE', 'MAPE']
        results_df = results_df[[c for c in cols if c in results_df.columns]]
        results_df = results_df.sort_values('R2', ascending=False).reset_index(drop=True)
        
        return results_df
    
    def cross_validate_all_models(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv: int = None
    ) -> pd.DataFrame:
        
        # Perform cross-validation for all models.
        
        cv = cv or model_config.cv_folds
        
        logger.info("=" * 60)
        logger.info(f"Cross-Validating All Models ({cv} folds)")
        logger.info("=" * 60)
        
        cv_results_list = []
        
        for name, model in self.models.items():
            try:
                cv_result = model.cross_validate(X, y, cv=cv)
                cv_result['Model'] = name
                self.cv_results[name] = cv_result
                
                cv_results_list.append({
                    'Model': name,
                    'R2_Mean': cv_result['R2_mean'],
                    'R2_Std': cv_result['R2_std'],
                    'MAE_Mean': cv_result['MAE_mean'],
                    'MAE_Std': cv_result['MAE_std'],
                    'RMSE_Mean': cv_result['RMSE_mean'],
                    'RMSE_Std': cv_result['RMSE_std'],
                    'Train_R2_Mean': cv_result['train_R2_mean']
                })
                
            except Exception as e:
                logger.error(f"Error in cross-validation for {name}: {e}")
        
        if not cv_results_list:
            return pd.DataFrame()
        
        cv_df = pd.DataFrame(cv_results_list)
        cv_df = cv_df.sort_values('R2_Mean', ascending=False).reset_index(drop=True)
        
        return cv_df
    
    def plot_model_comparison(
        self,
        results_df: pd.DataFrame,
        metric: str = 'R2'
    ) -> Optional[Path]:
        """
        Plot bar chart comparing models on a specific metric.
        
        Args:
            results_df: DataFrame with evaluation results
            metric: Metric to compare
        
        Returns:
            Path to saved plot
        """
        if results_df.empty or metric not in results_df.columns:
            logger.warning(f"Cannot plot comparison - no results for {metric}")
            return None
        
        logger.info(f"Generating model comparison plot for {metric}...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Sort by metric (higher is better for R2, lower for errors)
        ascending = metric not in ['R2']
        df_sorted = results_df.sort_values(metric, ascending=ascending)
        
        # Create color gradient
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_sorted)))
        if ascending:
            colors = colors[::-1]
        
        bars = ax.bar(
            df_sorted['Model'],
            df_sorted[metric],
            color=colors,
            edgecolor='white',
            linewidth=2
        )
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}' if metric == 'R2' else f'{height:,.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'Model Comparison: {metric}', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add horizontal line for mean
        mean_val = results_df[metric].mean()
        ax.axhline(mean_val, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_val:.4f}' if metric == 'R2' else f'Mean: {mean_val:,.2f}')
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        return self._save_figure(fig, f'model_comparison_{metric.lower()}')
    
    def plot_all_metrics_comparison(
        self,
        results_df: pd.DataFrame
    ) -> Optional[Path]:
        """
        Plot comprehensive comparison across all metrics.
        
        Args:
            results_df: DataFrame with evaluation results
        
        Returns:
            Path to saved plot
        """
        if results_df.empty:
            return None
        
        logger.info("Generating comprehensive metrics comparison plot...")
        
        metrics = ['R2', 'MAE', 'RMSE', 'MAPE']
        available_metrics = [m for m in metrics if m in results_df.columns]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(available_metrics):
            ax = axes[idx]
            
            ascending = metric not in ['R2']
            df_sorted = results_df.sort_values(metric, ascending=ascending)
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_sorted)))
            if ascending:
                colors = colors[::-1]
            
            bars = ax.barh(df_sorted['Model'], df_sorted[metric], color=colors)
            
            ax.set_xlabel(metric, fontsize=11)
            ax.set_title(f'{metric}', fontsize=12, fontweight='bold')
            
            # Add value labels
            for bar in bars:
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2,
                       f'{width:.4f}' if metric == 'R2' else f'{width:,.2f}',
                       ha='left', va='center', fontsize=9)
        
        # Hide empty subplots
        for idx in range(len(available_metrics), len(axes)):
            axes[idx].set_visible(False)
        
        fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return self._save_figure(fig, 'model_comparison_all_metrics')
    
    def plot_cv_comparison(
        self,
        cv_df: pd.DataFrame
    ) -> Optional[Path]:
        """
        Plot cross-validation results with error bars.
        
        Args:
            cv_df: DataFrame with cross-validation results
        
        Returns:
            Path to saved plot
        """
        if cv_df.empty:
            return None
        
        logger.info("Generating cross-validation comparison plot...")
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        metrics = [('R2', 'R2_Mean', 'R2_Std'), 
                   ('MAE', 'MAE_Mean', 'MAE_Std'),
                   ('RMSE', 'RMSE_Mean', 'RMSE_Std')]
        
        for idx, (metric_name, mean_col, std_col) in enumerate(metrics):
            ax = axes[idx]
            
            # Sort appropriately
            ascending = metric_name not in ['R2']
            df_sorted = cv_df.sort_values(mean_col, ascending=ascending)
            
            colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_sorted)))
            if ascending:
                colors = colors[::-1]
            
            ax.barh(
                df_sorted['Model'],
                df_sorted[mean_col],
                xerr=df_sorted[std_col],
                color=colors,
                capsize=5,
                edgecolor='white'
            )
            
            ax.set_xlabel(f'{metric_name} (mean ± std)', fontsize=11)
            ax.set_title(f'Cross-Validation: {metric_name}', fontsize=12, fontweight='bold')
        
        fig.suptitle('Cross-Validation Performance Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        return self._save_figure(fig, 'cv_comparison')
    
    def plot_predictions_comparison(
        self,
        y_test: Union[pd.Series, np.ndarray],
        top_n: int = 1
    ) -> Optional[Path]:
        """
        Plot Predicted vs Actual Fares scatter plot for best model.
        
        Args:
            y_test: True target values
            top_n: Number of top models to plot (1 = best model only)
        
        Returns:
            Path to saved plot
        """
        if not self.results:
            logger.warning("No results available for predictions comparison")
            return None
        
        logger.info("Generating Predicted vs Actual Fares scatter plot...")
        
        # Get best model
        sorted_models = sorted(
            self.results.items(),
            key=lambda x: x[1]['metrics']['R2'],
            reverse=True
        )[:top_n]
        
        if not sorted_models:
            logger.warning("No models found for predictions comparison")
            return None
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_true = np.array(y_test)
        name, result = sorted_models[0]
        y_pred = result['predictions']
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, edgecolors='white', linewidth=0.5, s=30, c='steelblue')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        # Add metrics annotation
        r2 = result['metrics']['R2']
        rmse = result['metrics']['RMSE']
        mae = result['metrics']['MAE']
        textstr = f'Model: {name}\n$R^2$ = {r2:.4f}\nRMSE = {rmse:,.2f}\nMAE = {mae:,.2f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
               verticalalignment='top', bbox=props)
        
        ax.set_xlabel('Actual Fares', fontsize=12)
        ax.set_ylabel('Predicted Fares', fontsize=12)
        ax.set_title('Predicted vs Actual Fares', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._save_figure(fig, 'predicted_vs_actual')
    
    def plot_bias_variance_tradeoff(
        self,
        cv_df: pd.DataFrame
    ) -> Optional[Path]:
        """
        Visualize bias-variance tradeoff across models.
        
        Args:
            cv_df: DataFrame with cross-validation results
        
        Returns:
            Path to saved plot
        """
        if cv_df.empty or 'Train_R2_Mean' not in cv_df.columns:
            return None
        
        logger.info("Generating bias-variance tradeoff plot...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(cv_df))
        width = 0.35
        
        # Training vs Test R²
        bars1 = ax.bar(x - width/2, cv_df['Train_R2_Mean'], width, 
                      label='Training R²', color='steelblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, cv_df['R2_Mean'], width, 
                      label='Test R² (CV)', color='coral', alpha=0.8)
        
        # Add error bars for test
        ax.errorbar(x + width/2, cv_df['R2_Mean'], yerr=cv_df['R2_Std'],
                   fmt='none', color='black', capsize=3)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('R² Score', fontsize=12)
        ax.set_title('Bias-Variance Tradeoff: Training vs Test Performance',
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(cv_df['Model'], rotation=45, ha='right')
        ax.legend(fontsize=10)
        
        # Annotate overfitting
        for idx, row in cv_df.iterrows():
            gap = row['Train_R2_Mean'] - row['R2_Mean']
            if gap > 0.1:  # Significant overfitting
                ax.annotate('', xy=(idx, row['Train_R2_Mean']),
                           fontsize=14, ha='center', va='bottom')
        
        plt.tight_layout()
        return self._save_figure(fig, 'bias_variance_tradeoff')
    
    def get_best_model(
        self,
        results_df: pd.DataFrame,
        metric: str = 'R2',
        higher_is_better: bool = True
    ) -> Tuple[str, BaseRegressor]:
        """
        Get the best performing model.
        
        Args:
            results_df: DataFrame with evaluation results
            metric: Metric to use for selection
            higher_is_better: Whether higher metric values are better
        
        Returns:
            Tuple of (model name, model instance)
        """
        if results_df.empty:
            raise ValueError("No results available for model selection")
        
        if higher_is_better:
            best_idx = results_df[metric].idxmax()
        else:
            best_idx = results_df[metric].idxmin()
        
        best_name = results_df.loc[best_idx, 'Model']
        best_model = self.models[best_name]
        
        logger.info(f"Best model: {best_name} ({metric}={results_df.loc[best_idx, metric]:.4f})")
        
        return best_name, best_model
    
    def generate_comparison_report(
        self,
        results_df: pd.DataFrame,
        cv_df: pd.DataFrame = None,
        save_report: bool = True
    ) -> str:
        """
        Generate comprehensive comparison report.
        
        Args:
            results_df: DataFrame with evaluation results
            cv_df: DataFrame with cross-validation results
            save_report: Whether to save report to file
        
        Returns:
            Report string
        """
        report_lines = [
            "=" * 70,
            "MODEL COMPARISON REPORT",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            "",
        ]
        
        # Test Set Results
        report_lines.extend([
            " TEST SET PERFORMANCE",
            "-" * 50,
        ])
        
        if not results_df.empty:
            for _, row in results_df.iterrows():
                report_lines.append(f"\n{row['Model']}:")
                report_lines.append(f"  R²: {row['R2']:.4f}")
                report_lines.append(f"  MAE: {row['MAE']:,.2f}")
                report_lines.append(f"  RMSE: {row['RMSE']:,.2f}")
                if 'MAPE' in row:
                    report_lines.append(f"  MAPE: {row['MAPE']:.2f}%")
        
        report_lines.append("")
        
        # Cross-Validation Results
        if cv_df is not None and not cv_df.empty:
            report_lines.extend([
                " CROSS-VALIDATION PERFORMANCE",
                "-" * 50,
            ])
            
            for _, row in cv_df.iterrows():
                report_lines.append(f"\n{row['Model']}:")
                report_lines.append(f"  R² (CV): {row['R2_Mean']:.4f} ± {row['R2_Std']:.4f}")
                report_lines.append(f"  MAE (CV): {row['MAE_Mean']:,.2f} ± {row['MAE_Std']:.2f}")
                report_lines.append(f"  RMSE (CV): {row['RMSE_Mean']:,.2f} ± {row['RMSE_Std']:.2f}")
            
            report_lines.append("")
        
        # Best Model
        if not results_df.empty:
            best_model = results_df.loc[results_df['R2'].idxmax()]
            report_lines.extend([
                "🏆 BEST MODEL",
                "-" * 50,
                f"  Model: {best_model['Model']}",
                f"  R²: {best_model['R2']:.4f}",
                f"  RMSE: {best_model['RMSE']:,.2f}",
                "",
            ])
        
        report_lines.extend([
            "=" * 70,
        ])
        
        report = "\n".join(report_lines)
        
        if save_report:
            filepath = REPORTS_DIR / f"model_comparison_{self.comparison_timestamp}.txt"
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Saved comparison report to: {filepath}")
        
        return report
    
    def save_best_model(
        self,
        results_df: pd.DataFrame,
        save_dir: Path = None
    ) -> Path:
        """
        Save the best performing model to disk.
        
        Args:
            results_df: DataFrame with evaluation results
            save_dir: Directory to save model
        
        Returns:
            Path to saved model
        """
        save_dir = save_dir or MODELS_DIR
        
        best_name, best_model = self.get_best_model(results_df)
        
        filepath = save_dir / f"best_model_{self.comparison_timestamp}.joblib"
        best_model.save_model(filepath)
        
        return filepath
    
    def run_full_comparison(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        run_cv: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete model comparison pipeline.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training target
            y_test: Test target
            run_cv: Whether to run cross-validation
        
        Returns:
            Dictionary with all comparison results
        """
        logger.info("=" * 70)
        logger.info("RUNNING FULL MODEL COMPARISON")
        logger.info("=" * 70)
        
        results = {
            'timestamp': self.comparison_timestamp,
            'n_models': len(self.models),
            'n_train': len(y_train),
            'n_test': len(y_test),
            'plots': []
        }
        
        # Train all models
        self.train_all_models(X_train, y_train)
        
        # Evaluate on test set
        results['test_results'] = self.evaluate_all_models(X_test, y_test)
        
        # Cross-validation
        if run_cv:
            results['cv_results'] = self.cross_validate_all_models(X_train, y_train)
        
        # Generate Predicted vs Actual plot (Plot 6 of 6 required visualizations)
        try:
            plot_path = self.plot_predictions_comparison(y_test)
            if plot_path:
                results['plots'].append(plot_path)
        except Exception as e:
            logger.error(f"Error generating predicted vs actual plot: {e}")
        
        # Generate report
        results['report'] = self.generate_comparison_report(
            results['test_results'],
            results.get('cv_results')
        )
        
        # Save best model
        try:
            results['best_model_path'] = self.save_best_model(results['test_results'])
        except Exception as e:
            logger.error(f"Error saving best model: {e}")
        
        logger.info("=" * 70)
        logger.info("MODEL COMPARISON COMPLETE")
        logger.info("=" * 70)
        
        return results


def compare_models(
    models: Dict[str, BaseRegressor],
    X_train: Union[pd.DataFrame, np.ndarray],
    X_test: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    run_cv: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for comparing multiple models.
    
    Args:
        models: Dictionary of model name to model instance
        X_train: Training features
        X_test: Test features
        y_train: Training target
        y_test: Test target
        run_cv: Whether to run cross-validation
    
    Returns:
        Dictionary with comparison results
    """
    comparison = ModelComparison(models=models)
    return comparison.run_full_comparison(X_train, X_test, y_train, y_test, run_cv)
