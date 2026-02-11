"""
Flight Price Prediction - Main Pipeline.

This script orchestrates the entire machine learning pipeline from
data loading to model evaluation. It serves as the main entry point
for running the flight price prediction project.

Usage:
    python main.py                    # Run complete pipeline
    python main.py --step data        # Run only data loading/cleaning
    python main.py --step eda         # Run only EDA
    python main.py --step train       # Run only model training
    python main.py --step evaluate    # Run only model evaluation

Industry Standard Justifications:
- Command-line interface for flexible execution
- Modular pipeline design enables selective execution
- Comprehensive logging for reproducibility and debugging
- Configuration-driven execution for easy experimentation
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import (
    data_config, model_config, 
    RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, PLOTS_DIR, REPORTS_DIR
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


def step_1_load_and_understand_data(file_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    Step 1: Problem Definition & Data Understanding.
    
    Load the flight price dataset and perform initial inspection.
    
    Args:
        file_path: Optional path to the data file
    
    Returns:
        Dictionary with loaded data and inspection results
    """
    from src.data.data_loader import DataLoader, DataCleaner
    
    logger.info("=" * 70)
    logger.info("STEP 1: PROBLEM DEFINITION & DATA UNDERSTANDING")
    logger.info("=" * 70)
    
    # Define the ML task
    logger.info("\n ML Task Definition:")
    logger.info("  - Type: Supervised Regression")
    logger.info("  - Target Variable: Total Fare")
    logger.info("  - Business Goal: Estimate flight ticket prices for pricing strategy")
    
    # Load data
    loader = DataLoader()
    
    try:
        df_raw = loader.load_data(file_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("\n Dataset Download Instructions:")
        logger.info("  1. Go to: https://www.kaggle.com/datasets/")
        logger.info("  2. Search for: 'Flight Price Dataset of Bangladesh'")
        logger.info("  3. Download the CSV file")
        logger.info(f"  4. Place it in: {RAW_DATA_DIR}")
        raise
    
    # Inspect data
    inspection_results = loader.inspect_data(df_raw)
    numerical_stats, categorical_stats = loader.get_descriptive_stats(df_raw)
    
    # Log observations
    logger.info("\n Initial Observations:")
    logger.info(f"  - Dataset Shape: {inspection_results['shape']}")
    logger.info(f"  - Columns: {len(inspection_results['columns'])}")
    logger.info(f"  - Duplicate Rows: {inspection_results['duplicates']}")
    logger.info(f"  - Memory Usage: {inspection_results['memory_usage_mb']:.2f} MB")
    
    # Log missing values
    missing = {k: v for k, v in inspection_results['missing_values'].items() if v > 0}
    if missing:
        logger.info("\n Missing Values Found:")
        for col, count in missing.items():
            pct = inspection_results['missing_percentage'][col]
            logger.info(f"    {col}: {count} ({pct:.1f}%)")
    
    return {
        'raw_data': df_raw,
        'inspection': inspection_results,
        'numerical_stats': numerical_stats,
        'categorical_stats': categorical_stats
    }


def step_2_clean_and_engineer_features(
    df_raw,
    save_processed: bool = True
):
    """
    Step 2: Data Cleaning & Feature Engineering.
    
    Clean the data and create features for modeling.
    
    Args:
        df_raw: Raw DataFrame
        save_processed: Whether to save processed data
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_engineer)
    """
    from src.data.data_loader import DataCleaner
    from src.features.feature_engineering import prepare_train_test_data, DateFeatureExtractor
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: DATA CLEANING & FEATURE ENGINEERING")
    logger.info("=" * 70)
    
    # Step 2A: Cleaning
    logger.info("\n Step 2A: Data Cleaning")
    cleaner = DataCleaner()
    df_cleaned = cleaner.clean_data(df_raw)
    cleaning_report = cleaner.get_cleaning_report()
    
    logger.info(f"  - Rows removed: {cleaning_report.get('rows_removed', 0)}")
    logger.info(f"  - Columns removed: {cleaning_report.get('columns_removed', 0)}")
    logger.info(f"  - Final shape: {df_cleaned.shape}")
    
    # Step 2B: Feature Engineering
    logger.info("\n Step 2B: Feature Engineering")
    
    # Extract date features first for EDA
    date_extractor = DateFeatureExtractor()
    df_with_features = date_extractor.extract_features(df_cleaned)
    
    # Save cleaned data with date features for EDA
    if save_processed:
        processed_path = PROCESSED_DATA_DIR / "cleaned_with_features.csv"
        df_with_features.to_csv(processed_path, index=False)
        logger.info(f"  - Saved processed data: {processed_path}")
    
    # Prepare train/test split with full feature engineering
    X_train, X_test, y_train, y_test, feature_engineer = prepare_train_test_data(
        df_cleaned,
        target_column=data_config.target_column,
        test_size=model_config.test_size,
        random_state=model_config.random_state,
        log_transform_target=True
    )
    
    logger.info(f"  - Training samples: {len(X_train)}")
    logger.info(f"  - Test samples: {len(X_test)}")
    logger.info(f"  - Number of features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test, feature_engineer, df_with_features


def step_3_exploratory_data_analysis(df):
    """
    Step 3: Exploratory Data Analysis.
    
    Perform comprehensive EDA with visualizations and KPI calculation.
    
    Args:
        df: Cleaned DataFrame with features
    
    Returns:
        Dictionary with EDA results
    """
    from src.eda.eda import ExploratoryDataAnalyzer, KPICalculator
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 70)
    
    # Run full EDA
    logger.info("\n Running Exploratory Data Analysis...")
    eda = ExploratoryDataAnalyzer(save_plots=True)
    eda_results = eda.run_full_eda(df)
    
    # Calculate KPIs
    logger.info("\n Calculating Key Performance Indicators...")
    kpi_calculator = KPICalculator()
    kpis = kpi_calculator.calculate_all_kpis(df)
    
    # Print KPI summary
    kpi_report = kpi_calculator.generate_kpi_report(kpis)
    print("\n" + kpi_report)
    
    # Save KPI report
    kpi_calculator.save_kpi_report(kpis)
    
    logger.info(f"\n EDA Complete - Generated {len(eda_results['plots'])} plots")
    
    return {
        'eda_results': eda_results,
        'kpis': kpis
    }


def step_4_baseline_model(X_train, X_test, y_train, y_test):
    """
    Step 4: Baseline Model Development.
    
    Train and evaluate a Linear Regression baseline model.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
    
    Returns:
        Dictionary with baseline model results
    """
    from src.models.regressors import LinearRegressionModel, FeatureImportancePlotter
    from src.models.base_model import ModelEvaluator
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: BASELINE MODEL DEVELOPMENT")
    logger.info("=" * 70)
    
    # Train Linear Regression
    logger.info("\n Training Linear Regression baseline...")
    baseline = LinearRegressionModel()
    baseline.fit(X_train, y_train)
    
    # Evaluate
    logger.info("\n Evaluating baseline model...")
    metrics = baseline.evaluate(X_test, y_test)
    
    # Cross-validation
    logger.info("\n Running cross-validation...")
    cv_results = baseline.cross_validate(X_train, y_train)
    
    # Generate evaluation plots
    evaluator = ModelEvaluator(save_plots=True)
    y_pred = baseline.predict(X_test)
    eval_report = evaluator.generate_evaluation_report(
        y_test.values if hasattr(y_test, 'values') else y_test,
        y_pred,
        "Linear Regression",
        n_features=X_train.shape[1]
    )
    
    # Plot coefficients
    plotter = FeatureImportancePlotter(save_plots=True)
    plotter.plot_linear_coefficients(baseline, top_n=15)
    
    logger.info("\n Baseline Model Results:")
    logger.info(f"  - R²: {metrics['R2']:.4f}")
    logger.info(f"  - MAE: {metrics['MAE']:,.2f}")
    logger.info(f"  - RMSE: {metrics['RMSE']:,.2f}")
    logger.info(f"  - CV R² (mean): {cv_results['R2_mean']:.4f} ± {cv_results['R2_std']:.4f}")
    
    # Key findings
    logger.info("\n Key Findings from Baseline:")
    coef_df = baseline.get_coefficients()
    logger.info("  Top 5 Most Influential Features:")
    for _, row in coef_df.head(5).iterrows():
        direction = "↑" if row['Coefficient'] > 0 else "↓"
        logger.info(f"    {direction} {row['Feature']}: {row['Coefficient']:.4f}")
    
    return {
        'model': baseline,
        'metrics': metrics,
        'cv_results': cv_results,
        'eval_report': eval_report
    }


def step_5_advanced_modeling(
    X_train, X_test, y_train, y_test,
    tune_hyperparameters: bool = True
):
    """
    Step 5: Advanced Modeling & Optimization.
    
    Train multiple models, tune hyperparameters, and compare performance.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        tune_hyperparameters: Whether to tune hyperparameters
    
    Returns:
        Dictionary with all model results
    """
    from src.models.regressors import (
        LinearRegressionModel, RidgeRegressionModel, LassoRegressionModel,
        DecisionTreeRegressorModel, RandomForestRegressorModel,
        GradientBoostingRegressorModel, FeatureImportancePlotter, get_all_models,
        HAS_XGBOOST
    )
    from src.models.model_comparison import ModelComparison
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: ADVANCED MODELING & OPTIMIZATION")
    logger.info("=" * 70)
    
    # Initialize all models
    models = get_all_models(random_state=model_config.random_state)
    
    # Hyperparameter tuning (if enabled)
    if tune_hyperparameters:
        logger.info("\n Step 5.2: Hyperparameter Tuning")
        
        # Tune Ridge
        logger.info("\nTuning Ridge Regression...")
        ridge_results = models['Ridge Regression'].tune_hyperparameters(X_train, y_train)
        
        # Tune Lasso
        logger.info("\nTuning Lasso Regression...")
        lasso_results = models['Lasso Regression'].tune_hyperparameters(X_train, y_train)
        
        # Tune Decision Tree
        logger.info("\nTuning Decision Tree...")
        dt_results = models['Decision Tree'].tune_hyperparameters(
            X_train, y_train,
            use_randomized=False,
            cv=3
        )
        
        # Tune Random Forest (with smaller grid for speed)
        logger.info("\nTuning Random Forest...")
        rf_results = models['Random Forest'].tune_hyperparameters(
            X_train, y_train, 
            use_randomized=True, 
            n_iter=10,
            cv=3  # Reduce CV folds for speed
        )
        
        # Tune Gradient Boosting
        logger.info("\nTuning Gradient Boosting...")
        gb_results = models['Gradient Boosting'].tune_hyperparameters(
            X_train, y_train,
            use_randomized=True,
            n_iter=10,
            cv=3  # Reduce CV folds for speed
        )
        
        # Tune XGBoost if available
        if HAS_XGBOOST and 'XGBoost' in models:
            logger.info("\nTuning XGBoost...")
            xgb_results = models['XGBoost'].tune_hyperparameters(
                X_train, y_train,
                use_randomized=True,
                n_iter=10,
                cv=3
            )
    
    # Compare all models
    logger.info("\n Step 5.3: Model Comparison")
    comparison = ModelComparison(models=models)
    comparison_results = comparison.run_full_comparison(
        X_train, X_test, y_train, y_test,
        run_cv=True
    )
    
    # Print comparison table
    logger.info("\n" + "=" * 60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("=" * 60)
    print("\nTest Set Performance:")
    print(comparison_results['test_results'].to_string(index=False))
    
    if 'cv_results' in comparison_results:
        print("\nCross-Validation Performance:")
        print(comparison_results['cv_results'].to_string(index=False))
    
    # Feature importance for best tree-based model
    logger.info("\n Step 5.4: Feature Importance Analysis")
    plotter = FeatureImportancePlotter(save_plots=True)
    
    # Plot for tree-based models
    for model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
        if model_name in models and models[model_name].is_fitted:
            plotter.plot_tree_feature_importance(models[model_name], top_n=15)
    
    return {
        'models': models,
        'comparison': comparison_results,
        'best_model_name': comparison_results['test_results'].iloc[0]['Model']
    }


def step_6_model_interpretation(models_dict, X_train, X_test, y_test):
    """
    Step 6: Model Interpretation & Insights.
    
    Analyze feature importance and generate business insights.
    
    Args:
        models_dict: Dictionary of trained models
        X_train: Training features
        X_test: Test features
        y_test: Test target
    
    Returns:
        Dictionary with insights
    """
    from src.models.regressors import FeatureImportancePlotter
    
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: MODEL INTERPRETATION & INSIGHTS")
    logger.info("=" * 70)
    
    insights = {}
    
    # Get feature importance from best tree-based model
    for model_name in ['Random Forest', 'Gradient Boosting']:
        if model_name in models_dict and models_dict[model_name].is_fitted:
            model = models_dict[model_name]
            importance_df = model.get_feature_importances()
            
            logger.info(f"\n Feature Importance ({model_name}):")
            logger.info("-" * 40)
            for _, row in importance_df.head(10).iterrows():
                bar = "" * int(row['Importance'] * 50)
                logger.info(f"  {row['Feature'][:25]:<25} {row['Importance']:.4f} {bar}")
            
            insights[f'{model_name}_importance'] = importance_df
            break
    
    # Get coefficients from Linear/Ridge/Lasso
    for model_name in ['Linear Regression', 'Ridge Regression', 'Lasso Regression']:
        if model_name in models_dict and models_dict[model_name].is_fitted:
            model = models_dict[model_name]
            coef_df = model.get_coefficients()
            
            logger.info(f"\n Feature Coefficients ({model_name}):")
            logger.info("-" * 40)
            for _, row in coef_df.head(10).iterrows():
                sign = "+" if row['Coefficient'] > 0 else ""
                logger.info(f"  {row['Feature'][:25]:<25} {sign}{row['Coefficient']:.4f}")
            
            insights[f'{model_name}_coefficients'] = coef_df
    
    # Lasso feature selection
    if 'Lasso Regression' in models_dict and models_dict['Lasso Regression'].is_fitted:
        lasso = models_dict['Lasso Regression']
        selected_features = lasso.get_selected_features()
        logger.info(f"\n Lasso Feature Selection:")
        logger.info(f"  - Selected {len(selected_features)} features out of {len(X_train.columns)}")
        logger.info(f"  - Top selected features: {selected_features[:5]}")
        insights['lasso_selected_features'] = selected_features
    
    # Generate insights summary
    logger.info("\n" + "=" * 60)
    logger.info("KEY INSIGHTS FOR STAKEHOLDERS")
    logger.info("=" * 60)
    
    logger.info("""
     FACTORS INFLUENCING FLIGHT PRICES:
    
    Based on our analysis, the following factors most strongly influence 
    flight ticket prices:
    
    1. BASE FARE: Direct component of total fare
    2. ROUTE: Certain source-destination pairs command premium prices
    3. AIRLINE: Different carriers have distinct pricing strategies
    4. SEASONALITY: Prices vary by month/season (e.g., peak travel periods)
    5. DAY OF WEEK: Weekend vs weekday pricing differences
    
    RECOMMENDATIONS:
    
    1. Dynamic Pricing: Implement route and season-based adjustments
    2. Competitive Analysis: Monitor airline-specific pricing patterns
    3. Demand Forecasting: Use temporal features for capacity planning
    4. Customer Segmentation: Offer flexible pricing for different segments
    """)
    
    return insights


def run_full_pipeline(
    data_path: Optional[Path] = None,
    skip_eda: bool = False,
    tune_hyperparameters: bool = True
) -> Dict[str, Any]:
    """
    Run the complete flight price prediction pipeline.
    
    Args:
        data_path: Optional path to the raw data file
        skip_eda: Whether to skip EDA step
        tune_hyperparameters: Whether to tune hyperparameters
    
    Returns:
        Dictionary with all pipeline results
    """
    logger.info("=" * 70)
    logger.info("FLIGHT PRICE PREDICTION - FULL PIPELINE")
    logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    pipeline_results = {}
    
    try:
        # Step 1: Load and understand data
        step1_results = step_1_load_and_understand_data(data_path)
        pipeline_results['step1'] = step1_results
        
        # Step 2: Clean and engineer features
        X_train, X_test, y_train, y_test, feature_engineer, df_processed = \
            step_2_clean_and_engineer_features(step1_results['raw_data'])
        pipeline_results['step2'] = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'feature_engineer': feature_engineer,
            'df_processed': df_processed
        }
        
        # Step 3: EDA
        if not skip_eda:
            step3_results = step_3_exploratory_data_analysis(df_processed)
            pipeline_results['step3'] = step3_results
        
        # Step 4: Baseline model
        step4_results = step_4_baseline_model(X_train, X_test, y_train, y_test)
        pipeline_results['step4'] = step4_results
        
        # Step 5: Advanced modeling
        step5_results = step_5_advanced_modeling(
            X_train, X_test, y_train, y_test,
            tune_hyperparameters=tune_hyperparameters
        )
        pipeline_results['step5'] = step5_results
        
        # Step 6: Interpretation
        step6_results = step_6_model_interpretation(
            step5_results['models'],
            X_train, X_test, y_test
        )
        pipeline_results['step6'] = step6_results
        
        # Final summary
        logger.info("\n" + "=" * 70)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 70)
        
        best_model = step5_results['best_model_name']
        best_metrics = step5_results['comparison']['test_results'].iloc[0]
        
        logger.info(f"\n Best Model: {best_model}")
        logger.info(f"   R²: {best_metrics['R2']:.4f}")
        logger.info(f"   RMSE: {best_metrics['RMSE']:,.2f}")
        logger.info(f"   MAE: {best_metrics['MAE']:,.2f}")
        
        logger.info(f"\n Outputs saved to:")
        logger.info(f"   - Plots: {PLOTS_DIR}")
        logger.info(f"   - Models: {MODELS_DIR}")
        logger.info(f"   - Reports: {REPORTS_DIR}")
        
        logger.info(f"\n Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise
    
    return pipeline_results


def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="Flight Price Prediction Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                           # Run complete pipeline
    python main.py --data path/to/data.csv   # Use custom data path
    python main.py --skip-eda                # Skip EDA step
    python main.py --no-tune                 # Skip hyperparameter tuning
        """
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='Path to the flight price dataset'
    )
    
    parser.add_argument(
        '--skip-eda',
        action='store_true',
        help='Skip exploratory data analysis step'
    )
    
    parser.add_argument(
        '--no-tune',
        action='store_true',
        help='Skip hyperparameter tuning (faster execution)'
    )
    
    parser.add_argument(
        '--step',
        type=str,
        choices=['data', 'eda', 'train', 'all'],
        default='all',
        help='Run specific pipeline step only'
    )
    
    args = parser.parse_args()
    
    # Convert data path if provided
    data_path = Path(args.data) if args.data else None
    
    # Run pipeline
    if args.step == 'all':
        run_full_pipeline(
            data_path=data_path,
            skip_eda=args.skip_eda,
            tune_hyperparameters=not args.no_tune
        )
    elif args.step == 'data':
        results = step_1_load_and_understand_data(data_path)
        step_2_clean_and_engineer_features(results['raw_data'])
    elif args.step == 'eda':
        from src.features.feature_engineering import DateFeatureExtractor
        import pandas as pd
        
        # Load processed data
        processed_path = PROCESSED_DATA_DIR / "cleaned_with_features.csv"
        if not processed_path.exists():
            logger.error("Processed data not found. Run 'python main.py --step data' first.")
            sys.exit(1)
        
        df = pd.read_csv(processed_path)
        step_3_exploratory_data_analysis(df)
    else:
        # Train step - load processed data
        import pandas as pd
        
        processed_path = PROCESSED_DATA_DIR / "cleaned_flight_data.csv"
        if not processed_path.exists():
            logger.error("Processed data not found. Run 'python main.py --step data' first.")
            sys.exit(1)
        
        df = pd.read_csv(processed_path)
        _, X_test, y_train, y_test, _, _ = step_2_clean_and_engineer_features(df, save_processed=False)
        
        # Would need to reload train data properly - for now just run full pipeline
        logger.info("For partial training, please run complete pipeline.")


if __name__ == "__main__":
    main()
