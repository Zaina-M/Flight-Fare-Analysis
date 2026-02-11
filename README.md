# Flight Price Prediction - Bangladesh 

A production-ready machine learning project for predicting flight ticket prices in Bangladesh.

##  Project Overview

This project implements a complete ML pipeline for predicting flight prices based on route, airline, and travel date. It follows industry best practices for reproducibility, maintainability, and production deployment.

**Business Goal:** Help airlines and travel platforms estimate ticket prices for pricing strategy and dynamic recommendations.

##  Project Structure

```
Flight Analysis/
├── main.py                     # Main pipeline entry point
├── app.py                      # Streamlit web application
├── requirements.txt            # Project dependencies
├── README.md                   # This file
│
├── config/
│   ├── __init__.py
│   └── config.py              # Centralized configuration
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_loader.py     # Data loading and cleaning
│   │
│   ├── eda/
│   │   ├── __init__.py
│   │   └── eda.py             # Exploratory data analysis
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── feature_engineering.py  # Feature engineering
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py      # Base model class
│   │   ├── regressors.py      # All regression models
│   │   └── model_comparison.py # Model comparison utilities
│   │
│   └── utils/
│       ├── __init__.py
│       └── logger.py          # Logging utilities
│
├── data/
│   ├── raw/                   # Raw data files
│   └── processed/             # Cleaned data files
│
├── visualization/                    # Saved model files
├── outputs/
│   ├── plots/                 # Generated visualizations
│   └── reports/              # Analysis reports
│
└── logs/                      # Log files
```

##  Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

1. Go to [Kaggle](https://www.kaggle.com)
2. Search for "Flight Price Dataset of Bangladesh"
3. Download `Flight_Price_Dataset_of_Bangladesh.csv`
4. Place it in `data/raw/`

### 3. Run the Pipeline

```bash
# Run the complete ML pipeline
python main.py

# Run specific steps
python main.py --step data      # Only data loading/cleaning
python main.py --step eda       # Only EDA
python main.py --skip-eda       # Skip EDA (faster)
python main.py --no-tune        # Skip hyperparameter tuning (faster)
```

### 4. Launch Web App (Stretch Challenge)

```bash
streamlit run app.py
```

##  Pipeline Steps

### Step 1: Data Understanding
- Load dataset and inspect structure
- Identify missing values, outliers, and data types
- Document assumptions and limitations

### Step 2: Data Cleaning & Feature Engineering
- Drop irrelevant columns
- Handle missing values (median/mode imputation)
- Normalize city names
- Create temporal features (Month, Day, Season, etc.)
- Encode categorical variables (One-Hot Encoding)
- Scale numerical features (StandardScaler)

### Step 3: Exploratory Data Analysis
- Descriptive statistics
- Distribution visualizations
- Correlation heatmap
- Fare analysis by airline, route, season

### Step 4: Baseline Model
- Linear Regression as baseline
- Evaluate using R², MAE, RMSE
- Analyze residuals

### Step 5: Advanced Modeling
- Multiple models: Ridge, Lasso, Decision Tree, Random Forest, Gradient Boosting
- Hyperparameter tuning with GridSearchCV/RandomizedSearchCV
- Cross-validation comparison
- Bias-variance analysis

### Step 6: Interpretation
- Feature importance analysis
- Coefficient interpretation
- Business insights generation

##  Models Implemented

| Model | Description |
|-------|-------------|
| Linear Regression | Baseline model |
| Ridge Regression | L2 regularization |
| Lasso Regression | L1 regularization (feature selection) |
| Decision Tree | Non-linear, interpretable |
| Random Forest | Ensemble of trees, reduces overfitting |
| Gradient Boosting | Sequential ensemble, often best performance |

##  Evaluation Metrics

- **R² (Coefficient of Determination)**: Variance explained by model
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Squared Error)**: Penalizes large errors
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error

##  Output Files

After running the pipeline:

- **Plots**: `outputs/plots/` - All visualizations
- **Reports**: `outputs/reports/` - KPI and comparison reports
- **Models**: `models/` - Saved trained models
- **Logs**: `logs/` - Execution logs

##  Configuration

All configuration is centralized in `config/config.py`:

```python
# Key parameters
test_size = 0.2          # Train-test split ratio
random_state = 42        # Reproducibility seed
cv_folds = 5             # Cross-validation folds
```

