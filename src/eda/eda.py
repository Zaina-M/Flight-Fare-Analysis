# Exploratory Data Analysis (EDA) Module.

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime
import warnings
import sys

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for production
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config.config import data_config, PLOTS_DIR, REPORTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Configure visualization style for consistency
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
FIGURE_DPI = 150
FIGURE_FORMAT = 'png'


class ExploratoryDataAnalyzer:
    # Comprehensive EDA class for flight price data.
    
    def __init__(self, save_plots: bool = True, plot_dir: Path = None):
        """
        Initialize ExploratoryDataAnalyzer.
        
        Args:
            save_plots: Whether to save plots to disk
            plot_dir: Directory for saving plots
        """
        self.save_plots = save_plots
        self.plot_dir = plot_dir or PLOTS_DIR
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"EDA Analyzer initialized. Plots will be saved to: {self.plot_dir}")
    
    def _save_figure(self, fig: plt.Figure, name: str) -> Optional[Path]:
        """
        Save figure to disk with timestamp.
        
        Args:
            fig: Matplotlib figure to save
            name: Base name for the file
        
        Returns:
            Path to saved figure or None
        """
        if self.save_plots:
            filename = f"{name}_{self.analysis_timestamp}.{FIGURE_FORMAT}"
            filepath = self.plot_dir / filename
            fig.savefig(filepath, dpi=FIGURE_DPI, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.info(f"Saved plot: {filepath}")
            plt.close(fig)
            return filepath
        plt.close(fig)
        return None
    
    def descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive descriptive statistics.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary containing various statistical summaries
        """
        logger.info("Computing descriptive statistics...")
        
        stats_dict = {}
        
        # Overall numerical statistics
        stats_dict['numerical_summary'] = df.describe().T
        
        # Percentile analysis for target variable
        if 'Total Fare' in df.columns:
            percentiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            stats_dict['fare_percentiles'] = pd.DataFrame({
                'Percentile': [f'{p*100:.0f}%' for p in percentiles],
                'Total Fare': [df['Total Fare'].quantile(p) for p in percentiles]
            })
        
        # Fare by airline
        if 'Airline' in df.columns and 'Total Fare' in df.columns:
            stats_dict['fare_by_airline'] = df.groupby('Airline')['Total Fare'].agg([
                'count', 'mean', 'std', 'min', 'median', 'max'
            ]).round(2).sort_values('mean', ascending=False)
        
        # Fare by source
        if 'Source' in df.columns and 'Total Fare' in df.columns:
            stats_dict['fare_by_source'] = df.groupby('Source')['Total Fare'].agg([
                'count', 'mean', 'std', 'min', 'median', 'max'
            ]).round(2).sort_values('mean', ascending=False)
        
        # Fare by destination
        if 'Destination' in df.columns and 'Total Fare' in df.columns:
            stats_dict['fare_by_destination'] = df.groupby('Destination')['Total Fare'].agg([
                'count', 'mean', 'std', 'min', 'median', 'max'
            ]).round(2).sort_values('mean', ascending=False)
        
        # Fare by season (if available)
        if 'Season' in df.columns and 'Total Fare' in df.columns:
            stats_dict['fare_by_season'] = df.groupby('Season')['Total Fare'].agg([
                'count', 'mean', 'std', 'min', 'median', 'max'
            ]).round(2).sort_values('mean', ascending=False)
        
        logger.info(f"Generated {len(stats_dict)} statistical summary tables")
        return stats_dict
    
    def correlation_analysis(
        self, 
        df: pd.DataFrame, 
        method: str = 'pearson'
    ) -> Tuple[pd.DataFrame, Optional[Path]]:
        """
        Compute and visualize correlation matrix.
        
        Args:
            df: Input DataFrame
            method: Correlation method ('pearson', 'spearman', 'kendall')
        
        Returns:
            Correlation matrix DataFrame and path to saved plot
        """
        logger.info(f"Computing {method} correlation matrix...")
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            logger.warning("Not enough numerical columns for correlation analysis")
            return pd.DataFrame(), None
        
        # Compute correlation
        corr_matrix = df[numerical_cols].corr(method=method)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f', 
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8},
            ax=ax
        )
        
        ax.set_title(f'Feature Correlation Heatmap ({method.capitalize()})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plot_path = self._save_figure(fig, 'correlation_heatmap')
        
        # Log high correlations with target
        if 'Total Fare' in numerical_cols:
            target_corr = corr_matrix['Total Fare'].drop('Total Fare').sort_values(
                key=abs, ascending=False
            )
            logger.info("Top correlations with Total Fare:")
            for feat, corr in target_corr.head(5).items():
                logger.info(f"  {feat}: {corr:.3f}")
        
        return corr_matrix, plot_path
    
    def plot_fare_distributions(self, df: pd.DataFrame) -> List[Path]:
        """
        Plot distribution of fare-related columns.
        
        Args:
            df: Input DataFrame
        
        Returns:
            List of paths to saved plots
        """
        logger.info("Generating fare distribution plots...")
        saved_plots = []
        
        fare_cols = ['Total Fare', 'Base Fare', 'Tax & Surcharge']
        available_cols = [col for col in fare_cols if col in df.columns]
        
        if not available_cols:
            logger.warning("No fare columns found for distribution plot")
            return saved_plots
        
        # Combined distribution plot
        fig, axes = plt.subplots(1, len(available_cols), figsize=(6*len(available_cols), 5))
        if len(available_cols) == 1:
            axes = [axes]
        
        for idx, col in enumerate(available_cols):
            data = df[col].dropna()
            
            # Histogram with KDE
            axes[idx].hist(data, bins=50, density=True, alpha=0.7, 
                          color='steelblue', edgecolor='white')
            
            # Add KDE line
            if len(data) > 10:
                try:
                    kde_x = np.linspace(data.min(), data.max(), 100)
                    kde = stats.gaussian_kde(data)
                    axes[idx].plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
                except Exception as e:
                    logger.warning(f"Could not compute KDE for {col}: {e}")
            
            # Add mean and median lines
            mean_val = data.mean()
            median_val = data.median()
            axes[idx].axvline(mean_val, color='green', linestyle='--', 
                             linewidth=2, label=f'Mean: {mean_val:,.0f}')
            axes[idx].axvline(median_val, color='orange', linestyle='--', 
                             linewidth=2, label=f'Median: {median_val:,.0f}')
            
            axes[idx].set_xlabel(col, fontsize=12)
            axes[idx].set_ylabel('Density', fontsize=12)
            axes[idx].set_title(f'Distribution of {col}', fontsize=13, fontweight='bold')
            axes[idx].legend(fontsize=9)
            
            # Add skewness annotation
            skewness = stats.skew(data)
            axes[idx].annotate(f'Skewness: {skewness:.2f}', 
                              xy=(0.95, 0.95), xycoords='axes fraction',
                              ha='right', va='top', fontsize=10,
                              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plot_path = self._save_figure(fig, 'fare_distributions')
        if plot_path:
            saved_plots.append(plot_path)
        
        return saved_plots
    
    def plot_fare_by_airline(self, df: pd.DataFrame) -> Optional[Path]:
        """
        Plot average fare by airline.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Path to saved plot
        """
        if 'Airline' not in df.columns or 'Total Fare' not in df.columns:
            logger.warning("Required columns not found for airline fare plot")
            return None
        
        logger.info("Generating fare by airline plot...")
        
        # Compute statistics
        airline_stats = df.groupby('Airline').agg({
            'Total Fare': ['mean', 'std', 'count']
        }).round(2)
        airline_stats.columns = ['Mean Fare', 'Std Fare', 'Count']
        airline_stats = airline_stats.sort_values('Mean Fare', ascending=True)
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        bars = ax.barh(
            airline_stats.index, 
            airline_stats['Mean Fare'],
            xerr=airline_stats['Std Fare'],
            color=plt.cm.viridis(np.linspace(0, 1, len(airline_stats))),
            edgecolor='white',
            capsize=3
        )
        
        # Add value labels
        for bar, count in zip(bars, airline_stats['Count']):
            width = bar.get_width()
            ax.text(width + airline_stats['Std Fare'].max() * 0.1, 
                   bar.get_y() + bar.get_height()/2,
                   f'{width:,.0f} (n={count})', 
                   ha='left', va='center', fontsize=9)
        
        ax.set_xlabel('Average Total Fare (BDT)', fontsize=12)
        ax.set_ylabel('Airline', fontsize=12)
        ax.set_title('Average Fare by Airline (with Standard Deviation)', 
                    fontsize=14, fontweight='bold')
        ax.axvline(df['Total Fare'].mean(), color='red', linestyle='--', 
                  linewidth=2, label=f"Overall Mean: {df['Total Fare'].mean():,.0f}")
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        return self._save_figure(fig, 'fare_by_airline')
    
    def plot_fare_variation_across_airlines(self, df: pd.DataFrame) -> Optional[Path]:
        """
        Create boxplot showing fare variation across airlines.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Path to saved plot
        """
        if 'Airline' not in df.columns or 'Total Fare' not in df.columns:
            logger.warning("Required columns not found for airline boxplot")
            return None
        
        logger.info("Generating airline fare boxplot...")
        
        # Order by median fare
        order = df.groupby('Airline')['Total Fare'].median().sort_values(ascending=False).index
        
        fig, ax = plt.subplots(figsize=(14, 6))
        
        sns.boxplot(
            data=df, 
            x='Airline', 
            y='Total Fare',
            order=order,
            palette='viridis',
            ax=ax,
            showfliers=True,
            flierprops={'markersize': 3, 'alpha': 0.5}
        )
        
        ax.set_xlabel('Airline', fontsize=12)
        ax.set_ylabel('Total Fare (BDT)', fontsize=12)
        ax.set_title('Fare Variation Across Airlines (Boxplot)', 
                    fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add mean line
        ax.axhline(df['Total Fare'].mean(), color='red', linestyle='--', 
                  linewidth=2, label=f"Overall Mean: {df['Total Fare'].mean():,.0f}")
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        return self._save_figure(fig, 'fare_boxplot_airlines')
    
    def plot_fare_by_season(self, df: pd.DataFrame) -> Optional[Path]:
        """
        Create boxplot showing fare variation across seasons.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Path to saved plot
        """
        if 'Season' not in df.columns or 'Total Fare' not in df.columns:
            logger.warning("Season or Total Fare column not found")
            return None
        
        logger.info("Generating seasonal fare boxplot...")
        
        # Define season order
        season_order = ['Winter', 'Summer', 'Monsoon', 'Autumn']
        available_seasons = [s for s in season_order if s in df['Season'].unique()]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.boxplot(
            data=df, 
            x='Season', 
            y='Total Fare',
            order=available_seasons,
            palette='coolwarm',
            ax=ax
        )
        
        ax.set_xlabel('Season', fontsize=12)
        ax.set_ylabel('Total Fare (BDT)', fontsize=12)
        ax.set_title('Fare Variation Across Seasons', fontsize=14, fontweight='bold')
        
        # Add statistical test results (Kruskal-Wallis)
        groups = [df[df['Season'] == s]['Total Fare'].dropna() for s in available_seasons]
        if all(len(g) > 5 for g in groups):
            try:
                stat, p_value = stats.kruskal(*groups)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                ax.annotate(f'Kruskal-Wallis p={p_value:.4f} ({significance})', 
                           xy=(0.02, 0.98), xycoords='axes fraction',
                           ha='left', va='top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            except Exception as e:
                logger.warning(f"Could not compute Kruskal-Wallis test: {e}")
        
        plt.tight_layout()
        return self._save_figure(fig, 'fare_by_season')
    
    def plot_fare_by_month(self, df: pd.DataFrame) -> Optional[Path]:
        """
        Plot average fare trend by month.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Path to saved plot
        """
        if 'Month' not in df.columns or 'Total Fare' not in df.columns:
            logger.warning("Month or Total Fare column not found")
            return None
        
        logger.info("Generating monthly fare trend plot...")
        
        monthly_stats = df.groupby('Month').agg({
            'Total Fare': ['mean', 'std', 'count']
        }).round(2)
        monthly_stats.columns = ['Mean Fare', 'Std Fare', 'Count']
        monthly_stats = monthly_stats.sort_index()
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Line plot with confidence interval
        ax.plot(monthly_stats.index, monthly_stats['Mean Fare'], 
               'b-o', linewidth=2, markersize=8, label='Mean Fare')
        ax.fill_between(
            monthly_stats.index,
            monthly_stats['Mean Fare'] - monthly_stats['Std Fare'],
            monthly_stats['Mean Fare'] + monthly_stats['Std Fare'],
            alpha=0.2, color='blue', label='±1 Std Dev'
        )
        
        # Add month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_names)
        
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Average Total Fare (BDT)', fontsize=12)
        ax.set_title('Average Fare Trend by Month', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Highlight peak months
        peak_month = monthly_stats['Mean Fare'].idxmax()
        ax.axvline(peak_month, color='red', linestyle='--', alpha=0.5)
        ax.annotate(f'Peak: {month_names[peak_month-1]}', 
                   xy=(peak_month, monthly_stats['Mean Fare'].max()),
                   xytext=(peak_month + 0.5, monthly_stats['Mean Fare'].max() * 1.02),
                   fontsize=10, color='red')
        
        plt.tight_layout()
        return self._save_figure(fig, 'fare_by_month')
    
    def plot_route_analysis(self, df: pd.DataFrame, top_n: int = 10) -> Optional[Path]:
        """
        Analyze and plot top routes by frequency and fare.
        
        Args:
            df: Input DataFrame
            top_n: Number of top routes to display
        
        Returns:
            Path to saved plot
        """
        if 'Source' not in df.columns or 'Destination' not in df.columns:
            logger.warning("Source or Destination column not found")
            return None
        
        logger.info(f"Generating route analysis plot (top {top_n})...")
        
        df = df.copy()
        df['Route'] = df['Source'] + ' → ' + df['Destination']
        
        route_stats = df.groupby('Route').agg({
            'Total Fare': ['mean', 'count']
        }).round(2)
        route_stats.columns = ['Mean Fare', 'Flight Count']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Top routes by frequency
        top_by_freq = route_stats.nlargest(top_n, 'Flight Count')
        axes[0].barh(top_by_freq.index, top_by_freq['Flight Count'], 
                    color='steelblue', edgecolor='white')
        axes[0].set_xlabel('Number of Flights', fontsize=12)
        axes[0].set_ylabel('Route', fontsize=12)
        axes[0].set_title(f'Top {top_n} Routes by Frequency', fontsize=13, fontweight='bold')
        for i, (route, row) in enumerate(top_by_freq.iterrows()):
            axes[0].text(row['Flight Count'] + top_by_freq['Flight Count'].max() * 0.02, 
                        i, f"{int(row['Flight Count'])}", va='center', fontsize=9)
        
        # Top routes by fare
        top_by_fare = route_stats.nlargest(top_n, 'Mean Fare')
        axes[1].barh(top_by_fare.index, top_by_fare['Mean Fare'], 
                    color='coral', edgecolor='white')
        axes[1].set_xlabel('Average Fare (BDT)', fontsize=12)
        axes[1].set_ylabel('Route', fontsize=12)
        axes[1].set_title(f'Top {top_n} Routes by Average Fare', fontsize=13, fontweight='bold')
        for i, (route, row) in enumerate(top_by_fare.iterrows()):
            axes[1].text(row['Mean Fare'] + top_by_fare['Mean Fare'].max() * 0.02, 
                        i, f"{row['Mean Fare']:,.0f}", va='center', fontsize=9)
        
        plt.tight_layout()
        return self._save_figure(fig, 'route_analysis')
    
    def run_full_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute streamlined EDA pipeline with consolidated, explanatory plots.
        
        Generates fewer but more informative multi-panel figures instead of
        many separate single-purpose plots.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary containing all EDA results and plot paths
        """
        logger.info("=" * 60)
        logger.info("Starting Exploratory Data Analysis")
        logger.info("=" * 60)
        
        results = {
            'timestamp': self.analysis_timestamp,
            'data_shape': df.shape,
            'statistics': {},
            'plots': []
        }
        
        # Generate statistics
        results['statistics'] = self.descriptive_statistics(df)
        
        # Correlation analysis (1 plot)
        corr_matrix, corr_plot = self.correlation_analysis(df)
        results['correlation_matrix'] = corr_matrix
        if corr_plot:
            results['plots'].append(corr_plot)
        
        # Consolidated pricing overview (1 multi-panel plot)
        try:
            overview_path = self._plot_pricing_overview(df)
            if overview_path:
                results['plots'].append(overview_path)
        except Exception as e:
            logger.error(f"Error in pricing overview: {e}")
        
        # Consolidated temporal analysis (1 multi-panel plot)
        try:
            temporal_path = self._plot_temporal_analysis(df)
            if temporal_path:
                results['plots'].append(temporal_path)
        except Exception as e:
            logger.error(f"Error in temporal analysis: {e}")
        
        # Route analysis (1 plot)
        try:
            route_path = self.plot_route_analysis(df, top_n=10)
            if route_path:
                results['plots'].append(route_path)
        except Exception as e:
            logger.error(f"Error in route analysis: {e}")
        
        logger.info("=" * 60)
        logger.info(f"EDA Complete. Generated {len(results['plots'])} consolidated plots")
        logger.info("=" * 60)
        
        return results
    
    def _plot_pricing_overview(self, df: pd.DataFrame) -> Optional[Path]:
        """
        Consolidated pricing overview: fare distribution + fare by airline + fare by season.
        One figure with 3 panels instead of 4 separate plots.
        """
        logger.info("Generating consolidated pricing overview...")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Panel 1: Fare distribution with KDE
        if 'Total Fare' in df.columns:
            data = df['Total Fare'].dropna()
            axes[0].hist(data, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='white')
            try:
                kde_x = np.linspace(data.min(), data.max(), 100)
                kde = stats.gaussian_kde(data)
                axes[0].plot(kde_x, kde(kde_x), 'r-', linewidth=2, label='KDE')
            except Exception:
                pass
            axes[0].axvline(data.mean(), color='green', linestyle='--', lw=2, label=f'Mean: {data.mean():,.0f}')
            axes[0].axvline(data.median(), color='orange', linestyle='--', lw=2, label=f'Median: {data.median():,.0f}')
            axes[0].set_xlabel('Total Fare (BDT)')
            axes[0].set_title('Fare Distribution', fontweight='bold')
            axes[0].legend(fontsize=8)
        
        # Panel 2: Boxplot by airline
        if 'Airline' in df.columns and 'Total Fare' in df.columns:
            order = df.groupby('Airline')['Total Fare'].median().sort_values(ascending=False).index
            sns.boxplot(data=df, x='Total Fare', y='Airline', order=order,
                       palette='viridis', ax=axes[1], showfliers=False)
            axes[1].set_title('Fare by Airline', fontweight='bold')
            axes[1].set_xlabel('Total Fare (BDT)')
        
        # Panel 3: Boxplot by season
        if 'Season' in df.columns and 'Total Fare' in df.columns:
            season_order = [s for s in ['Winter', 'Summer', 'Monsoon', 'Autumn'] if s in df['Season'].unique()]
            sns.boxplot(data=df, x='Season', y='Total Fare', order=season_order,
                       palette='coolwarm', ax=axes[2], showfliers=False)
            axes[2].set_title('Fare by Season', fontweight='bold')
            axes[2].set_ylabel('Total Fare (BDT)')
        
        fig.suptitle('Pricing Overview', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return self._save_figure(fig, 'pricing_overview')
    
    def _plot_temporal_analysis(self, df: pd.DataFrame) -> Optional[Path]:
        """
        Consolidated temporal analysis: monthly trend + day-of-week pattern.
        One figure with 2 panels instead of 2 separate plots.
        """
        if 'Month' not in df.columns or 'Total Fare' not in df.columns:
            return None
        
        logger.info("Generating consolidated temporal analysis...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: Monthly fare trend
        monthly = df.groupby('Month')['Total Fare'].agg(['mean', 'std']).sort_index()
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        axes[0].plot(monthly.index, monthly['mean'], 'b-o', linewidth=2, markersize=6)
        axes[0].fill_between(monthly.index, monthly['mean'] - monthly['std'],
                            monthly['mean'] + monthly['std'], alpha=0.2, color='blue')
        axes[0].set_xticks(range(1, 13))
        axes[0].set_xticklabels(month_names, fontsize=9)
        axes[0].set_xlabel('Month')
        axes[0].set_ylabel('Average Fare (BDT)')
        axes[0].set_title('Monthly Fare Trend', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Panel 2: Day of week pattern
        if 'DayOfWeek' in df.columns:
            dow = df.groupby('DayOfWeek')['Total Fare'].mean().sort_index()
            day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            colors = ['coral' if i >= 5 else 'steelblue' for i in dow.index]
            axes[1].bar(dow.index, dow.values, color=colors, edgecolor='white')
            axes[1].set_xticks(range(7))
            axes[1].set_xticklabels(day_names)
            axes[1].set_xlabel('Day of Week')
            axes[1].set_ylabel('Average Fare (BDT)')
            axes[1].set_title('Fare by Day of Week', fontweight='bold')
            axes[1].axhline(dow.mean(), color='red', linestyle='--', alpha=0.7, label='Mean')
            axes[1].legend(fontsize=9)
        
        fig.suptitle('Temporal Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return self._save_figure(fig, 'temporal_analysis')


class KPICalculator:
    """
    Calculate Key Performance Indicators for the flight dataset.
    
    Computes business-relevant metrics for reporting and insights.
    """
    
    def __init__(self):
        """Initialize KPICalculator."""
        logger.info("KPICalculator initialized")
    
    def calculate_all_kpis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate all business KPIs.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary of computed KPIs
        """
        logger.info("Calculating Key Performance Indicators...")
        
        kpis = {}
        
        # Average fare per airline
        if 'Airline' in df.columns and 'Total Fare' in df.columns:
            kpis['avg_fare_by_airline'] = df.groupby('Airline')['Total Fare'].mean().round(2).to_dict()
            kpis['best_value_airline'] = df.groupby('Airline')['Total Fare'].mean().idxmin()
            kpis['premium_airline'] = df.groupby('Airline')['Total Fare'].mean().idxmax()
        
        # Most popular route
        if 'Source' in df.columns and 'Destination' in df.columns:
            df_temp = df.copy()
            df_temp['Route'] = df_temp['Source'] + ' to ' + df_temp['Destination']
            route_counts = df_temp['Route'].value_counts()
            kpis['most_popular_route'] = route_counts.index[0]
            kpis['most_popular_route_count'] = int(route_counts.iloc[0])
            kpis['total_unique_routes'] = route_counts.nunique()
        
        # Top 5 most expensive routes
        if 'Source' in df.columns and 'Destination' in df.columns and 'Total Fare' in df.columns:
            df_temp = df.copy()
            df_temp['Route'] = df_temp['Source'] + ' to ' + df_temp['Destination']
            expensive_routes = df_temp.groupby('Route')['Total Fare'].mean().nlargest(5)
            kpis['top_5_expensive_routes'] = expensive_routes.round(2).to_dict()
        
        # Seasonal fare variation
        if 'Season' in df.columns and 'Total Fare' in df.columns:
            seasonal_fares = df.groupby('Season')['Total Fare'].agg(['mean', 'std']).round(2)
            kpis['seasonal_avg_fares'] = seasonal_fares['mean'].to_dict()
            kpis['peak_season'] = seasonal_fares['mean'].idxmax()
            kpis['low_season'] = seasonal_fares['mean'].idxmin()
            
            # Seasonal price variation coefficient
            kpis['seasonal_price_variation_pct'] = round(
                (seasonal_fares['mean'].max() - seasonal_fares['mean'].min()) / 
                seasonal_fares['mean'].mean() * 100, 2
            )
        
        # Overall statistics
        if 'Total Fare' in df.columns:
            kpis['overall_avg_fare'] = round(df['Total Fare'].mean(), 2)
            kpis['overall_median_fare'] = round(df['Total Fare'].median(), 2)
            kpis['fare_std'] = round(df['Total Fare'].std(), 2)
            kpis['fare_range'] = {
                'min': round(df['Total Fare'].min(), 2),
                'max': round(df['Total Fare'].max(), 2)
            }
        
        # Tax ratio analysis
        if 'Tax & Surcharge' in df.columns and 'Total Fare' in df.columns:
            df_temp = df[df['Total Fare'] > 0].copy()
            df_temp['Tax_Ratio'] = df_temp['Tax & Surcharge'] / df_temp['Total Fare'] * 100
            kpis['avg_tax_percentage'] = round(df_temp['Tax_Ratio'].mean(), 2)
        
        # Day of week patterns
        if 'DayOfWeek' in df.columns and 'Total Fare' in df.columns:
            dow_fares = df.groupby('DayOfWeek')['Total Fare'].mean().round(2)
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                        'Friday', 'Saturday', 'Sunday']
            kpis['fare_by_day_of_week'] = {day_names[i]: fare for i, fare in dow_fares.items() if i < 7}
            kpis['cheapest_day'] = day_names[dow_fares.idxmin()] if dow_fares.idxmin() < 7 else 'Unknown'
            kpis['most_expensive_day'] = day_names[dow_fares.idxmax()] if dow_fares.idxmax() < 7 else 'Unknown'
        
        logger.info(f"Calculated {len(kpis)} KPIs")
        return kpis
    
    def generate_kpi_report(self, kpis: Dict[str, Any]) -> str:
        """
        Generate a formatted KPI report for non-technical stakeholders.
        
        Args:
            kpis: Dictionary of computed KPIs
        
        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 70,
            "FLIGHT PRICE ANALYSIS - KEY PERFORMANCE INDICATORS",
            "=" * 70,
            "",
        ]
        
        # Overall Pricing
        if 'overall_avg_fare' in kpis:
            report_lines.extend([
                "📊 OVERALL PRICING METRICS",
                "-" * 40,
                f"  Average Fare:     BDT {kpis['overall_avg_fare']:,.2f}",
                f"  Median Fare:      BDT {kpis['overall_median_fare']:,.2f}",
                f"  Standard Dev:     BDT {kpis['fare_std']:,.2f}",
                f"  Fare Range:       BDT {kpis['fare_range']['min']:,.2f} - {kpis['fare_range']['max']:,.2f}",
                ""
            ])
        
        # Airline Insights
        if 'best_value_airline' in kpis:
            report_lines.extend([
                "✈️ AIRLINE INSIGHTS",
                "-" * 40,
                f"  Best Value Airline:    {kpis['best_value_airline']}",
                f"  Premium Airline:       {kpis['premium_airline']}",
                ""
            ])
        
        # Route Analysis
        if 'most_popular_route' in kpis:
            report_lines.extend([
                "🗺️ ROUTE ANALYSIS",
                "-" * 40,
                f"  Most Popular Route:    {kpis['most_popular_route']}",
                f"                         ({kpis['most_popular_route_count']:,} flights)",
                f"  Total Unique Routes:   {kpis.get('total_unique_routes', 'N/A')}",
                ""
            ])
        
        # Top 5 Expensive Routes
        if 'top_5_expensive_routes' in kpis:
            report_lines.extend([
                "💰 TOP 5 MOST EXPENSIVE ROUTES",
                "-" * 40,
            ])
            for i, (route, fare) in enumerate(kpis['top_5_expensive_routes'].items(), 1):
                report_lines.append(f"  {i}. {route}: BDT {fare:,.2f}")
            report_lines.append("")
        
        # Seasonal Patterns
        if 'peak_season' in kpis:
            report_lines.extend([
                "📅 SEASONAL PATTERNS",
                "-" * 40,
                f"  Peak Season:           {kpis['peak_season']}",
                f"  Low Season:            {kpis['low_season']}",
                f"  Seasonal Variation:    {kpis['seasonal_price_variation_pct']:.1f}%",
                ""
            ])
        
        # Day of Week Patterns
        if 'cheapest_day' in kpis:
            report_lines.extend([
                "📆 DAY OF WEEK PATTERNS",
                "-" * 40,
                f"  Cheapest Day:          {kpis['cheapest_day']}",
                f"  Most Expensive Day:    {kpis['most_expensive_day']}",
                ""
            ])
        
        # Tax Analysis
        if 'avg_tax_percentage' in kpis:
            report_lines.extend([
                "💵 TAX ANALYSIS",
                "-" * 40,
                f"  Average Tax Percentage: {kpis['avg_tax_percentage']:.1f}%",
                ""
            ])
        
        report_lines.extend([
            "=" * 70,
            f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
        ])
        
        report = "\n".join(report_lines)
        logger.info("Generated KPI report")
        
        return report
    
    def save_kpi_report(self, kpis: Dict[str, Any], filename: str = None) -> Path:
        """
        Save KPI report to file.
        
        Args:
            kpis: Dictionary of computed KPIs
            filename: Optional filename for the report
        
        Returns:
            Path to saved report
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"kpi_report_{timestamp}.txt"
        
        report = self.generate_kpi_report(kpis)
        filepath = REPORTS_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Saved KPI report to: {filepath}")
        return filepath
