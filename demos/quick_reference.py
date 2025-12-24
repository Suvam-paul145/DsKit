"""
Quick Reference Guide - dskit Demo Functions
============================================

This file provides a quick reference to all demo functions across all files.
Import and use individual functions as needed.
"""

# Import all demo modules
from pathlib import Path
import sys

# Add demos folder to path
demos_dir = Path(__file__).parent
sys.path.insert(0, str(demos_dir))


# =============================================================================
# DATA I/O OPERATIONS (01_data_io_demo.py)
# =============================================================================

def data_io_demos():
    """
    Run all Data I/O demos:
    - Loading data from various formats (CSV, Excel, JSON, Parquet)
    - Batch loading from folders
    - Saving data in multiple formats
    """
    from demos.demo_01_data_io import (
        demo_basic_loading,
        demo_folder_loading,
        demo_save_operations
    )
    demo_basic_loading()
    demo_folder_loading()
    demo_save_operations()


# =============================================================================
# DATA CLEANING (02_data_cleaning_demo.py)
# =============================================================================

def data_cleaning_demos():
    """
    Run all Data Cleaning demos:
    - Fixing data types automatically
    - Standardizing column names
    - Replacing special characters
    - Handling missing values
    - Detecting and removing outliers
    - Text/NLP cleaning
    """
    from demos.demo_02_data_cleaning import (
        demo_fix_dtypes,
        demo_rename_columns,
        demo_replace_specials,
        demo_missing_values,
        demo_outliers,
        demo_nlp_cleaning
    )
    demo_fix_dtypes()
    demo_rename_columns()
    demo_replace_specials()
    demo_missing_values()
    demo_outliers()
    demo_nlp_cleaning()


# =============================================================================
# EXPLORATORY DATA ANALYSIS (03_eda_demo.py)
# =============================================================================

def eda_demos():
    """
    Run all EDA demos:
    - Basic statistics
    - Quick EDA overview
    - Comprehensive EDA
    - Data health check
    - Feature analysis
    """
    from demos.demo_03_eda import (
        demo_basic_stats,
        demo_quick_eda,
        demo_comprehensive_eda,
        demo_health_check,
        demo_feature_analysis
    )
    demo_basic_stats()
    demo_quick_eda()
    demo_comprehensive_eda()
    demo_health_check()
    demo_feature_analysis()


# =============================================================================
# VISUALIZATION (04_visualization_demo.py)
# =============================================================================

def visualization_demos():
    """
    Run all Visualization demos:
    - Missing value patterns
    - Distribution histograms
    - Boxplots for outliers
    - Correlation heatmaps
    - Pairplots
    """
    from demos.demo_04_visualization import (
        demo_missingness_plot,
        demo_histograms,
        demo_boxplots,
        demo_correlation_heatmap,
        demo_pairplot
    )
    demo_missingness_plot()
    demo_histograms()
    demo_boxplots()
    demo_correlation_heatmap()
    demo_pairplot()


# =============================================================================
# PREPROCESSING (05_preprocessing_demo.py)
# =============================================================================

def preprocessing_demos():
    """
    Run all Preprocessing demos:
    - Automatic categorical encoding
    - Feature scaling (Standard, MinMax, Robust)
    - Train-test splitting
    - Complete preprocessing pipeline
    """
    from demos.demo_05_preprocessing import (
        demo_auto_encode,
        demo_auto_scale,
        demo_train_test_split,
        demo_complete_pipeline
    )
    demo_auto_encode()
    demo_auto_scale()
    demo_train_test_split()
    demo_complete_pipeline()


# =============================================================================
# MACHINE LEARNING MODELING (06_modeling_demo.py)
# =============================================================================

def modeling_demos():
    """
    Run all Modeling demos:
    - Quick model training
    - Model comparison
    - Hyperparameter optimization
    - Model evaluation
    - Error analysis
    """
    from demos.demo_06_modeling import (
        demo_quick_model,
        demo_compare_models,
        demo_hyperparameter_tuning,
        demo_model_evaluation,
        demo_error_analysis
    )
    demo_quick_model()
    demo_compare_models()
    demo_hyperparameter_tuning()
    demo_model_evaluation()
    demo_error_analysis()


# =============================================================================
# FEATURE ENGINEERING (07_feature_engineering_demo.py)
# =============================================================================

def feature_engineering_demos():
    """
    Run all Feature Engineering demos:
    - Polynomial features
    - Date feature extraction
    - Binning and discretization
    - Univariate feature selection
    - Recursive Feature Elimination
    - Principal Component Analysis
    - Aggregation features
    - Target encoding
    """
    from demos.demo_07_feature_engineering import (
        demo_polynomial_features,
        demo_date_features,
        demo_binning_features,
        demo_feature_selection_univariate,
        demo_feature_selection_rfe,
        demo_pca,
        demo_aggregation_features,
        demo_target_encoding
    )
    demo_polynomial_features()
    demo_date_features()
    demo_binning_features()
    demo_feature_selection_univariate()
    demo_feature_selection_rfe()
    demo_pca()
    demo_aggregation_features()
    demo_target_encoding()


# =============================================================================
# NLP UTILITIES (08_nlp_demo.py)
# =============================================================================

def nlp_demos():
    """
    Run all NLP demos:
    - Basic text statistics
    - Advanced text cleaning
    - Text feature extraction
    - Sentiment analysis
    - Complete NLP pipeline
    """
    from demos.demo_08_nlp import (
        demo_text_stats,
        demo_text_cleaning,
        demo_text_features,
        demo_sentiment_analysis,
        demo_complete_nlp_pipeline
    )
    demo_text_stats()
    demo_text_cleaning()
    demo_text_features()
    demo_sentiment_analysis()
    demo_complete_nlp_pipeline()


# =============================================================================
# QUICK FUNCTION INDEX
# =============================================================================

DEMO_INDEX = {
    # Data I/O
    'load_data': '01_data_io_demo.py - demo_basic_loading()',
    'batch_load': '01_data_io_demo.py - demo_folder_loading()',
    'save_data': '01_data_io_demo.py - demo_save_operations()',
    
    # Data Cleaning
    'fix_types': '02_data_cleaning_demo.py - demo_fix_dtypes()',
    'clean_columns': '02_data_cleaning_demo.py - demo_rename_columns()',
    'handle_missing': '02_data_cleaning_demo.py - demo_missing_values()',
    'remove_outliers': '02_data_cleaning_demo.py - demo_outliers()',
    
    # EDA
    'basic_stats': '03_eda_demo.py - demo_basic_stats()',
    'quick_eda': '03_eda_demo.py - demo_quick_eda()',
    'comprehensive_eda': '03_eda_demo.py - demo_comprehensive_eda()',
    
    # Visualization
    'plot_missing': '04_visualization_demo.py - demo_missingness_plot()',
    'plot_distributions': '04_visualization_demo.py - demo_histograms()',
    'plot_correlations': '04_visualization_demo.py - demo_correlation_heatmap()',
    
    # Preprocessing
    'encode_features': '05_preprocessing_demo.py - demo_auto_encode()',
    'scale_features': '05_preprocessing_demo.py - demo_auto_scale()',
    'split_data': '05_preprocessing_demo.py - demo_train_test_split()',
    
    # Modeling
    'train_model': '06_modeling_demo.py - demo_quick_model()',
    'compare_models': '06_modeling_demo.py - demo_compare_models()',
    'tune_hyperparameters': '06_modeling_demo.py - demo_hyperparameter_tuning()',
    'evaluate_model': '06_modeling_demo.py - demo_model_evaluation()',
    
    # Feature Engineering
    'polynomial_features': '07_feature_engineering_demo.py - demo_polynomial_features()',
    'date_features': '07_feature_engineering_demo.py - demo_date_features()',
    'feature_selection': '07_feature_engineering_demo.py - demo_feature_selection_univariate()',
    'pca': '07_feature_engineering_demo.py - demo_pca()',
    
    # NLP
    'text_cleaning': '08_nlp_demo.py - demo_text_cleaning()',
    'text_features': '08_nlp_demo.py - demo_text_features()',
    'sentiment': '08_nlp_demo.py - demo_sentiment_analysis()',
    
    # Advanced
    'automl': '10_automl_demo.py',
    'hyperplanes': '11_hyperplane_demo.py',
    'complete_pipeline': '12_complete_pipeline_demo.py'
}


def print_demo_index():
    """Print the complete demo function index"""
    print("\n" + "=" * 70)
    print("DSKIT DEMO FUNCTION INDEX".center(70))
    print("=" * 70 + "\n")
    
    print("Quick reference to find specific functionality:\n")
    
    categories = {
        'Data I/O': ['load_data', 'batch_load', 'save_data'],
        'Data Cleaning': ['fix_types', 'clean_columns', 'handle_missing', 'remove_outliers'],
        'EDA': ['basic_stats', 'quick_eda', 'comprehensive_eda'],
        'Visualization': ['plot_missing', 'plot_distributions', 'plot_correlations'],
        'Preprocessing': ['encode_features', 'scale_features', 'split_data'],
        'Modeling': ['train_model', 'compare_models', 'tune_hyperparameters', 'evaluate_model'],
        'Feature Engineering': ['polynomial_features', 'date_features', 'feature_selection', 'pca'],
        'NLP': ['text_cleaning', 'text_features', 'sentiment'],
        'Advanced': ['automl', 'hyperplanes', 'complete_pipeline']
    }
    
    for category, functions in categories.items():
        print(f"\n{category}:")
        print("-" * 70)
        for func in functions:
            location = DEMO_INDEX.get(func, 'Unknown')
            print(f"  • {func:25} → {location}")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    print_demo_index()
    
    print("\nTo use a specific demo function:")
    print("  from demos.quick_reference import data_io_demos")
    print("  data_io_demos()")
    print("\nOr import specific functions:")
    print("  from demos.demo_01_data_io import demo_basic_loading")
    print("  demo_basic_loading()\n")
