"""
Ak-dskit Demos Package
======================

This package contains comprehensive demonstrations of all dskit functionality.

Usage:
    # Run all demos
    python run_all_demos.py
    
    # Run individual demos
    python 01_data_io_demo.py
    python 02_data_cleaning_demo.py
    # ... etc
    
    # Import and use specific demo functions
    from demos.quick_reference import data_io_demos
    data_io_demos()

Available Demos:
    01 - Data I/O Operations
    02 - Data Cleaning
    03 - Exploratory Data Analysis (EDA)
    04 - Data Visualization
    05 - Data Preprocessing
    06 - Machine Learning Modeling
    07 - Feature Engineering
    08 - NLP Utilities
    09 - Advanced Visualization
    10 - AutoML & Optimization
    11 - Hyperplane Visualization
    12 - Complete End-to-End Pipeline

Documentation:
    See README.md in this folder for detailed information.
"""

__version__ = "1.0.0"
__author__ = "Ak-dskit Team"

# Make key demo functions easily accessible
__all__ = [
    'run_all_demos',
    'quick_reference',
    'DEMO_FILES',
    'DEMO_DESCRIPTIONS'
]

# Demo file listing
DEMO_FILES = [
    '01_data_io_demo.py',
    '02_data_cleaning_demo.py',
    '03_eda_demo.py',
    '04_visualization_demo.py',
    '05_preprocessing_demo.py',
    '06_modeling_demo.py',
    '07_feature_engineering_demo.py',
    '08_nlp_demo.py',
    '09_advanced_visualization_demo.py',
    '10_automl_demo.py',
    '11_hyperplane_demo.py',
    '12_complete_pipeline_demo.py'
]

DEMO_DESCRIPTIONS = {
    '01_data_io_demo.py': 'Data Input/Output Operations',
    '02_data_cleaning_demo.py': 'Data Cleaning',
    '03_eda_demo.py': 'Exploratory Data Analysis',
    '04_visualization_demo.py': 'Data Visualization',
    '05_preprocessing_demo.py': 'Data Preprocessing',
    '06_modeling_demo.py': 'Machine Learning Modeling',
    '07_feature_engineering_demo.py': 'Feature Engineering',
    '08_nlp_demo.py': 'NLP Utilities',
    '09_advanced_visualization_demo.py': 'Advanced Visualization',
    '10_automl_demo.py': 'AutoML & Optimization',
    '11_hyperplane_demo.py': 'Hyperplane Visualization',
    '12_complete_pipeline_demo.py': 'End-to-End ML Pipeline'
}


def list_demos():
    """List all available demos with descriptions"""
    print("\n" + "=" * 70)
    print("AVAILABLE DSKIT DEMOS".center(70))
    print("=" * 70 + "\n")
    
    for i, (file, desc) in enumerate(DEMO_DESCRIPTIONS.items(), 1):
        print(f"{i:2d}. {desc}")
        print(f"    File: {file}\n")
    
    print("=" * 70)
    print(f"Total: {len(DEMO_FILES)} demos available")
    print("=" * 70 + "\n")
    
    print("To run:")
    print("  • All demos: python run_all_demos.py")
    print("  • Single demo: python 01_data_io_demo.py")
    print("  • See README.md for more options\n")


if __name__ == "__main__":
    list_demos()
