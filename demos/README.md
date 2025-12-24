# 🚀 Ak-dskit Demos

Welcome to the comprehensive demo collection for **Ak-dskit**! This folder contains 12 detailed demonstration scripts showcasing all major functionalities of the library.

## 📚 Demo Overview

Each demo file is self-contained and demonstrates specific features of dskit with clear examples and explanations.

### Core Functionality Demos

1. **[01_data_io_demo.py](01_data_io_demo.py)** - Data Input/Output Operations

   - Load data from CSV, Excel, JSON, Parquet
   - Batch loading from folders
   - Save data in multiple formats
   - Smart data type detection

2. **[02_data_cleaning_demo.py](02_data_cleaning_demo.py)** - Data Cleaning

   - Automatic data type fixing
   - Column name standardization
   - Special character replacement
   - Missing value analysis and imputation
   - Outlier detection and removal
   - Text/NLP cleaning

3. **[03_eda_demo.py](03_eda_demo.py)** - Exploratory Data Analysis

   - Basic statistics
   - Quick EDA overview
   - Comprehensive EDA with visualizations
   - Data health checks
   - Feature analysis reports

4. **[04_visualization_demo.py](04_visualization_demo.py)** - Data Visualization

   - Missing value patterns
   - Distribution histograms
   - Boxplots for outliers
   - Correlation heatmaps
   - Pairplots

5. **[05_preprocessing_demo.py](05_preprocessing_demo.py)** - Data Preprocessing

   - Automatic categorical encoding
   - Feature scaling (Standard, MinMax, Robust)
   - Train-test splitting
   - Complete preprocessing pipeline

6. **[06_modeling_demo.py](06_modeling_demo.py)** - Machine Learning Modeling
   - Quick model training
   - Model comparison
   - Hyperparameter optimization
   - Model evaluation
   - Error analysis

### Advanced Feature Demos

7. **[07_feature_engineering_demo.py](07_feature_engineering_demo.py)** - Feature Engineering

   - Polynomial features
   - Date feature extraction
   - Binning and discretization
   - Univariate feature selection
   - Recursive Feature Elimination (RFE)
   - Principal Component Analysis (PCA)
   - Aggregation features
   - Target encoding

8. **[08_nlp_demo.py](08_nlp_demo.py)** - NLP Utilities

   - Text statistics
   - Advanced text cleaning
   - Text feature extraction
   - Sentiment analysis
   - Complete NLP pipeline

9. **[09_advanced_visualization_demo.py](09_advanced_visualization_demo.py)** - Advanced Visualization

   - Feature importance plots
   - Target distribution analysis
   - Feature vs target relationships
   - Advanced correlation analysis
   - Missing pattern visualization
   - Outlier visualization

10. **[10_automl_demo.py](10_automl_demo.py)** - AutoML & Optimization

    - Default parameter spaces
    - Random search optimization
    - Grid search optimization
    - Bayesian optimization
    - Method comparison

11. **[11_hyperplane_demo.py](11_hyperplane_demo.py)** - Hyperplane Visualization

    - Hyperplane class usage
    - SVM hyperplane visualization
    - Logistic regression hyperplane
    - Hyperplane extraction
    - Algorithm comparison

12. **[12_complete_pipeline_demo.py](12_complete_pipeline_demo.py)** - End-to-End Pipeline
    - Complete ML workflow
    - Data loading → Cleaning → EDA → Feature Engineering
    - Preprocessing → Modeling → Evaluation → Interpretation
    - Best practices demonstration

## 🚀 Quick Start

### Run All Demos

```bash
# Navigate to demos folder
cd demos

# Run individual demos
python 01_data_io_demo.py
python 02_data_cleaning_demo.py
# ... and so on

# Or run all demos at once
python run_all_demos.py
```

### Run Specific Demo

```python
# Example: Run data cleaning demo
python 02_data_cleaning_demo.py
```

### Interactive Usage

```python
# Import and use demo functions interactively
from demos.demo_01_data_io import demo_basic_loading
demo_basic_loading()
```

## 📋 Requirements

All demos use the standard dskit installation:

```bash
pip install Ak-dskit
```

Some advanced features may require additional packages:

- `textblob` for sentiment analysis
- `hyperopt` or `optuna` for advanced optimization

## 🎯 Learning Path

**Beginners**: Start with demos 1-6 (Core Functionality)

1. Data I/O → Data Cleaning → EDA
2. Visualization → Preprocessing → Modeling

**Intermediate**: Continue with demos 7-9 (Advanced Features) 3. Feature Engineering → NLP → Advanced Visualization

**Advanced**: Explore demos 10-12 (AutoML & Pipelines) 4. AutoML → Hyperplane → Complete Pipeline

## 📊 Demo Features

Each demo includes:

- ✅ Clear section headers and descriptions
- ✅ Step-by-step explanations
- ✅ Sample data generation
- ✅ Multiple use case examples
- ✅ Output interpretation
- ✅ Best practices
- ✅ Error handling examples

## 💡 Tips

1. **Read the code**: Each demo is well-commented and self-explanatory
2. **Modify parameters**: Experiment with different settings
3. **Use sample data**: All demos create their own sample datasets
4. **Check outputs**: Each demo prints detailed progress and results
5. **Visualizations**: Some demos create plots (saved as temporary files)

## 🔧 Customization

All demos can be easily customized:

```python
# Example: Modify demo parameters
from demos.demo_06_modeling import demo_compare_models

# Use your own data
import pandas as pd
df = pd.read_csv('your_data.csv')

# Customize and run
demo_compare_models(df, target_col='your_target')
```

## 📖 Documentation

For detailed API documentation, see:

- [API Reference](../docs/API_REFERENCE.md)
- [Feature Catalog](../docs/DSKIT_FEATURE_CATALOG.md)
- [Complete ML Pipeline Guide](../docs/COMPLETE_ML_PIPELINE_COMPARISON.md)

## 🤝 Contributing

To add a new demo:

1. Follow the existing demo structure
2. Include comprehensive examples
3. Add clear documentation
4. Update this README

## 📞 Support

- **Issues**: https://github.com/Programmers-Paradise/DsKit/issues
- **Documentation**: https://github.com/Programmers-Paradise/DsKit
- **PyPI**: https://pypi.org/project/Ak-dskit/

## 📝 License

These demos are part of the Ak-dskit package and follow the same license.

---

**Happy Learning! 🎉**

Start with [01_data_io_demo.py](01_data_io_demo.py) and work your way through to [12_complete_pipeline_demo.py](12_complete_pipeline_demo.py) for a complete understanding of dskit capabilities!
