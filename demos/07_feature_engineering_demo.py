"""
Demo: Feature Engineering
=========================
This demo showcases feature engineering functions in dskit.
"""

from dskit import (
    create_polynomial_features, create_date_features,
    create_binning_features, select_features_univariate,
    select_features_rfe, apply_pca,
    create_aggregation_features, create_target_encoding
)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_data():
    """Create sample dataset for feature engineering"""
    np.random.seed(42)
    n = 200
    
    # Generate dates
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n)]
    
    return pd.DataFrame({
        'date': dates,
        'x1': np.random.uniform(0, 10, n),
        'x2': np.random.uniform(0, 10, n),
        'age': np.random.randint(18, 70, n),
        'income': np.random.randint(30000, 150000, n),
        'group': np.random.choice(['A', 'B', 'C'], n),
        'target': np.random.choice([0, 1], n)
    })


def demo_polynomial_features():
    """Demo 1: Create polynomial features"""
    print("=" * 60)
    print("DEMO 1: Polynomial Features")
    print("=" * 60)
    
    df = create_sample_data()
    df_numeric = df[['x1', 'x2', 'age']].copy()
    
    print(f"\n📊 Original features: {list(df_numeric.columns)}")
    print(f"   Shape: {df_numeric.shape}")
    
    print("\n🔧 Creating polynomial features (degree=2)...")
    df_poly = create_polynomial_features(df_numeric, degree=2, interaction_only=False)
    
    print(f"\n✓ New features: {list(df_poly.columns)}")
    print(f"   Shape: {df_poly.shape}")
    print(f"   Added {df_poly.shape[1] - df_numeric.shape[1]} new features")


def demo_date_features():
    """Demo 2: Extract date features"""
    print("\n" + "=" * 60)
    print("DEMO 2: Date Features Extraction")
    print("=" * 60)
    
    df = create_sample_data()
    
    print(f"\n📊 Original columns: {list(df.columns)}")
    print("\n📅 Sample dates:")
    print(df['date'].head())
    
    print("\n🔧 Extracting date features...")
    df_dates = create_date_features(df, date_cols=['date'])
    
    print(f"\n✓ New columns: {list(df_dates.columns)}")
    print("\n📊 Sample of extracted features:")
    date_cols = [col for col in df_dates.columns if 'date' in col.lower()]
    print(df_dates[date_cols].head())


def demo_binning_features():
    """Demo 3: Create binned features"""
    print("\n" + "=" * 60)
    print("DEMO 3: Binning Features")
    print("=" * 60)
    
    df = create_sample_data()
    
    print("\n📊 Original age distribution:")
    print(df['age'].describe())
    
    print("\n🔧 Creating binned features (5 bins, quantile strategy)...")
    df_binned = create_binning_features(
        df, 
        numeric_cols=['age', 'income'], 
        n_bins=5, 
        strategy='quantile'
    )
    
    print("\n✓ Binned features created:")
    bin_cols = [col for col in df_binned.columns if '_bin' in col]
    print(f"   New columns: {bin_cols}")
    
    print("\n📊 Age bins distribution:")
    print(df_binned['age_bin'].value_counts().sort_index())


def demo_feature_selection_univariate():
    """Demo 4: Univariate feature selection"""
    print("\n" + "=" * 60)
    print("DEMO 4: Univariate Feature Selection")
    print("=" * 60)
    
    df = create_sample_data()
    X = df[['x1', 'x2', 'age', 'income']]
    y = df['target']
    
    print(f"\n📊 Total features: {X.shape[1]}")
    print(f"   Features: {list(X.columns)}")
    
    print("\n🔧 Selecting top 3 features using univariate selection...")
    X_selected, selected_features = select_features_univariate(
        X, y, k=3, task='classification'
    )
    
    print(f"\n✓ Selected features: {selected_features}")
    print(f"   Shape: {X_selected.shape}")


def demo_feature_selection_rfe():
    """Demo 5: Recursive Feature Elimination"""
    print("\n" + "=" * 60)
    print("DEMO 5: Recursive Feature Elimination (RFE)")
    print("=" * 60)
    
    df = create_sample_data()
    X = df[['x1', 'x2', 'age', 'income']]
    y = df['target']
    
    print(f"\n📊 Total features: {X.shape[1]}")
    print(f"   Features: {list(X.columns)}")
    
    print("\n🔧 Selecting top 2 features using RFE...")
    X_selected, selected_features = select_features_rfe(
        X, y, n_features=2, task='classification'
    )
    
    print(f"\n✓ Selected features: {selected_features}")
    print(f"   Shape: {X_selected.shape}")


def demo_pca():
    """Demo 6: Principal Component Analysis"""
    print("\n" + "=" * 60)
    print("DEMO 6: Principal Component Analysis (PCA)")
    print("=" * 60)
    
    df = create_sample_data()
    df_numeric = df[['x1', 'x2', 'age', 'income']]
    
    print(f"\n📊 Original shape: {df_numeric.shape}")
    
    print("\n🔧 Applying PCA (95% variance threshold)...")
    df_pca, pca_obj = apply_pca(df_numeric, variance_threshold=0.95)
    
    print(f"\n✓ PCA completed:")
    print(f"   New shape: {df_pca.shape}")
    print(f"   Components kept: {df_pca.shape[1]}")
    print(f"   Explained variance: {pca_obj.explained_variance_ratio_.sum():.4f}")


def demo_aggregation_features():
    """Demo 7: Create aggregation features"""
    print("\n" + "=" * 60)
    print("DEMO 7: Aggregation Features")
    print("=" * 60)
    
    df = create_sample_data()
    
    print(f"\n📊 Original shape: {df.shape}")
    print(f"   Groups: {df['group'].unique()}")
    
    print("\n🔧 Creating aggregation features by group...")
    df_agg = create_aggregation_features(
        df, 
        group_col='group', 
        agg_cols=['age', 'income'],
        agg_funcs=['mean', 'std', 'min', 'max']
    )
    
    print(f"\n✓ Aggregation features created:")
    print(f"   New shape: {df_agg.shape}")
    agg_cols = [col for col in df_agg.columns if '_mean' in col or '_std' in col]
    print(f"   Sample columns: {agg_cols[:4]}")


def demo_target_encoding():
    """Demo 8: Target encoding"""
    print("\n" + "=" * 60)
    print("DEMO 8: Target Encoding")
    print("=" * 60)
    
    df = create_sample_data()
    
    print(f"\n📊 Categorical column 'group': {df['group'].unique()}")
    print(f"   Target column: binary (0/1)")
    
    print("\n🔧 Applying target encoding...")
    df_encoded = create_target_encoding(
        df, 
        categorical_cols=['group'], 
        target_col='target',
        smoothing=1.0
    )
    
    print(f"\n✓ Target encoding completed:")
    print("\n📊 Encoded values:")
    print(df_encoded.groupby('group')['group_target_enc'].first())


if __name__ == "__main__":
    print("\n" + "🔧" * 30)
    print("FEATURE ENGINEERING DEMO".center(60))
    print("🔧" * 30 + "\n")
    
    demo_polynomial_features()
    demo_date_features()
    demo_binning_features()
    demo_feature_selection_univariate()
    demo_feature_selection_rfe()
    demo_pca()
    demo_aggregation_features()
    demo_target_encoding()
    
    print("\n" + "✅" * 30)
    print("ALL DEMOS COMPLETED".center(60))
    print("✅" * 30 + "\n")
