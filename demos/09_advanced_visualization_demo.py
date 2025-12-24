"""
Demo: Advanced Visualization
============================
This demo showcases advanced visualization functions in dskit.
"""

from dskit import (
    plot_feature_importance, plot_target_distribution,
    plot_feature_vs_target, plot_correlation_advanced,
    plot_missing_patterns_advanced, plot_outliers_advanced
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

def create_sample_data():
    """Create sample dataset for visualization"""
    np.random.seed(42)
    n = 200
    
    df = pd.DataFrame({
        'age': np.random.randint(18, 70, n),
        'income': np.random.randint(30000, 150000, n),
        'experience': np.random.randint(0, 30, n),
        'satisfaction': np.random.uniform(1, 10, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'target': np.random.choice([0, 1], n)
    })
    
    # Introduce missing values
    df.loc[np.random.choice(df.index, 20), 'age'] = np.nan
    df.loc[np.random.choice(df.index, 15), 'income'] = np.nan
    
    # Introduce outliers
    df.loc[np.random.choice(df.index, 5), 'income'] = np.random.randint(500000, 1000000, 5)
    
    return df


def demo_feature_importance():
    """Demo 1: Plot feature importance"""
    print("=" * 60)
    print("DEMO 1: Feature Importance Visualization")
    print("=" * 60)
    
    df = create_sample_data().dropna()
    X = df[['age', 'income', 'experience', 'satisfaction']]
    y = df['target']
    
    print("\n🤖 Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    print("\n📊 Creating feature importance plot...")
    try:
        plot_feature_importance(model, feature_names=X.columns, top_n=10)
        plt.savefig('temp_feature_importance.png', bbox_inches='tight', dpi=100)
        print("✓ Feature importance plot created")
        plt.close()
    except Exception as e:
        print(f"⚠️ Plot generation skipped: {str(e)}")


def demo_target_distribution():
    """Demo 2: Plot target distribution"""
    print("\n" + "=" * 60)
    print("DEMO 2: Target Distribution Visualization")
    print("=" * 60)
    
    df = create_sample_data()
    
    print("\n📊 Creating target distribution plot...")
    try:
        plot_target_distribution(df, target_col='target', task='classification')
        plt.savefig('temp_target_dist.png', bbox_inches='tight', dpi=100)
        print("✓ Target distribution plot created")
        print(f"\n   Class distribution:")
        print(df['target'].value_counts())
        plt.close()
    except Exception as e:
        print(f"⚠️ Plot generation skipped: {str(e)}")


def demo_feature_vs_target():
    """Demo 3: Plot feature vs target relationship"""
    print("\n" + "=" * 60)
    print("DEMO 3: Feature vs Target Visualization")
    print("=" * 60)
    
    df = create_sample_data().dropna()
    
    print("\n📊 Creating feature vs target plots...")
    try:
        plot_feature_vs_target(df, feature_col='age', target_col='target', task='classification')
        plt.savefig('temp_feature_target.png', bbox_inches='tight', dpi=100)
        print("✓ Feature vs target plot created")
        plt.close()
    except Exception as e:
        print(f"⚠️ Plot generation skipped: {str(e)}")


def demo_correlation_advanced():
    """Demo 4: Advanced correlation heatmap"""
    print("\n" + "=" * 60)
    print("DEMO 4: Advanced Correlation Visualization")
    print("=" * 60)
    
    df = create_sample_data().dropna()
    
    print("\n📊 Creating advanced correlation heatmap...")
    print("   (Showing correlations above 0.3 threshold)")
    try:
        plot_correlation_advanced(df, method='pearson', threshold=0.3)
        plt.savefig('temp_correlation_advanced.png', bbox_inches='tight', dpi=100)
        print("✓ Advanced correlation plot created")
        plt.close()
    except Exception as e:
        print(f"⚠️ Plot generation skipped: {str(e)}")


def demo_missing_patterns():
    """Demo 5: Advanced missing patterns visualization"""
    print("\n" + "=" * 60)
    print("DEMO 5: Missing Patterns Visualization")
    print("=" * 60)
    
    df = create_sample_data()
    
    print("\n📊 Missing values summary:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    print("\n📊 Creating missing patterns visualization...")
    try:
        plot_missing_patterns_advanced(df)
        plt.savefig('temp_missing_patterns.png', bbox_inches='tight', dpi=100)
        print("✓ Missing patterns plot created")
        plt.close()
    except Exception as e:
        print(f"⚠️ Plot generation skipped: {str(e)}")


def demo_outliers_advanced():
    """Demo 6: Advanced outlier visualization"""
    print("\n" + "=" * 60)
    print("DEMO 6: Advanced Outlier Visualization")
    print("=" * 60)
    
    df = create_sample_data()
    
    print("\n📊 Creating advanced outlier visualization...")
    try:
        plot_outliers_advanced(df, method='iqr')
        plt.savefig('temp_outliers_advanced.png', bbox_inches='tight', dpi=100)
        print("✓ Advanced outlier plot created")
        plt.close()
    except Exception as e:
        print(f"⚠️ Plot generation skipped: {str(e)}")


def cleanup():
    """Clean up temporary files"""
    import os
    temp_files = [
        'temp_feature_importance.png', 'temp_target_dist.png',
        'temp_feature_target.png', 'temp_correlation_advanced.png',
        'temp_missing_patterns.png', 'temp_outliers_advanced.png'
    ]
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
    print("\n🧹 Cleaned up temporary files")


if __name__ == "__main__":
    print("\n" + "📈" * 30)
    print("ADVANCED VISUALIZATION DEMO".center(60))
    print("📈" * 30 + "\n")
    
    try:
        demo_feature_importance()
        demo_target_distribution()
        demo_feature_vs_target()
        demo_correlation_advanced()
        demo_missing_patterns()
        demo_outliers_advanced()
    finally:
        cleanup()
    
    print("\n" + "✅" * 30)
    print("ALL DEMOS COMPLETED".center(60))
    print("✅" * 30 + "\n")
    print("\n💡 Note: In interactive environment, plots would be displayed")
