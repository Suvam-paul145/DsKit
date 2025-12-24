"""
Demo: Visualization Functions
=============================
This demo showcases all visualization functions in dskit.
"""

from dskit import (
    plot_missingness, plot_histograms, plot_boxplots,
    plot_correlation_heatmap, plot_pairplot
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_sample_data():
    """Create sample dataset with various characteristics"""
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        'feature_1': np.random.normal(50, 10, n),
        'feature_2': np.random.normal(100, 20, n),
        'feature_3': np.random.exponential(5, n),
        'feature_4': np.random.uniform(0, 100, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'target': np.random.choice([0, 1], n)
    })
    
    # Introduce some missing values
    df.loc[np.random.choice(df.index, 10), 'feature_1'] = np.nan
    df.loc[np.random.choice(df.index, 15), 'feature_2'] = np.nan
    
    return df


def demo_missingness_plot():
    """Demo 1: Visualize missing values"""
    print("=" * 60)
    print("DEMO 1: Missing Values Visualization")
    print("=" * 60)
    
    df = create_sample_data()
    
    print("\n📊 Creating missingness plot...")
    try:
        plot_missingness(df)
        plt.savefig('temp_missingness.png', bbox_inches='tight', dpi=100)
        print("✓ Missingness plot created and saved")
        plt.close()
    except Exception as e:
        print(f"⚠️ Plot generation skipped: {str(e)}")


def demo_histograms():
    """Demo 2: Distribution histograms"""
    print("\n" + "=" * 60)
    print("DEMO 2: Feature Distributions")
    print("=" * 60)
    
    df = create_sample_data()
    
    print("\n📊 Creating histogram plots...")
    try:
        plot_histograms(df, bins=20)
        plt.savefig('temp_histograms.png', bbox_inches='tight', dpi=100)
        print("✓ Histogram plots created and saved")
        plt.close()
    except Exception as e:
        print(f"⚠️ Plot generation skipped: {str(e)}")


def demo_boxplots():
    """Demo 3: Boxplots for outlier detection"""
    print("\n" + "=" * 60)
    print("DEMO 3: Boxplots for Outliers")
    print("=" * 60)
    
    df = create_sample_data()
    
    print("\n📊 Creating boxplots...")
    try:
        plot_boxplots(df)
        plt.savefig('temp_boxplots.png', bbox_inches='tight', dpi=100)
        print("✓ Boxplots created and saved")
        plt.close()
    except Exception as e:
        print(f"⚠️ Plot generation skipped: {str(e)}")


def demo_correlation_heatmap():
    """Demo 4: Correlation heatmap"""
    print("\n" + "=" * 60)
    print("DEMO 4: Correlation Heatmap")
    print("=" * 60)
    
    df = create_sample_data()
    
    print("\n📊 Creating correlation heatmap...")
    try:
        plot_correlation_heatmap(df)
        plt.savefig('temp_correlation.png', bbox_inches='tight', dpi=100)
        print("✓ Correlation heatmap created and saved")
        plt.close()
    except Exception as e:
        print(f"⚠️ Plot generation skipped: {str(e)}")


def demo_pairplot():
    """Demo 5: Pairplot for feature relationships"""
    print("\n" + "=" * 60)
    print("DEMO 5: Pairplot")
    print("=" * 60)
    
    df = create_sample_data()
    # Select subset of features for clarity
    df_subset = df[['feature_1', 'feature_2', 'feature_3', 'category']].dropna()
    
    print("\n📊 Creating pairplot...")
    try:
        plot_pairplot(df_subset, hue='category')
        plt.savefig('temp_pairplot.png', bbox_inches='tight', dpi=100)
        print("✓ Pairplot created and saved")
        plt.close()
    except Exception as e:
        print(f"⚠️ Plot generation skipped: {str(e)}")


def cleanup():
    """Clean up temporary plot files"""
    import os
    temp_files = [
        'temp_missingness.png', 'temp_histograms.png',
        'temp_boxplots.png', 'temp_correlation.png', 'temp_pairplot.png'
    ]
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
    print("\n🧹 Cleaned up temporary plot files")


if __name__ == "__main__":
    print("\n" + "📈" * 30)
    print("VISUALIZATION DEMO".center(60))
    print("📈" * 30 + "\n")
    
    try:
        demo_missingness_plot()
        demo_histograms()
        demo_boxplots()
        demo_correlation_heatmap()
        demo_pairplot()
    finally:
        cleanup()
    
    print("\n" + "✅" * 30)
    print("ALL DEMOS COMPLETED".center(60))
    print("✅" * 30 + "\n")
    print("\n💡 Note: In interactive environment, plots would be displayed")
