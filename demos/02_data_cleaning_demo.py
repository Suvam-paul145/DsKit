"""
Demo: Data Cleaning Operations
==============================
This demo showcases all data cleaning functions in dskit.
"""

from dskit import (
    fix_dtypes, rename_columns_auto, replace_specials,
    missing_summary, fill_missing, outlier_summary, 
    remove_outliers, simple_nlp_clean
)
import pandas as pd
import numpy as np

def demo_fix_dtypes():
    """Demo 1: Automatic data type fixing"""
    print("=" * 60)
    print("DEMO 1: Fixing Data Types")
    print("=" * 60)
    
    # Create data with wrong types
    df = pd.DataFrame({
        'id': ['1', '2', '3', '4', '5'],
        'price': ['10.50', '20.75', '15.00', '30.25', '25.50'],
        'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'is_active': ['True', 'False', 'True', 'True', 'False']
    })
    
    print("\n📊 Original data types:")
    print(df.dtypes)
    
    print("\n🔧 Fixing data types...")
    df_fixed = fix_dtypes(df)
    
    print("\n✓ Fixed data types:")
    print(df_fixed.dtypes)


def demo_rename_columns():
    """Demo 2: Automatic column name standardization"""
    print("\n" + "=" * 60)
    print("DEMO 2: Standardizing Column Names")
    print("=" * 60)
    
    # Create data with messy column names
    df = pd.DataFrame({
        'User Name': [1, 2, 3],
        'Email-Address': ['a@b.com', 'c@d.com', 'e@f.com'],
        'Purchase Date (YYYY-MM-DD)': ['2023-01-01', '2023-01-02', '2023-01-03'],
        'Total $$$': [100, 200, 300]
    })
    
    print("\n📊 Original column names:")
    print(list(df.columns))
    
    print("\n🔧 Standardizing column names...")
    df_renamed = rename_columns_auto(df)
    
    print("\n✓ Standardized column names:")
    print(list(df_renamed.columns))


def demo_replace_specials():
    """Demo 3: Replace special characters"""
    print("\n" + "=" * 60)
    print("DEMO 3: Replacing Special Characters")
    print("=" * 60)
    
    df = pd.DataFrame({
        'name': ['John@Doe', 'Jane#Smith', 'Bob$Brown'],
        'email': ['john@example.com', 'jane#test.com', 'bob$mail.com']
    })
    
    print("\n📊 Original data:")
    print(df)
    
    print("\n🔧 Removing special characters...")
    df_clean = replace_specials(df, chars_to_remove=r'[@#$]', replacement='')
    
    print("\n✓ Cleaned data:")
    print(df_clean)


def demo_missing_values():
    """Demo 4: Handle missing values"""
    print("\n" + "=" * 60)
    print("DEMO 4: Missing Values Analysis and Imputation")
    print("=" * 60)
    
    # Create data with missing values
    df = pd.DataFrame({
        'age': [25, np.nan, 35, 40, np.nan, 30],
        'salary': [50000, 60000, np.nan, 80000, 90000, np.nan],
        'city': ['NYC', 'LA', None, 'Chicago', 'NYC', 'LA'],
        'score': [85, 90, 88, np.nan, 92, 87]
    })
    
    print("\n📊 Data with missing values:")
    print(df)
    
    print("\n📈 Missing values summary:")
    summary = missing_summary(df)
    print(summary)
    
    print("\n🔧 Filling missing values (auto strategy)...")
    df_filled = fill_missing(df, strategy='auto')
    
    print("\n✓ Data after filling:")
    print(df_filled)
    print("\n✓ No more missing values:")
    print(df_filled.isnull().sum())


def demo_outliers():
    """Demo 5: Outlier detection and removal"""
    print("\n" + "=" * 60)
    print("DEMO 5: Outlier Detection and Removal")
    print("=" * 60)
    
    # Create data with outliers
    np.random.seed(42)
    df = pd.DataFrame({
        'price': np.concatenate([np.random.normal(100, 10, 95), [500, 600, 700, 800, 900]]),
        'quantity': np.concatenate([np.random.normal(50, 5, 95), [200, 250, 300, 350, 400]])
    })
    
    print(f"\n📊 Dataset size: {len(df)} rows")
    print("\n📈 Outlier summary:")
    summary = outlier_summary(df, method='iqr')
    print(summary)
    
    print("\n🔧 Removing outliers...")
    df_clean = remove_outliers(df, method='iqr', threshold=1.5)
    
    print(f"\n✓ Dataset size after removal: {len(df_clean)} rows")
    print(f"✓ Removed {len(df) - len(df_clean)} outliers")


def demo_nlp_cleaning():
    """Demo 6: Text/NLP cleaning"""
    print("\n" + "=" * 60)
    print("DEMO 6: Text Cleaning")
    print("=" * 60)
    
    df = pd.DataFrame({
        'review': [
            'This is GREAT!!!',
            'terrible   product',
            'LOVE IT <3',
            'bad bad BAD'
        ],
        'comment': [
            'HIGHLY recommend!!!',
            'do not buy',
            'amazing',
            'worst purchase ever'
        ]
    })
    
    print("\n📊 Original text data:")
    print(df)
    
    print("\n🔧 Cleaning text (lowercase, remove extra spaces)...")
    df_clean = simple_nlp_clean(df)
    
    print("\n✓ Cleaned text data:")
    print(df_clean)


if __name__ == "__main__":
    print("\n" + "🧹" * 30)
    print("DATA CLEANING OPERATIONS DEMO".center(60))
    print("🧹" * 30 + "\n")
    
    demo_fix_dtypes()
    demo_rename_columns()
    demo_replace_specials()
    demo_missing_values()
    demo_outliers()
    demo_nlp_cleaning()
    
    print("\n" + "✅" * 30)
    print("ALL DEMOS COMPLETED".center(60))
    print("✅" * 30 + "\n")
