"""
Demo: Data Preprocessing Operations
===================================
This demo showcases preprocessing functions in dskit.
"""

from dskit import auto_encode, auto_scale, train_test_auto
import pandas as pd
import numpy as np

def create_sample_data():
    """Create sample dataset for preprocessing"""
    np.random.seed(42)
    return pd.DataFrame({
        'age': np.random.randint(18, 70, 200),
        'salary': np.random.randint(30000, 150000, 200),
        'experience': np.random.randint(0, 30, 200),
        'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing'], 200),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 200),
        'location': np.random.choice(['NYC', 'SF', 'LA', 'Chicago', 'Boston'], 200),
        'performance': np.random.choice(['Low', 'Medium', 'High'], 200)
    })


def demo_auto_encode():
    """Demo 1: Automatic encoding of categorical variables"""
    print("=" * 60)
    print("DEMO 1: Automatic Categorical Encoding")
    print("=" * 60)
    
    df = create_sample_data()
    
    print("\n📊 Original data shape:", df.shape)
    print("📊 Original columns:", list(df.columns))
    print("\n📊 Categorical columns:")
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    for col in cat_cols:
        print(f"  - {col}: {df[col].nunique()} unique values")
    
    print("\n🔧 Applying automatic encoding...")
    print("   (Uses One-Hot for low cardinality, Label for high cardinality)")
    
    df_encoded = auto_encode(df, max_unique_for_onehot=10)
    
    print("\n✓ Encoded data shape:", df_encoded.shape)
    print("✓ New columns:", list(df_encoded.columns))
    print(f"✓ Added {df_encoded.shape[1] - df.shape[1]} new columns")
    
    print("\n📊 Sample of encoded data:")
    print(df_encoded.head())


def demo_auto_scale():
    """Demo 2: Automatic feature scaling"""
    print("\n" + "=" * 60)
    print("DEMO 2: Automatic Feature Scaling")
    print("=" * 60)
    
    df = create_sample_data()
    df_encoded = auto_encode(df)
    
    print("\n📊 Original numeric ranges:")
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns[:3]
    for col in numeric_cols:
        print(f"  {col}: [{df_encoded[col].min():.2f}, {df_encoded[col].max():.2f}]")
    
    # Standard scaling
    print("\n🔧 Applying Standard Scaling...")
    df_standard = auto_scale(df_encoded, method='standard')
    
    print("\n✓ After Standard Scaling:")
    for col in numeric_cols:
        print(f"  {col}: mean={df_standard[col].mean():.4f}, std={df_standard[col].std():.4f}")
    
    # MinMax scaling
    print("\n🔧 Applying MinMax Scaling...")
    df_minmax = auto_scale(df_encoded, method='minmax')
    
    print("\n✓ After MinMax Scaling:")
    for col in numeric_cols:
        print(f"  {col}: [{df_minmax[col].min():.2f}, {df_minmax[col].max():.2f}]")
    
    # Robust scaling
    print("\n🔧 Applying Robust Scaling...")
    df_robust = auto_scale(df_encoded, method='robust')
    
    print("\n✓ After Robust Scaling:")
    for col in numeric_cols:
        print(f"  {col}: median={df_robust[col].median():.4f}")


def demo_train_test_split():
    """Demo 3: Automatic train-test splitting"""
    print("\n" + "=" * 60)
    print("DEMO 3: Train-Test Split")
    print("=" * 60)
    
    df = create_sample_data()
    df_encoded = auto_encode(df)
    df_scaled = auto_scale(df_encoded)
    
    print("\n📊 Full dataset shape:", df_scaled.shape)
    
    print("\n🔧 Splitting data (80-20 split)...")
    X_train, X_test, y_train, y_test = train_test_auto(
        df_scaled, 
        target='performance', 
        test_size=0.2, 
        random_state=42
    )
    
    print("\n✓ Split completed:")
    print(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"  Target distribution in train: {y_train.value_counts().to_dict()}")
    print(f"  Target distribution in test: {y_test.value_counts().to_dict()}")


def demo_complete_pipeline():
    """Demo 4: Complete preprocessing pipeline"""
    print("\n" + "=" * 60)
    print("DEMO 4: Complete Preprocessing Pipeline")
    print("=" * 60)
    
    print("\n📊 Starting with raw data...")
    df = create_sample_data()
    print(f"  Shape: {df.shape}")
    
    print("\n🔧 Step 1: Encoding categorical variables...")
    df_encoded = auto_encode(df, max_unique_for_onehot=10)
    print(f"  ✓ Shape after encoding: {df_encoded.shape}")
    
    print("\n🔧 Step 2: Scaling features...")
    df_scaled = auto_scale(df_encoded, method='standard')
    print(f"  ✓ Shape after scaling: {df_scaled.shape}")
    
    print("\n🔧 Step 3: Splitting into train/test...")
    X_train, X_test, y_train, y_test = train_test_auto(
        df_scaled, 
        target='performance', 
        test_size=0.2, 
        random_state=42
    )
    print(f"  ✓ Train set: {X_train.shape}")
    print(f"  ✓ Test set: {X_test.shape}")
    
    print("\n✅ Preprocessing pipeline completed!")
    print("   Data is now ready for modeling")


if __name__ == "__main__":
    print("\n" + "⚙️" * 30)
    print("PREPROCESSING OPERATIONS DEMO".center(60))
    print("⚙️" * 30 + "\n")
    
    demo_auto_encode()
    demo_auto_scale()
    demo_train_test_split()
    demo_complete_pipeline()
    
    print("\n" + "✅" * 30)
    print("ALL DEMOS COMPLETED".center(60))
    print("✅" * 30 + "\n")
