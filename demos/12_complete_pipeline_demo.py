"""
Demo: Complete End-to-End ML Pipeline
=====================================
This demo showcases a complete ML workflow using dskit.
"""

from dskit import dskit
import pandas as pd
import numpy as np

def create_comprehensive_dataset():
    """Create a comprehensive dataset for end-to-end demo"""
    np.random.seed(42)
    n = 500
    
    # Generate dates
    from datetime import datetime, timedelta
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i % 365) for i in range(n)]
    
    df = pd.DataFrame({
        'date': dates,
        'age': np.random.randint(18, 70, n),
        'income': np.random.randint(30000, 150000, n),
        'experience': np.random.randint(0, 30, n),
        'department': np.random.choice(['IT', 'HR', 'Sales', 'Marketing', 'Finance'], n),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n),
        'location': np.random.choice(['NYC', 'SF', 'LA', 'Chicago', 'Boston'], n),
        'satisfaction_score': np.random.uniform(1, 10, n).round(2),
        'projects_completed': np.random.randint(0, 50, n),
        'rating': np.random.uniform(1, 5, n).round(1)
    })
    
    # Introduce missing values (realistic scenario)
    missing_indices = np.random.choice(df.index, size=int(n * 0.05), replace=False)
    df.loc[missing_indices, 'income'] = np.nan
    
    missing_indices = np.random.choice(df.index, size=int(n * 0.03), replace=False)
    df.loc[missing_indices, 'satisfaction_score'] = np.nan
    
    # Create target variable
    df['performance'] = np.where(
        (df['satisfaction_score'].fillna(5) > 7) & 
        (df['projects_completed'] > 20) &
        (df['experience'] > 5),
        'High',
        np.where(
            (df['satisfaction_score'].fillna(5) > 4) & 
            (df['projects_completed'] > 10),
            'Medium',
            'Low'
        )
    )
    
    return df


def demo_complete_pipeline():
    """Demo: Complete end-to-end ML pipeline"""
    print("=" * 60)
    print("COMPLETE END-TO-END ML PIPELINE")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n📥 STEP 1: Data Loading")
    print("-" * 60)
    df = create_comprehensive_dataset()
    print(f"✓ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    
    # Step 2: Initial EDA
    print("\n📊 STEP 2: Initial Exploration")
    print("-" * 60)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
    print(f"\n  Data types:")
    print(df.dtypes.value_counts())
    print(f"\n  Missing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0])
    
    # Step 3: Data Cleaning
    print("\n🧹 STEP 3: Data Cleaning")
    print("-" * 60)
    
    from dskit import fix_dtypes, fill_missing, rename_columns_auto
    
    print("  → Fixing data types...")
    df = fix_dtypes(df)
    print("    ✓ Data types fixed")
    
    print("  → Standardizing column names...")
    df = rename_columns_auto(df)
    print("    ✓ Column names standardized")
    
    print("  → Filling missing values...")
    df = fill_missing(df, strategy='auto')
    print("    ✓ Missing values imputed")
    print(f"    Remaining missing: {df.isnull().sum().sum()}")
    
    # Step 4: Feature Engineering
    print("\n🔧 STEP 4: Feature Engineering")
    print("-" * 60)
    
    from dskit import create_date_features, create_binning_features
    
    print("  → Extracting date features...")
    df = create_date_features(df, date_cols=['date'])
    print("    ✓ Date features extracted")
    
    print("  → Creating binned features...")
    df = create_binning_features(df, numeric_cols=['age', 'income'], n_bins=5)
    print("    ✓ Binned features created")
    
    print(f"\n  Total features after engineering: {len(df.columns)}")
    
    # Step 5: Preprocessing
    print("\n⚙️ STEP 5: Data Preprocessing")
    print("-" * 60)
    
    from dskit import auto_encode, auto_scale, train_test_auto
    
    print("  → Encoding categorical variables...")
    df_encoded = auto_encode(df, max_unique_for_onehot=10)
    print(f"    ✓ Encoded. Shape: {df_encoded.shape}")
    
    print("  → Scaling features...")
    df_scaled = auto_scale(df_encoded, method='standard')
    print(f"    ✓ Scaled. Shape: {df_scaled.shape}")
    
    print("  → Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_auto(
        df_scaled, target='performance', test_size=0.2, random_state=42
    )
    print(f"    ✓ Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Step 6: Model Training
    print("\n🤖 STEP 6: Model Training")
    print("-" * 60)
    
    from dskit import compare_models
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    
    print("  → Training and comparing multiple models...")
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    results = compare_models(X_train, y_train, X_test, y_test, models=models, task='classification')
    print("\n    ✓ Model comparison completed")
    print(results)
    
    # Step 7: Model Evaluation
    print("\n📈 STEP 7: Model Evaluation")
    print("-" * 60)
    
    from dskit import evaluate_model
    
    print("  → Training best model (Random Forest)...")
    best_model = RandomForestClassifier(n_estimators=100, random_state=42)
    best_model.fit(X_train, y_train)
    
    print("  → Evaluating model...")
    metrics = evaluate_model(best_model, X_test, y_test, task='classification')
    print("\n    ✓ Evaluation metrics:")
    for metric, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"      {metric}: {value:.4f}")
    
    # Step 8: Model Interpretation
    print("\n🔍 STEP 8: Model Interpretation")
    print("-" * 60)
    
    print("  → Analyzing feature importance...")
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importances
        }).sort_values('importance', ascending=False).head(10)
        
        print("\n    ✓ Top 10 important features:")
        for idx, row in feature_importance.iterrows():
            print(f"      {row['feature']}: {row['importance']:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"✓ Data loaded: {len(df)} samples")
    print(f"✓ Features engineered: {df_scaled.shape[1]} total features")
    print(f"✓ Models compared: {len(models)}")
    print(f"✓ Best model accuracy: {best_model.score(X_test, y_test):.4f}")
    print("\n🎉 End-to-end pipeline completed successfully!")


if __name__ == "__main__":
    print("\n" + "🚀" * 30)
    print("END-TO-END ML PIPELINE DEMO".center(60))
    print("🚀" * 30 + "\n")
    
    demo_complete_pipeline()
    
    print("\n" + "✅" * 30)
    print("DEMO COMPLETED".center(60))
    print("✅" * 30 + "\n")
