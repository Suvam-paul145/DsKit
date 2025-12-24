"""
Demo: AutoML and Hyperparameter Optimization
============================================
This demo showcases AutoML functions in dskit.
"""

from dskit import (
    auto_tune_model, get_default_param_space,
    auto_encode, auto_scale, train_test_auto
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

def create_sample_data():
    """Create sample dataset"""
    np.random.seed(42)
    n = 300
    
    df = pd.DataFrame({
        'feature_1': np.random.normal(50, 10, n),
        'feature_2': np.random.normal(100, 20, n),
        'feature_3': np.random.exponential(5, n),
        'feature_4': np.random.uniform(0, 100, n),
        'category': np.random.choice(['A', 'B'], n)
    })
    
    # Create target
    df['target'] = (
        (df['feature_1'] > 50) & 
        (df['feature_2'] > 100)
    ).astype(int)
    
    return df


def demo_default_param_space():
    """Demo 1: Get default parameter spaces"""
    print("=" * 60)
    print("DEMO 1: Default Parameter Spaces")
    print("=" * 60)
    
    models = ['rf', 'xgb', 'lgbm', 'lr']
    
    for model in models:
        print(f"\n📋 Parameter space for {model.upper()}:")
        try:
            params = get_default_param_space(model, task='classification')
            for param, values in params.items():
                if isinstance(values, list):
                    print(f"   {param}: {values}")
                else:
                    print(f"   {param}: {values}")
        except Exception as e:
            print(f"   ⚠️ {str(e)}")


def demo_random_search():
    """Demo 2: Random search optimization"""
    print("\n" + "=" * 60)
    print("DEMO 2: Random Search Optimization")
    print("=" * 60)
    
    df = create_sample_data()
    df_encoded = auto_encode(df)
    df_scaled = auto_scale(df_encoded)
    X_train, X_test, y_train, y_test = train_test_auto(df_scaled, target='target')
    
    print("\n🔧 Running Random Search for Random Forest...")
    print("   Method: random")
    print("   Max evaluations: 20")
    
    best_model, best_params, best_score = auto_tune_model(
        RandomForestClassifier,
        X_train, y_train,
        method='random',
        max_evals=20,
        task='classification',
        model_name='rf'
    )
    
    print("\n✓ Optimization completed:")
    print(f"   Best score: {best_score:.4f}")
    print(f"   Best parameters:")
    for param, value in best_params.items():
        print(f"      {param}: {value}")


def demo_grid_search():
    """Demo 3: Grid search optimization"""
    print("\n" + "=" * 60)
    print("DEMO 3: Grid Search Optimization")
    print("=" * 60)
    
    df = create_sample_data()
    df_encoded = auto_encode(df)
    df_scaled = auto_scale(df_encoded)
    X_train, X_test, y_train, y_test = train_test_auto(df_scaled, target='target')
    
    print("\n🔧 Running Grid Search for Logistic Regression...")
    print("   Method: grid")
    
    best_model, best_params, best_score = auto_tune_model(
        LogisticRegression,
        X_train, y_train,
        method='grid',
        task='classification',
        model_name='lr'
    )
    
    print("\n✓ Optimization completed:")
    print(f"   Best score: {best_score:.4f}")
    print(f"   Best parameters:")
    for param, value in best_params.items():
        print(f"      {param}: {value}")


def demo_bayesian_optimization():
    """Demo 4: Bayesian optimization"""
    print("\n" + "=" * 60)
    print("DEMO 4: Bayesian Optimization")
    print("=" * 60)
    
    df = create_sample_data()
    df_encoded = auto_encode(df)
    df_scaled = auto_scale(df_encoded)
    X_train, X_test, y_train, y_test = train_test_auto(df_scaled, target='target')
    
    print("\n🔧 Running Bayesian Optimization for Random Forest...")
    print("   Method: bayesian")
    print("   Max evaluations: 15")
    
    try:
        best_model, best_params, best_score = auto_tune_model(
            RandomForestClassifier,
            X_train, y_train,
            method='bayesian',
            max_evals=15,
            task='classification',
            model_name='rf'
        )
        
        print("\n✓ Optimization completed:")
        print(f"   Best score: {best_score:.4f}")
        print(f"   Best parameters:")
        for param, value in best_params.items():
            print(f"      {param}: {value}")
    except Exception as e:
        print(f"\n⚠️ Note: {str(e)}")
        print("   (Some optimization methods may require additional packages)")


def demo_compare_optimization_methods():
    """Demo 5: Compare optimization methods"""
    print("\n" + "=" * 60)
    print("DEMO 5: Compare Optimization Methods")
    print("=" * 60)
    
    df = create_sample_data()
    df_encoded = auto_encode(df)
    df_scaled = auto_scale(df_encoded)
    X_train, X_test, y_train, y_test = train_test_auto(df_scaled, target='target')
    
    methods = ['random', 'grid']
    results = {}
    
    for method in methods:
        print(f"\n🔧 Testing {method.upper()} search...")
        try:
            best_model, best_params, best_score = auto_tune_model(
                RandomForestClassifier,
                X_train, y_train,
                method=method,
                max_evals=10,
                task='classification',
                model_name='rf'
            )
            results[method] = best_score
            print(f"   ✓ {method}: {best_score:.4f}")
        except Exception as e:
            print(f"   ⚠️ {method}: {str(e)}")
    
    if results:
        print("\n📊 Comparison Results:")
        for method, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"   {method.capitalize()}: {score:.4f}")


if __name__ == "__main__":
    print("\n" + "🤖" * 30)
    print("AUTOML & OPTIMIZATION DEMO".center(60))
    print("🤖" * 30 + "\n")
    
    demo_default_param_space()
    demo_random_search()
    demo_grid_search()
    demo_bayesian_optimization()
    demo_compare_optimization_methods()
    
    print("\n" + "✅" * 30)
    print("ALL DEMOS COMPLETED".center(60))
    print("✅" * 30 + "\n")
