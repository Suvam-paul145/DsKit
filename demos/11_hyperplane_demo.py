"""
Demo: Hyperplane Visualization
==============================
This demo showcases hyperplane visualization functions in dskit.
"""

from dskit import (
    Hyperplane, HyperplaneExtractor,
    plot_svm_hyperplane, plot_logistic_hyperplane,
    extract_hyperplane
)
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Perceptron
import matplotlib.pyplot as plt

def create_2d_classification_data():
    """Create 2D linearly separable data"""
    np.random.seed(42)
    
    # Class 0
    X0 = np.random.randn(50, 2) + np.array([2, 2])
    y0 = np.zeros(50)
    
    # Class 1
    X1 = np.random.randn(50, 2) + np.array([-2, -2])
    y1 = np.ones(50)
    
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])
    
    return X, y


def demo_hyperplane_class():
    """Demo 1: Hyperplane class basics"""
    print("=" * 60)
    print("DEMO 1: Hyperplane Class")
    print("=" * 60)
    
    # Create a simple hyperplane: x + y = 1
    coefficients = np.array([1, 1])
    intercept = -1
    
    print("\n📐 Creating hyperplane: x + y = 1")
    hyperplane = Hyperplane(coefficients, intercept)
    
    print(f"\n✓ Hyperplane created:")
    print(f"   Coefficients: {hyperplane.coefficients}")
    print(f"   Intercept: {hyperplane.intercept}")
    print(f"   Dimensions: {hyperplane.n_dimensions}")
    
    # Test some points
    test_points = np.array([[0, 1], [1, 0], [2, 2]])
    print(f"\n📊 Testing points:")
    for point in test_points:
        distance = hyperplane.distance_to(point)
        side = "above" if distance > 0 else "below" if distance < 0 else "on"
        print(f"   Point {point}: distance={distance:.4f} ({side} hyperplane)")


def demo_svm_hyperplane():
    """Demo 2: SVM hyperplane visualization"""
    print("\n" + "=" * 60)
    print("DEMO 2: SVM Hyperplane Visualization")
    print("=" * 60)
    
    X, y = create_2d_classification_data()
    
    print("\n🤖 Training SVM classifier...")
    model = SVC(kernel='linear', C=1.0, random_state=42)
    model.fit(X, y)
    
    print("✓ SVM trained")
    print(f"   Support vectors: {len(model.support_vectors_)}")
    
    print("\n📊 Creating hyperplane visualization...")
    try:
        plot_svm_hyperplane(model, X, y)
        plt.savefig('temp_svm_hyperplane.png', bbox_inches='tight', dpi=100)
        print("✓ SVM hyperplane plot created")
        plt.close()
    except Exception as e:
        print(f"⚠️ Plot generation skipped: {str(e)}")


def demo_logistic_hyperplane():
    """Demo 3: Logistic regression hyperplane"""
    print("\n" + "=" * 60)
    print("DEMO 3: Logistic Regression Hyperplane")
    print("=" * 60)
    
    X, y = create_2d_classification_data()
    
    print("\n🤖 Training Logistic Regression...")
    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    
    print("✓ Logistic Regression trained")
    print(f"   Accuracy: {model.score(X, y):.4f}")
    
    print("\n📊 Creating hyperplane visualization...")
    try:
        plot_logistic_hyperplane(model, X, y)
        plt.savefig('temp_logistic_hyperplane.png', bbox_inches='tight', dpi=100)
        print("✓ Logistic regression hyperplane plot created")
        plt.close()
    except Exception as e:
        print(f"⚠️ Plot generation skipped: {str(e)}")


def demo_hyperplane_extraction():
    """Demo 4: Extract hyperplane from models"""
    print("\n" + "=" * 60)
    print("DEMO 4: Hyperplane Extraction")
    print("=" * 60)
    
    X, y = create_2d_classification_data()
    
    # Train multiple models
    models = {
        'SVM': SVC(kernel='linear', C=1.0, random_state=42),
        'Logistic': LogisticRegression(random_state=42),
        'Perceptron': Perceptron(random_state=42)
    }
    
    print("\n🤖 Training models and extracting hyperplanes...")
    
    for name, model in models.items():
        model.fit(X, y)
        
        print(f"\n📐 {name}:")
        try:
            extractor = extract_hyperplane(model)
            hp = extractor.get_hyperplane()
            
            print(f"   ✓ Hyperplane extracted")
            print(f"   Coefficients: {hp.coefficients}")
            print(f"   Intercept: {hp.intercept:.4f}")
            print(f"   Norm: {np.linalg.norm(hp.coefficients):.4f}")
        except Exception as e:
            print(f"   ⚠️ {str(e)}")


def demo_hyperplane_comparison():
    """Demo 5: Compare hyperplanes from different algorithms"""
    print("\n" + "=" * 60)
    print("DEMO 5: Hyperplane Comparison")
    print("=" * 60)
    
    X, y = create_2d_classification_data()
    
    # Train multiple models
    print("\n🤖 Training multiple classifiers...")
    
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X, y)
    
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X, y)
    
    models_dict = {
        'SVM': svm_model,
        'Logistic Regression': lr_model
    }
    
    print("✓ Models trained")
    
    print("\n📊 Comparing hyperplanes...")
    for name, model in models_dict.items():
        print(f"\n   {name}:")
        print(f"      Accuracy: {model.score(X, y):.4f}")
        try:
            extractor = extract_hyperplane(model)
            hp = extractor.get_hyperplane()
            print(f"      Coefficients: [{hp.coefficients[0]:.4f}, {hp.coefficients[1]:.4f}]")
            print(f"      Intercept: {hp.intercept:.4f}")
        except:
            pass


def cleanup():
    """Clean up temporary files"""
    import os
    temp_files = ['temp_svm_hyperplane.png', 'temp_logistic_hyperplane.png']
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)
    print("\n🧹 Cleaned up temporary files")


if __name__ == "__main__":
    print("\n" + "📐" * 30)
    print("HYPERPLANE VISUALIZATION DEMO".center(60))
    print("📐" * 30 + "\n")
    
    try:
        demo_hyperplane_class()
        demo_svm_hyperplane()
        demo_logistic_hyperplane()
        demo_hyperplane_extraction()
        demo_hyperplane_comparison()
    finally:
        cleanup()
    
    print("\n" + "✅" * 30)
    print("ALL DEMOS COMPLETED".center(60))
    print("✅" * 30 + "\n")
    print("\n💡 Note: In interactive environment, plots would be displayed")
