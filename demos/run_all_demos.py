"""
Run All Demos - Master Script
==============================
This script runs all dskit demo files in sequence.
"""

import sys
import importlib.util
import os
from pathlib import Path

# Demo files in order
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


def run_demo(demo_file):
    """Run a single demo file"""
    demo_path = Path(__file__).parent / demo_file
    
    if not demo_path.exists():
        print(f"❌ Demo file not found: {demo_file}")
        return False
    
    try:
        # Load and execute the demo module
        spec = importlib.util.spec_from_file_location("demo_module", demo_path)
        demo_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(demo_module)
        return True
    except Exception as e:
        print(f"❌ Error running {demo_file}: {str(e)}")
        return False


def main():
    """Run all demos"""
    print("\n" + "=" * 70)
    print("DSKIT COMPREHENSIVE DEMO SUITE".center(70))
    print("=" * 70 + "\n")
    
    print("This will run all 12 demo files in sequence.")
    print("Each demo showcases different functionality of dskit.\n")
    
    # Option to run specific demos
    print("Options:")
    print("  1. Run all demos (1-12)")
    print("  2. Run core demos (1-6)")
    print("  3. Run advanced demos (7-12)")
    print("  4. Run specific demo")
    print("  5. Exit\n")
    
    choice = input("Enter your choice (1-5): ").strip()
    
    if choice == '1':
        demos_to_run = DEMO_FILES
    elif choice == '2':
        demos_to_run = DEMO_FILES[:6]
    elif choice == '3':
        demos_to_run = DEMO_FILES[6:]
    elif choice == '4':
        print("\nAvailable demos:")
        for i, (file, desc) in enumerate(DEMO_DESCRIPTIONS.items(), 1):
            print(f"  {i:2d}. {desc}")
        demo_num = input("\nEnter demo number (1-12): ").strip()
        try:
            demo_idx = int(demo_num) - 1
            if 0 <= demo_idx < len(DEMO_FILES):
                demos_to_run = [DEMO_FILES[demo_idx]]
            else:
                print("Invalid demo number!")
                return
        except ValueError:
            print("Invalid input!")
            return
    elif choice == '5':
        print("Exiting...")
        return
    else:
        print("Invalid choice!")
        return
    
    # Run selected demos
    total = len(demos_to_run)
    success_count = 0
    failed_demos = []
    
    print(f"\n{'=' * 70}")
    print(f"Running {total} demo(s)...".center(70))
    print(f"{'=' * 70}\n")
    
    for i, demo_file in enumerate(demos_to_run, 1):
        description = DEMO_DESCRIPTIONS.get(demo_file, "Unknown Demo")
        
        print(f"\n{'#' * 70}")
        print(f"DEMO {i}/{total}: {description}".center(70))
        print(f"File: {demo_file}".center(70))
        print(f"{'#' * 70}\n")
        
        if run_demo(demo_file):
            success_count += 1
            print(f"\n✅ Demo {i}/{total} completed successfully!\n")
        else:
            failed_demos.append((i, demo_file, description))
            print(f"\n❌ Demo {i}/{total} failed!\n")
        
        # Pause between demos (except for last one)
        if i < total:
            input("\nPress Enter to continue to next demo...")
    
    # Summary
    print("\n" + "=" * 70)
    print("DEMO SUITE SUMMARY".center(70))
    print("=" * 70 + "\n")
    
    print(f"Total demos run: {total}")
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed: {len(failed_demos)}")
    
    if failed_demos:
        print("\nFailed demos:")
        for num, file, desc in failed_demos:
            print(f"  {num}. {desc} ({file})")
    else:
        print("\n🎉 All demos completed successfully!")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Demo suite interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
