# train.py (Updated metric extraction block)
import pandas as pd
import os
from src.model_training import train_stacking_model
from src.evaluation import evaluate_model

def main():
    print("🚀 Starting Offline Stacking Training Pipeline...")
    
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("❌ Error: Missing datasets! Verify train.csv and test.csv are in data/")
        return

    print("📊 Loading training and testing records...")
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    print("⚙️  Fitting Stacking Classifier Architecture (RandomForest + XGBoost)...")
    print("⏱️  This takes a moment due to internal 5-Fold Stratified Cross-Validation...")
    
    model, X_test, y_test = train_stacking_model(df_train, df_test)
    print("✅ Stacking Model successfully trained and saved to 'models/model.pkl'!")

    print("🧪 Evaluating performance on independent test partition...")
    accuracy, cm, report = evaluate_model(model, X_test, y_test)
    
    print(f"\n🎯 Out-of-Sample Test Accuracy Score: {accuracy:.4f}")
    print("\n📋 Detailed Classification Metrics:")

    # Dynamically find the key for the positive class (Churn = 1)
    # This prevents KeyErrors if keys are stored as '1', 1, or '1.0'
    churn_key = None
    for key in ['1', 1, '1.0', 'Yes', 'yes']:
        if key in report:
            churn_key = key
            break

    if churn_key:
        print(f"   - Precision (Churn): {report[churn_key]['precision']:.2f}")
        print(f"   - Recall (Churn): {report[churn_key]['recall']:.2f}")
        print(f"   - F1-Score (Churn): {report[churn_key]['f1-score']:.2f}")
    else:
        print("   ⚠️ Positive class metric keys not found in standard formats.")
        print("   Available report keys:", list(report.keys()))
        
    print("\n🏁 Process complete. You can now safely launch your Streamlit app!")

if __name__ == "__main__":
    main()