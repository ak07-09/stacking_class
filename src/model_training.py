import os
import joblib
from sklearn.pipeline import Pipeline
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from src.data_preprocessing import preprocess_data

def train_stacking_model(train_df, test_df):
    # Setup preprocessor using training data dimensions
    X_train, y_train, preprocessor = preprocess_data(train_df)
    X_test, y_test, _ = preprocess_data(test_df)
    
    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, eval_metric='logloss', random_state=42))
    ]
    
    meta_learner = LogisticRegression(C=0.1, random_state=42)
    
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        stack_method='predict_proba',
        cv=5,
        n_jobs=-1
    )
    
    full_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', stacking_clf)
    ])
    
    # Fit on the entire dedicated training set
    full_pipeline.fit(X_train, y_train)
    
    os.makedirs('models', exist_ok=True)
    joblib.dump(full_pipeline, 'models/model.pkl')
    
    return full_pipeline, X_test, y_test