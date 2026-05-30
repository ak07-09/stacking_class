import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df, target_col='Churn'):
    df_clean = df.copy()
    
    # 1. Standardize column names to fix potential casing issues (e.g., 'churn' vs 'Churn')
    found_target = None
    for col in df_clean.columns:
        if col.lower() == target_col.lower():
            found_target = col
            break
            
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
        
    if 'TotalCharges' in df_clean.columns:
        df_clean['TotalCharges'] = df_clean['TotalCharges'].replace(' ', np.nan)
        df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'])
        df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median())

    # 2. Extract and safely parse the target column
    if found_target and found_target in df_clean.columns:
        # Step A: Drop any row where the target label itself is explicitly missing/NaN
        df_clean = df_clean.dropna(subset=[found_target])
        
        # Step B: Isolate X and y
        X = df_clean.drop(columns=[found_target])
        
        # Step C: Parse Target based on type
        if df_clean[found_target].dtype == 'object':
            y = df_clean[found_target].str.strip().str.capitalize().map({'Yes': 1, 'No': 0})
        else:
            y = df_clean[found_target].astype(float).astype(int)
            
        # Hard check for any accidental missing target rows that mapping caused
        if y.isnull().any():
            valid_idx = y.dropna().index
            X = X.loc[valid_idx]
            y = y.loc[valid_idx]
    else:
        X = df_clean
        y = None
    
    # Clean up any leftover NaN values in the feature space just in case
    for col in X.select_dtypes(include=['int64', 'float64']).columns:
        X[col] = X[col].fillna(X[col].median())
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else 'Unknown')

    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    num_pipeline = Pipeline([('scaler', StandardScaler())])
    cat_pipeline = Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])
    
    return X, y, preprocessor