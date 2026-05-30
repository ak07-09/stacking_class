import streamlit as st
import pandas as pd
import joblib
import os

from src.evaluation import evaluate_model
from src.visualization import (
    plot_tenure_distribution,
    plot_monthly_charges,
    plot_churn_count,
    plot_contract_distribution,
    plot_heatmap,
    plot_confusion_matrix_viz
)
from src.prediction import make_prediction

st.set_page_config(page_title='Subscription Churn Stacking Suite', layout='wide')
st.title('📉 Subscription Customer Churn Stacking Classifier Workspace')

# --- CACHED DATA LOADING ---
@st.cache_data
def load_datasets():
    train_path = 'data/train.csv'
    test_path = 'data/test.csv'
    if os.path.exists(train_path) and os.path.exists(test_path):
        return pd.read_csv(train_path), pd.read_csv(test_path)
    return None, None

df_train, df_test = load_datasets()

# --- INSTANT ARTIFACT LOADING ---
@st.cache_resource
def load_production_model():
    model_path = 'models/model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_production_model()

if df_train is not None and df_test is not None:
    # --- NAVIGATION ---
    st.sidebar.header('Navigation Pipeline')
    page = st.sidebar.radio('Go To Stage:', [
        'Dataset Explorer', 
        'Exploratory Data Analysis', 
        'Model Evaluation & Metrics', 
        'Live Inference Engine'
    ])

    # ==========================================
    # 1. DATASET EXPLORER
    # ==========================================
    if page == 'Dataset Explorer':
        st.header('Raw Records Partition Explorer')
        tab1, tab2 = st.tabs(["📋 Training Split (train.csv)", "🧪 Testing Split (test.csv)"])
        with tab1:
            st.dataframe(df_train.head(10), use_container_width=True)
            st.metric(label="Total Training Rows", value=df_train.shape[0])
        with tab2:
            st.dataframe(df_test.head(10), use_container_width=True)
            st.metric(label="Total Testing Rows", value=df_test.shape[0])

        st.subheader("Data Variable Typing Details")
        st.write(df_train.dtypes.astype(str).to_dict())

    # ==========================================
    # 2. EXPLORATORY DATA ANALYSIS
    # ==========================================
    elif page == 'Exploratory Data Analysis':
        st.header('Exploratory Data Analysis Dashboard')
        
        # Dynamically verify columns to prevent plotting errors if names differ slightly
        has_tenure = 'Tenure' in df_train.columns
        has_spend = 'Total Spend' in df_train.columns
        has_contract = 'Contract Length' in df_train.columns
        
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            st.write("### Target Class Balance Status")
            st.pyplot(plot_churn_count(df_train))
        with r1c2:
            st.write("### Numerical System Correlation Matrix")
            st.pyplot(plot_heatmap(df_train))
            
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            st.write("### Customer Account Lifespan Breakdown")
            if has_tenure:
                st.pyplot(plot_tenure_distribution(df_train))
            else:
                st.info("Column 'Tenure' not detected for distribution plotting.")
        with r2c2:
            st.write("### Risk Distribution Across Structural Tiers")
            if has_contract:
                st.pyplot(plot_contract_distribution(df_train))
            elif has_spend:
                st.pyplot(plot_monthly_charges(df_train))
            else:
                st.info("Expected structural category dimensions not discovered.")

    # ==========================================
    # 3. MODEL EVALUATION & METRICS
    # ==========================================
    elif page == 'Model Evaluation & Metrics':
        st.header('Ensemble Performance Metrics')
        if model is None:
            st.error("⚠️ Pre-trained model artifact missing! Run 'python train.py' in your terminal before viewing this tab.")
        else:
            st.success("🤖 Production-ready Stacking Framework successfully read from disk storage.")
            
            from src.data_preprocessing import preprocess_data
            X_test, y_test, _ = preprocess_data(df_test)
            
            # Run test predictions out-of-sample against our pipeline
            accuracy, cm, report = evaluate_model(model, X_test, y_test)
            
            st.subheader("Performance Evaluation Metrics")
            st.metric(label="Independent Test Split Accuracy Score", value=f"{accuracy:.4f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Confusion Matrix")
                st.pyplot(plot_confusion_matrix_viz(cm))
            with col2:
                st.subheader("Detailed Classification Report Metrics")
                st.json(report)

    # ==========================================
    # 4. LIVE INFERENCE ENGINE
    # ==========================================
    elif page == 'Live Inference Engine':
        st.header('Live Inference Profile Engine')
        if model is None:
            st.error("⚠️ Model file not found. Run 'python train.py' in the terminal first.")
        else:
            st.markdown("Enter custom consumer attributes below to run live evaluation against the Stacking ensemble network:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                gender = st.selectbox('Gender', ['Male', 'Female'])
                age = st.number_input('Age', min_value=18, max_value=100, value=35)
                tenure = st.slider('Tenure Months With Firm', min_value=0, max_value=120, value=12)
                subscription_type = st.selectbox('Subscription Type', ['Basic', 'Standard', 'Premium'])
                
            with col2:
                contract_length = st.selectbox('Contract Length Type', ['Month-to-month', 'One year', 'Two year'])
                usage_frequency = st.slider('Usage Frequency (Times/Month)', min_value=0, max_value=50, value=15)
                support_calls = st.number_input('Support Calls Made', min_value=0, max_value=50, value=2)
                
            with col3:
                payment_delay = st.number_input('Payment Delay (Days)', min_value=0, max_value=90, value=0)
                last_interaction = st.slider('Days Since Last Interaction', min_value=0, max_value=30, value=5)
                total_spend = st.number_input('Total Cumulative Spend ($)', min_value=0.0, value=500.0)

            # Map the input dictionary keys to EXACTLY match what the training data used
            # CustomerID is set to a float (0.0) so sklearn's StandardScaler processes it without error
            input_data = {
                'CustomerID': 0.0,
                'Gender': gender,
                'Age': age,
                'Tenure': tenure,
                'Subscription Type': subscription_type,
                'Contract Length': contract_length,
                'Usage Frequency': usage_frequency,
                'Support Calls': support_calls,
                'Payment Delay': payment_delay,
                'Last Interaction': last_interaction,
                'Total Spend': total_spend
            }
            
            if st.button('🔮 Calculate Stacking Prediction'):
                pred, prob = make_prediction(model, input_data)
                
                st.subheader("Ensemble Risk Output Analysis")
                if pred == 1:
                    st.error(f"⚠️ High Risk Warning: Customer is flagged as likely to CHURN. (Attrition Level Probability: {prob:.2%})")
                else:
                    st.success(f"✅ Stable Structural Account: Customer is likely to RETAIN. (Retention Safety Margin Probability: {(1-prob):.2%})")