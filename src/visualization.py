import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_tenure_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', kde=True, ax=ax, palette='Set1')
    ax.set_title('Tenure Distribution by Churn Status')
    plt.close(fig)
    return fig

def plot_monthly_charges(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.kdeplot(data=df, x='MonthlyCharges', hue='Churn', fill=True, common_norm=False, ax=ax, palette='Set1')
    ax.set_title('Monthly Charges Density Profile')
    plt.close(fig)
    return fig

def plot_churn_count(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x='Churn', ax=ax, palette='pastel')
    ax.set_title('Total Count of Customer Churn Status')
    plt.close(fig)
    return fig

def plot_contract_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x='Contract', hue='Churn', ax=ax, palette='viridis')
    ax.set_title('Churn Rate Distribution across Contract Types')
    plt.close(fig)
    return fig

def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Numeric Feature Correlation Heatmap')
    plt.close(fig)
    return fig

def plot_confusion_matrix_viz(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'], ax=ax)
    ax.set_xlabel('Predicted Classes')
    ax.set_ylabel('True Classes')
    ax.set_title('Ensemble Confusion Matrix')
    plt.close(fig)
    return fig

def plot_feature_importance(model, feature_names):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Extract structural coefficient importances from our Logistic Regression Meta-Learner
    meta_model = model.named_steps['model'].final_estimator_
    importances = meta_model.coef_[0]
    
    # Keep up to top 15 meta weights for clear scaling
    indices = np.argsort(np.abs(importances))[-15:]
    
    ax.barh(range(len(indices)), importances[indices], color='teal', align='center')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Meta-Learner Feature Coefficient Weight')
    ax.set_title('Top Weights from Meta-Learner Layer')
    plt.close(fig)
    return fig