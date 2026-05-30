import matplotlib.pyplot as plt
import seaborn as sns

def get_case_safe_column(df, target_name):
    """Helper function to find a column name ignoring case sensitivity."""
    for col in df.columns:
        if col.lower() == target_name.lower():
            return col
    return None

def plot_tenure_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    col_x = get_case_safe_column(df, 'tenure')
    col_hue = get_case_safe_column(df, 'churn')
    
    if col_x and col_hue:
        sns.histplot(data=df, x=col_x, hue=col_hue, multiple='stack', kde=True, ax=ax, palette='Set1')
    elif col_x:
        sns.histplot(data=df, x=col_x, kde=True, ax=ax, color='teal')
    ax.set_title('Tenure Distribution Profiles')
    plt.close(fig)
    return fig

def plot_monthly_charges(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    col_x = get_case_safe_column(df, 'total spend') or get_case_safe_column(df, 'monthlycharges')
    col_hue = get_case_safe_column(df, 'churn')
    
    if col_x and col_hue:
        sns.kdeplot(data=df, x=col_x, hue=col_hue, fill=True, common_norm=False, ax=ax, palette='Set1')
    elif col_x:
        sns.kdeplot(data=df, x=col_x, fill=True, ax=ax, color='orange')
    ax.set_title('Financial Density Profile')
    plt.close(fig)
    return fig

def plot_churn_count(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    col_x = get_case_safe_column(df, 'churn')
    if col_x:
        sns.countplot(data=df, x=col_x, ax=ax, palette='pastel')
    ax.set_title('Total Count of Target Class')
    plt.close(fig)
    return fig

def plot_contract_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    col_x = get_case_safe_column(df, 'contract length') or get_case_safe_column(df, 'contract')
    col_hue = get_case_safe_column(df, 'churn')
    
    if col_x and col_hue:
        sns.countplot(data=df, x=col_x, hue=col_hue, ax=ax, palette='viridis')
    elif col_x:
        sns.countplot(data=df, x=col_x, ax=ax, palette='viridis')
    ax.set_title('Class Dynamics Across Structural Tiers')
    plt.close(fig)
    return fig

def plot_heatmap(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    # Remove CustomerID from correlation matrix if it exists
    id_col = get_case_safe_column(numeric_df, 'customerid')
    if id_col and id_col in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=[id_col])
        
    if not numeric_df.empty:
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    ax.set_title('Numerical Feature Correlation Heatmap')
    plt.close(fig)
    return fig

def plot_confusion_matrix_viz(cm):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Retain (0)', 'Churn (1)'], yticklabels=['Retain (0)', 'Churn (1)'], ax=ax)
    ax.set_xlabel('Predicted Classes')
    ax.set_ylabel('True Classes')
    ax.set_title('Ensemble Confusion Matrix')
    plt.close(fig)
    return fig
