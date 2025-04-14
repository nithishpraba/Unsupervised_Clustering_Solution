import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data_for_training(df):
    """
    Preprocess the mall customers DataFrame for training the clustering model.
    
    This function:
      - Drops missing values.
      - Expects the following numeric columns: "Age", "Annual_Income", and "Spending_Score".
      - Scales the numeric features using StandardScaler.
    
    Returns:
      X_scaled: The scaled feature array.
      df_clean: The cleaned DataFrame (useful for later reference or visualization).
    """
    df_clean = df.dropna().copy()
    
    # Use the exact column names from your CSV
    numeric_columns = ["Age", "Annual_Income", "Spending_Score"]
    
    # Verify that all required columns exist
    if not all(col in df_clean.columns for col in numeric_columns):
        raise Exception("Not all expected numeric columns are found. Found columns: " + str(df_clean.columns.tolist()))
    
    X_numeric = df_clean[numeric_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    return X_scaled, df_clean

def preprocess_data_for_inference(df):
    """
    Preprocess the mall customers DataFrame for inference/visualization.
    
    This function:
      - Drops missing values.
      - Expects that the DataFrame contains the feature columns:
        "Age", "Annual_Income", and "Spending_Score".
      - Scales the features using StandardScaler.
      
    NOTE: In production, you should save the scaler fitted during training and reuse it here.
    
    Returns:
      X_scaled: The scaled feature array.
    """
    df_clean = df.dropna().copy()
    numeric_columns = ["Age", "Annual_Income", "Spending_Score"]
    X_numeric = df_clean[numeric_columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    
    return X_scaled
