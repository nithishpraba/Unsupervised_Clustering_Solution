import streamlit as st
import joblib
import os
import pandas as pd
import plotly.express as px
from utils.logger import get_logger
from utils.preprocessing import preprocess_data_for_inference

logger = get_logger(__name__)

st.title("Mall Customers Clustering App")
st.write("This app clusters mall customers based on Age, Annual Income, and Spending Score.")

# Load the trained clustering model
model_path = os.path.join("models", "model.pkl")
try:
    model = joblib.load(model_path)
    st.success("Clustering model loaded successfully!")
    logger.info("Model loaded from %s", model_path)
except Exception as e:
    st.error("Failed to load the clustering model. Please run the training script first.")
    logger.error("Error loading model: %s", e)
    st.stop()

# Load the dataset for visualization
data_path = os.path.join("data", "mall_customers.csv")
try:
    df = pd.read_csv(data_path)
    st.write("Dataset Preview:")
    st.dataframe(df.head())
except Exception as e:
    st.error("Failed to load the dataset.")
    st.error(e)
    st.stop()

# Preprocess the data for clustering (using numeric columns)
try:
    df_clean = df.dropna().copy()
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    numeric_columns = ["Age", "Annual_Income", "Spending_Score"]
    X_numeric = df_clean[numeric_columns]
    X_scaled = scaler.fit_transform(X_numeric)
except Exception as e:
    st.error("Error during data preprocessing.")
    st.error(e)
    st.stop()

# Predict cluster labels using the loaded model
try:
    labels = model.predict(X_scaled)
    df["Cluster"] = labels
    st.write("Data with Cluster Assignments:")
    st.dataframe(df.head())
except Exception as e:
    st.error("Error during clustering prediction.")
    st.error(e)
    st.stop()

# Visualization: allow the user to select two features to plot clusters
st.subheader("Cluster Visualization")
columns = numeric_columns  # Use the three numeric columns
x_axis = st.selectbox("Select X-axis", columns, index=0)
y_axis = st.selectbox("Select Y-axis", columns, index=1)

if x_axis and y_axis:
    try:
        fig = px.scatter(df, x=x_axis, y=y_axis, color="Cluster", title="Cluster Visualization")
        st.plotly_chart(fig)
    except Exception as e:
        st.error("Error generating visualization.")
        st.error(e)
