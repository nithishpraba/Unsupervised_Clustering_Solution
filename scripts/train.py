import os
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from utils.logger import get_logger
from utils.preprocessing import preprocess_data_for_training

logger = get_logger(__name__)

def train_model():
    # Load dataset from the data folder
    data_path = os.path.join("data", "mall_customers.csv")
    try:
        df = pd.read_csv(data_path)
        logger.info("Dataset loaded. Shape: %s", df.shape)
    except Exception as e:
        logger.error("Error loading data from %s: %s", data_path, e)
        return None

    # Preprocess the data for training
    try:
        X_scaled, df_clean = preprocess_data_for_training(df)
        logger.info("Data preprocessing complete.")
    except Exception as e:
        logger.error("Error during preprocessing: %s", e)
        return None

    # Train a KMeans clustering model (for example, with 3 clusters)
    try:
        model = KMeans(n_clusters=3, random_state=42)
        model.fit(X_scaled)
        logger.info("KMeans model training complete.")
    except Exception as e:
        logger.error("Error during model training: %s", e)
        return None

    return model

def main():
    model = train_model()
    if model is not None:
        os.makedirs("models", exist_ok=True)
        model_path = os.path.join("models", "model.pkl")
        joblib.dump(model, model_path)
        logger.info("Model saved to %s", model_path)
        print("Training complete and model saved at", model_path)
    else:
        print("Training failed. See log for details.")

if __name__ == "__main__":
    main()
