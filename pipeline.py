import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib  # Import joblib for saving models
from flaml import AutoML

def run_automl(df: pd.DataFrame, target_col: str):
    """Runs AutoML on the dataset and returns the trained model and feature data."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Convert object columns to category
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    automl = AutoML()
    task_type = "classification" if y.nunique() <= 20 else "regression"
    metric = "accuracy" if task_type == "classification" else "r2"
    
    
    if task_type == "classification" and not pd.api.types.is_numeric_dtype(y):
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        joblib.dump(label_encoder, "label_encoder.pkl")  # Save encoder
    else:
        y_encoded = y
        label_encoder = None
        
        
    automl = AutoML()

    automl_settings = {
        "time_budget": 60,  # seconds
        "metric": metric,
        "task": task_type,
        "log_file_name": "automl.log",
    }

    automl.fit(X_train=X, y_train=y_encoded, **automl_settings)

    joblib.dump(automl.model, "trained_model.pkl")  # Save the model
    joblib.dump(X.dtypes.to_dict(), "trained_dtypes.pkl")  # Save dtypes for consistent prediction

    return automl.model, X


def predict(model, df: pd.DataFrame):
    """Predict with the trained model, ensuring column types match training."""
    dtype_map = joblib.load("trained_dtypes.pkl")
    for col, dtype in dtype_map.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    preds = model.predict(df)
    
    
    # Decode labels if classification
    if os.path.exists("label_encoder.pkl"):
        label_encoder = joblib.load("label_encoder.pkl")
        preds = label_encoder.inverse_transform(preds)

    return preds


def detect_uninformative_columns(df: pd.DataFrame):
    """Detects columns that are likely uninformative."""
    drop_cols = []
    for col in df.columns:
        if df[col].nunique() == len(df):  # Unique values → ID-like
            drop_cols.append(col)
        elif df[col].dtype == 'object' and df[col].nunique() / len(df) > 0.95:
            drop_cols.append(col)
    return drop_cols


import shap
import matplotlib.pyplot as plt
import os
import pandas as pd

from sklearn.preprocessing import LabelEncoder

def generate_shap(model, X):
    print("🧪 SHAP Debug: Model type =", type(model))
    print("🧪 SHAP Debug: X shape =", X.shape)
    print("🧪 SHAP Debug: Columns =", list(X.columns))

    # Extract raw model from FLAML
    native_model = getattr(model, "model", model)

    # Try to align X with model input features
    try:
        model_features = model.feature_names_in_
        X = X[model_features].copy()
    except AttributeError:
        model_features = list(X.columns)

    # Convert object and category types
    for col in X.columns:
        if X[col].dtype == "object" or pd.api.types.is_categorical_dtype(X[col]):
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        elif pd.api.types.is_integer_dtype(X[col]) or pd.api.types.is_float_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors='coerce')
        else:
            X[col] = X[col].astype(str).astype("category").cat.codes

    X = X.fillna(0)

    try:
        # Use TreeExplainer for LGBM models
        explainer = shap.TreeExplainer(native_model)
        shap_values = explainer.shap_values(X)

        os.makedirs("outputs", exist_ok=True)
        shap.summary_plot(shap_values, X, show=False)
        plt.tight_layout()
        plt.savefig("outputs/shap_plot.png")
        plt.close()
        print("✅ SHAP plot saved to outputs/shap_plot.png")

    except Exception as e:
        print("❌ SHAP generation error:", e)
        raise RuntimeError("SHAP failed: " + str(e))


def plot_target_distribution(df: pd.DataFrame, target_col: str):
    """Plots and saves the target column distribution as an image."""
    plt.figure(figsize=(8, 4))
    try:
        sns.countplot(data=df, x=target_col)
    except Exception:
        sns.histplot(data=df, x=target_col, bins=30)

    plt.title("Target Column Distribution")
    plt.xticks(rotation=45)
    os.makedirs("outputs", exist_ok=True)  # Ensure outputs directory exists
    plt.tight_layout()
    plt.savefig("outputs/target_dist.png")
    plt.close()
