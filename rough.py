import streamlit as st
import pandas as pd
import os
import pickle
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from PIL import Image
import base64

from agent import AutoMLAgent
from pipeline import run_automl, predict, generate_shap, plot_target_distribution, detect_uninformative_columns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_background(full_bg_path, sidebar_bg_path):
    full_bg_ext = full_bg_path.split('.')[-1]
    sidebar_ext = sidebar_bg_path.split('.')[-1]

    full_bg = get_base64(full_bg_path)
    sidebar_bg = get_base64(sidebar_bg_path)

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/{full_bg_ext};base64,{full_bg}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            background-image: url("data:image/{sidebar_ext};base64,{sidebar_bg}");
            background-size: cover;
            background-repeat: no-repeat;
        }}
        footer {{ visibility: hidden; }}
        .footer-container {{
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #2c3e50;
            color: white;
            text-align: right;
            padding: 10px;
            font-size: 14px;
            z-index: 100;
        }}
        .stButton>button {{
            background-color: #2980b9;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.5em 1em;
        }}
        .stSelectbox, .stTextInput>div>div>input, .stTextArea>div>textarea {{
            background-color: #f5f5f5 !important;
            color: #333;
        }}
        .stMarkdown h1, h2, h3, h4, h5 {{
            color: #1e272e;
        }}
        </style>
        <div class="footer-container">
            ℹ️ Need Help? Contact: sairamanmathivelan@gmail.com
        </div>
        """,
        unsafe_allow_html=True
    )

st.set_page_config(layout="wide", page_title="AutoML Deployment Agent", page_icon="🤖")

set_background("background.jpg","sidebar.jpg")
st.title("AutoML Deployment Agent by MSR")

for key in ["df", "model_trained", "deploy_clicked", "target_col"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "df" else False

st.sidebar.header("📌 Navigation")
page = st.sidebar.radio("Select Page", ["Upload Dataset", "Explore Dataset", "Run AutoML", "Training Status", "Retrain Model"])

if page == "Upload Dataset":
    uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ Dataset uploaded successfully!")
        st.dataframe(df.head())

        st.subheader("🧼 Gemini Cleaning Suggestions")
        if st.button("✨ Generate Cleaning Suggestions"):
            agent = AutoMLAgent()
            with st.spinner("Generating suggestions..."):
                suggestion = agent.get_cleaning_suggestion(df)
                code = agent.get_cleaning_code(df)
            st.markdown(suggestion)
            st.code(code, language="python")
            st.session_state.cleaning_code = code

        if "cleaning_code" in st.session_state:
            if st.button("✅ Apply Cleaning Suggestions"):
                try:
                    code = st.session_state.cleaning_code
                    local_vars = {}
                    exec(code, globals(), local_vars)
                    clean_data = local_vars["clean_data"]
                    df_cleaned = clean_data(df)
                    st.session_state.df = df_cleaned
                    st.success("✅ Cleaning applied successfully!")
                    with st.expander("🔍 Preview Cleaned Data"):
                        st.dataframe(df_cleaned.head())
                except Exception as e:
                    st.error(f"Error while applying cleaning code: {e}")

def run_eda(df):
    st.header("🔎 Exploratory Data Analysis")
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head(50))

    st.subheader("📈 Basic Statistics")
    st.write(df.describe(include="all"))

    st.subheader("📉 Missing Values Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df.isnull(), cbar=False, cmap="YlOrRd", ax=ax)
    st.pyplot(fig)

    st.subheader("🧮 Feature Types")
    feature_types = df.dtypes.reset_index()
    feature_types.columns = ["Feature", "Type"]
    st.dataframe(feature_types)

    st.subheader("📌 Feature Distribution")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Select a feature", numeric_cols)
        fig = px.histogram(df, x=selected_col, marginal="box", nbins=30)
        st.plotly_chart(fig)
    else:
        st.warning("No numeric columns available.")

    st.subheader("📉 Correlation Heatmap")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns.")

if st.session_state.df is not None:
    df = st.session_state.df

    if page == "Explore Dataset":
        run_eda(df)

    elif page == "Run AutoML":
        st.subheader("🎯 Select Target Column")
        target = st.selectbox("Select Target Column", df.columns, index=df.columns.get_loc(st.session_state.target_col) if st.session_state.target_col in df.columns else 0)
        st.success(f"✅ Selected: {target}")

        if st.button("🚀 Run AutoML Agent"):
            agent = AutoMLAgent()
            with st.spinner("🔍 Gemini Agent analyzing..."):
                task_type = agent.get_task_type(df)

            st.session_state.target_col = target

            st.markdown(f"### 🧠 Detected Task Type: **{task_type.upper()}**")
            st.markdown("### 📊 Target Distribution")
            plot_target_distribution(df, target)
            st.image("outputs/target_dist.png")

            st.markdown("### ⚙️ Training FLAML Model...")

            import time
            start_time = time.time()

            progress = st.progress(0)
            model, X = run_automl(df, target)
            progress.progress(100)

            with open("trained_model.pkl", "wb") as f:
                pickle.dump(model, f)

            feature_types = X.dtypes.apply(lambda dt: dt.name).to_dict()
            with open("feature_types.pkl", "wb") as f:
                pickle.dump(feature_types, f)

            st.session_state.model_trained = True
            st.success("✅ Model training completed!")
            end_time = time.time()

            st.markdown("### 🧾 Training Summary")
            st.write(f"Model Type: `{model.estimator}`")
            st.write(f"Training Duration: `{end_time - start_time:.2f}` seconds")

            st.markdown("### 📈 SHAP Feature Importance")
            try:
                generate_shap(model, X)
                st.image("outputs/shap_plot.png", caption="SHAP Feature Importance")
            except Exception as e:
                st.error(f"⚠️ Failed to generate SHAP plot: {e}")

        if st.session_state.model_trained:
            st.markdown("### 🚀 Deploy Model")
            if st.button("🔌 Deploy Model"):
                if not st.session_state.deploy_clicked:
                    st.session_state.deploy_clicked = True
                    try:
                        subprocess.Popen(["streamlit", "run", "predictor_ui.py"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        st.toast("✅ Predictor UI launched!", icon="🚀")
                    except Exception as e:
                        st.error(f"❌ Launch failed: {e}")
                else:
                    st.info("ℹ️ Prediction UI is already running.")

    elif page == "Training Status":
        if os.path.exists("trained_model.pkl"):
            st.success("✅ Model is trained and ready.")

            if st.button("📈 Show SHAP Plot"):
                if os.path.exists("outputs/shap_plot.png"):
                    st.image("outputs/shap_plot.png")
                else:
                    st.warning("⚠️ SHAP plot not found.")

            if st.button("🔌 Deploy Model"):
                if not st.session_state.deploy_clicked:
                    st.session_state.deploy_clicked = True
                    try:
                        subprocess.Popen(["streamlit", "run", "predictor_ui.py"], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
                        st.toast("✅ Predictor UI launched!", icon="🚀")
                    except Exception as e:
                        st.error(f"❌ Launch failed: {e}")
                else:
                    st.info("ℹ️ Prediction UI is already running.")
        else:
            st.warning("⚠️ Model is not yet trained.")

    elif page == "Retrain Model":
        if os.path.exists("trained_model.pkl"):
            if st.button("🔁 Retrain Now"):
                with open("trained_model.pkl", "rb") as f:
                    model = pickle.load(f)
                st.warning("🔄 Retraining logic placeholder: Implement as needed.")
