import streamlit as st
import pandas as pd
import os
import pickle
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import base64

from agent import AutoMLAgent
from pipeline import run_automl, predict, generate_shap, plot_target_distribution, detect_uninformative_columns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- BRANDING CONFIGURATION ---
APP_TITLE = "Personal Analytics Platform"
DEVELOPER_NAME = "" 
CONTACT_EMAIL = ""    
FOOTER_TEXT = f"Analytics Dashboard | {DEVELOPER_NAME} {CONTACT_EMAIL}"

# --- APP CONFIGURATION ---
st.set_page_config(layout="wide", page_title=APP_TITLE, page_icon="📊")

def apply_custom_style():
    st.markdown(
        f"""
        <style>
        /* Main background - Clean Analytics Look */
        .stApp {{
            background-color: #f8f9fa;
        }}

        /* Professional Sidebar */
        [data-testid="stSidebar"] {{
            background-color: #ffffff;
            border-right: 1px solid #e0e0e0;
        }}

        /* Modern Custom Footer */
        footer {{
            visibility: hidden;
        }}
        .footer-container {{
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #ffffff;
            color: #636e72;
            text-align: center;
            padding: 8px;
            font-size: 12px;
            border-top: 1px solid #e0e0e0;
            z-index: 100;
        }}

        /* Buttons and Inputs */
        .stButton>button {{
            background-color: #0984e3;
            color: white;
            border: none;
            border-radius: 4px;
            font-weight: 500;
            transition: all 0.3s;
        }}
        .stButton>button:hover {{
            background-color: #74b9ff;
            color: white;
        }}

        .stSelectbox, .stTextInput input {{
            border-radius: 4px !important;
        }}

        /* Header Styling */
        h1, h2, h3 {{
            color: #2d3436;
            font-family: 'Inter', sans-serif;
        }}
        </style>

        <div class="footer-container">
            {FOOTER_TEXT}
        </div>
        """,
        unsafe_allow_html=True
    )

apply_custom_style()

st.title(f"🚀 {APP_TITLE}")
if DEVELOPER_NAME:
    st.caption(f"Developed by {DEVELOPER_NAME}")

# --- Session State Initialization ---
for key in ["df", "model_trained", "deploy_clicked", "target_col"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "df" else False

# --- Sidebar Navigation ---
st.sidebar.header("🧭 Navigation")

pages = ["Upload Dataset", "Explore Dataset", "Run ML Agent", "Training Status", "Retrain Model"]

if "selected_page" not in st.session_state:
    st.session_state.selected_page = pages[0]

for p in pages:
    if st.sidebar.button(p, key=f"nav_{p}", use_container_width=True):
        st.session_state.selected_page = p
        
page = st.session_state.selected_page

# --- Dataset Upload ---
if page == "Upload Dataset":
    st.subheader("📁 Data Ingestion")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ Dataset loaded successfully!")
        st.dataframe(df.head(), use_container_width=True)

        st.divider()
        st.subheader("✨ Intelligent Data Cleaning")
        if st.button("Generate Cleaning Suggestions"):
            agent = AutoMLAgent()
            with st.spinner("Analyzing data structure..."):
                suggestion = agent.get_cleaning_suggestion(df)
                code = agent.get_cleaning_code(df)
            st.markdown(suggestion)
            st.code(code, language="python")
            st.session_state.cleaning_code = code

        if "cleaning_code" in st.session_state:
            if st.button("Apply Cleaning Suggestions"):
                try:
                    code = st.session_state.cleaning_code
                    local_vars = {}
                    exec(code, globals(), local_vars)
                    clean_data = local_vars["clean_data"]
                    df_cleaned = clean_data(df)
                    st.session_state.df = df_cleaned
                    st.success("✅ Transformation applied!")
                    with st.expander("🔍 Preview Cleaned Data"):
                        st.dataframe(df_cleaned.head())
                except Exception as e:
                    st.error(f"Transformation error: {e}")
                    
# --- EDA Function ---
def run_eda(df):
    st.header("🔎 Exploratory Insights")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Statistics")
        st.write(df.describe(include="all"))
    with col2:
        st.subheader("🧮 Data Schema")
        feature_types = df.dtypes.reset_index()
        feature_types.columns = ["Feature", "Type"]
        st.dataframe(feature_types, use_container_width=True)

    st.divider()

    st.subheader("📉 Missing Values Analysis")
    fig, ax = plt.subplots(figsize=(10, 2))
    sns.heatmap(df.isnull(), cbar=False, cmap="Blues", ax=ax)
    st.pyplot(fig)

    st.divider()

    st.subheader("📌 Feature Distribution")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Select Feature to Visualize", numeric_cols)
        fig = px.histogram(df, x=selected_col, marginal="box", nbins=30, color_discrete_sequence=['#0984e3'])
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📉 Correlation Matrix")
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

# --- Main Control Logic ---
if st.session_state.df is not None:
    df = st.session_state.df

    if page == "Explore Dataset":
        run_eda(df)

    elif page == "Run ML Agent":
        st.subheader("🎯 Model Target Selection")
        target = st.selectbox("Define Target Variable", df.columns, index=df.columns.get_loc(st.session_state.target_col) if st.session_state.target_col in df.columns else 0)

        if st.button("🚀 Execute AutoML Pipeline"):
            agent = AutoMLAgent()
            with st.spinner("Agent evaluating task..."):
                task_type = agent.get_task_type(df)

            st.session_state.target_col = target
            st.info(f"Detected Task Type: {task_type.upper()}")
            
            plot_target_distribution(df, target)
            st.image("outputs/target_dist.png")

            st.markdown("### ⚙️ Training Engine")
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
            end_time = time.time()
            st.success(f"Model Training Optimized in {end_time - start_time:.2f}s")
            
            try:
                generate_shap(model, X)
                st.image("outputs/shap_plot.png", caption="Feature Importance Analysis")
            except:
                st.warning("Feature importance visualization bypassed.")

    elif page == "Training Status":
        if os.path.exists("trained_model.pkl"):
            st.success("✅ Model Artifacts Ready")
            if st.button("📈 View Model Insights"):
                if os.path.exists("outputs/shap_plot.png"):
                    st.image("outputs/shap_plot.png")

    elif page == "Retrain Model":
        if os.path.exists("trained_model.pkl"):
            if st.button("🔁 Reset & Retrain Pipeline"):
                st.info("Pipeline resetting...")
        else:
            st.warning("No active model found in workspace.")

# --- Deployment Logic (Unified) ---
if st.session_state.model_trained or os.path.exists("trained_model.pkl"):
    if page in ["Run ML Agent", "Training Status", "Retrain Model"]:
        st.divider()
        st.subheader("🌐 Service Deployment")
        if st.button("🔌 Launch Predictor API & UI"):
            if not st.session_state.deploy_clicked:
                st.session_state.deploy_clicked = True
                try:
                    subprocess.Popen(["streamlit", "run", "predictor_ui.py"])
                    st.toast("Predictor service online!", icon="🚀")
                except Exception as e:
                    st.error(f"Deployment failed: {e}")
            else:
                st.info("Predictor service is already active.")

# --- Export Model ---
if os.path.exists("trained_model.pkl"):
    with open("trained_model.pkl", "rb") as f:
        st.sidebar.download_button(
            label="📦 Export Trained Model",
            data=f,
            file_name="trained_model.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )
