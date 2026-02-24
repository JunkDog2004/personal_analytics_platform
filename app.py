import streamlit as st
import pandas as pd
import os
import pickle
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import numpy as np

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

# Set Plotly default to dark template
pio.templates.default = "plotly_dark"

def apply_custom_style():
    st.markdown(
        f"""
        <style>
        /* Main background - Soft Dark Slate */
        .stApp {{
            background-color: #1e1e2e;
            color: #cdd6f4;
        }}

        /* Sidebar - Deeper Charcoal */
        [data-testid="stSidebar"] {{
            background-color: #181825;
            border-right: 1px solid #313244;
        }}

        /* Sidebar text color */
        [data-testid="stSidebar"] .stMarkdown {{
            color: #cdd6f4;
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
            background-color: #11111b;
            color: #a6adc8;
            text-align: center;
            padding: 8px;
            font-size: 12px;
            border-top: 1px solid #313244;
            z-index: 100;
        }}

        /* Soft Blue Buttons */
        .stButton>button {{
            background-color: #89b4fa;
            color: #11111b;
            border: none;
            border-radius: 6px;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        .stButton>button:hover {{
            background-color: #b4befe;
            color: #11111b;
            transform: translateY(-1px);
        }}

        /* Inputs and Text Areas */
        .stSelectbox div, .stTextInput input, .stFileUploader {{
            background-color: #313244 !important;
            color: #cdd6f4 !important;
            border-radius: 6px !important;
        }}

        /* Header Styling */
        h1, h2, h3, h4 {{
            color: #f5e0dc;
            font-family: 'Inter', sans-serif;
            font-weight: 700;
        }}

        /* Dataframe styling for dark mode */
        .stDataFrame {{
            border: 1px solid #313244;
            border-radius: 8px;
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
                except Exception as e:
                    st.error(f"Transformation error: {e}")
                    
# --- EDA Function ---
def run_eda(df):
    st.header("🔎 Exploratory Insights")

    # Metrics Row
    m1, m2, m3 = st.columns(3)
    m1.metric("Rows", df.shape[0])
    m2.metric("Columns", df.shape[1])
    m3.metric("Missing Cells", df.isna().sum().sum())

    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 Statistics Summary")
        st.write(df.describe(include="all"))
    with col2:
        st.subheader("🧮 Data Types")
        st.dataframe(df.dtypes.astype(str), use_container_width=True)

    st.divider()

    st.subheader("📉 Missing Values Analysis")
    # Using a dark-friendly colormap
    fig, ax = plt.subplots(figsize=(10, 2), facecolor='#1e1e2e')
    sns.heatmap(df.isnull(), cbar=False, cmap="mako", ax=ax)
    ax.set_facecolor('#1e1e2e')
    st.pyplot(fig)

    st.divider()

    st.subheader("📌 Feature Distribution")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_cols:
        selected_col = st.selectbox("Select Feature", numeric_cols)
        fig = px.histogram(df, x=selected_col, marginal="box", nbins=30, color_discrete_sequence=['#89b4fa'])
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
            
            # Target Distribution
            plot_target_distribution(df, target)
            st.image("outputs/target_dist.png")

            st.markdown("### ⚙️ Training Engine")
            progress = st.progress(0)
            model, X = run_automl(df, target)
            progress.progress(100)

            with open("trained_model.pkl", "wb") as f:
                pickle.dump(model, f)

            st.session_state.model_trained = True
            st.success("Model Training Optimized!")

# --- Deployment Logic ---
if st.session_state.model_trained or os.path.exists("trained_model.pkl"):
    if page in ["Run ML Agent", "Training Status"]:
        st.divider()
        st.subheader("🌐 Service Deployment")
        if st.button("🔌 Launch Predictor API"):
            if not st.session_state.deploy_clicked:
                st.session_state.deploy_clicked = True
                try:
                    subprocess.Popen(["streamlit", "run", "predictor_ui.py"])
                    st.toast("Predictor service online!", icon="🚀")
                except Exception as e:
                    st.error(f"Deployment failed: {e}")

# --- Sidebar Export ---
if os.path.exists("trained_model.pkl"):
    with open("trained_model.pkl", "rb") as f:
        st.sidebar.download_button(
            label="📦 Export Model",
            data=f,
            file_name="trained_model.pkl",
            mime="application/octet-stream",
            use_container_width=True
        )
