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
import time

from agent import AutoMLAgent
from pipeline import run_automl, predict, generate_shap, plot_target_distribution, detect_uninformative_columns

# --- Page Config ---
st.set_page_config(
    layout="wide", 
    page_title="AutoML Agent Pro", 
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

# --- Modern Styling (Glassmorphism & Clean Refinement) ---
def local_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #2D3436;
        }

        /* Background Gradient */
        .stApp {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: rgba(255, 255, 255, 0.8);
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.3);
        }

        /* Modern Button Styling */
        .stButton>button {
            width: 100%;
            border-radius: 12px;
            border: none;
            height: 3em;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
            color: #fff;
        }

        /* Card-like containers for data */
        div[data-testid="stExpander"], .stDataFrame, .element-container {
            background-color: white;
            border-radius: 15px;
            padding: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.02);
        }

        /* Status & Metric Improvements */
        [data-testid="stMetricValue"] {
            font-size: 1.8rem;
            font-weight: 700;
            color: #4834d4;
        }
        
        /* Custom Footer */
        .footer {
            position: fixed;
            bottom: 10px;
            right: 20px;
            font-size: 12px;
            color: #636e72;
            background: rgba(255,255,255,0.5);
            padding: 5px 15px;
            border-radius: 20px;
        }
        </style>
        
        <div class="footer">🚀 Powered by Gemini & FLAML | Dev: MSR</div>
    """, unsafe_allow_html=True)

local_css()

# --- Session State Initialization ---
for key in ["df", "model_trained", "deploy_clicked", "target_col", "selected_page"]:
    if key not in st.session_state:
        st.session_state[key] = None if key == "df" else (pages[0] if key == "selected_page" else False)

# --- Sidebar Navigation (Modernized) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80)
    st.title("AutoML Pro")
    st.markdown("---")
    
    pages = ["📂 Upload Dataset", "🔍 Explore Data", "🧠 Train Model", "📊 Insights", "⚙️ Management"]
    
    # Selection using a cleaner radio or individual buttons
    selection = st.radio("Navigation", pages, label_visibility="collapsed")
    st.session_state.selected_page = selection
    
    st.markdown("---")
    if st.session_state.df is not None:
        st.success("Dataset Loaded")
    if st.session_state.model_trained:
        st.info("Model: Ready ✅")

page = st.session_state.selected_page

# --- Page: Upload Dataset ---
if "Upload" in page:
    st.header("📂 Data Onboarding")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader("Drop your CSV file here", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.dataframe(df.head(10), use_container_width=True)

    with col2:
        if st.session_state.df is not None:
            st.markdown("### ✨ AI Pre-processing")
            if st.button("Generate Cleaning Plan"):
                agent = AutoMLAgent()
                with st.spinner("Gemini is analyzing structure..."):
                    suggestion = agent.get_cleaning_suggestion(st.session_state.df)
                    code = agent.get_cleaning_code(st.session_state.df)
                    st.session_state.cleaning_code = code
                    st.markdown(suggestion)
            
            if "cleaning_code" in st.session_state:
                if st.button("Apply Auto-Clean"):
                    # Execution logic remains same
                    local_vars = {}
                    exec(st.session_state.cleaning_code, globals(), local_vars)
                    st.session_state.df = local_vars["clean_data"](st.session_state.df)
                    st.toast("Data Refined!", icon="🪄")

# --- Page: Explore Dataset ---
elif "Explore" in page:
    if st.session_state.df is not None:
        st.header("🔍 Visual Intelligence")
        df = st.session_state.df
        
        tab1, tab2, tab3 = st.tabs(["Overview", "Distributions", "Correlations"])
        
        with tab1:
            m1, m2, m3 = st.columns(3)
            m1.metric("Rows", df.shape[0])
            m2.metric("Columns", df.shape[1])
            m3.metric("Missing Values", df.isna().sum().sum())
            st.dataframe(df.describe(), use_container_width=True)
            
        with tab2:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            selected_col = st.selectbox("Select Feature to Analyze", numeric_cols)
            fig = px.histogram(df, x=selected_col, marginal="violin", template="plotly_white", color_discrete_sequence=['#764ba2'])
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Please upload a dataset first.")

# --- Page: Train Model ---
elif "Train" in page:
    if st.session_state.df is not None:
        st.header("🧠 Training Engine")
        df = st.session_state.df
        
        col1, col2 = st.columns([1, 2])
        with col1:
            target = st.selectbox("Select Target Variable", df.columns)
            st.session_state.target_col = target
            
            if st.button("🚀 Start AutoML Pipeline"):
                agent = AutoMLAgent()
                with st.status("Training in progress...", expanded=True) as status:
                    st.write("Detecting task type...")
                    task_type = agent.get_task_type(df)
                    
                    st.write("Optimizing Hyperparameters (FLAML)...")
                    model, X = run_automl(df, target)
                    
                    # Saving logic
                    with open("trained_model.pkl", "wb") as f:
                        pickle.dump(model, f)
                    
                    st.session_state.model_trained = True
                    status.update(label="Training Complete!", state="complete", expanded=False)
                
                st.balloons()
        
        with col2:
            if st.session_state.model_trained:
                st.success(f"Model trained successfully! (Target: {target})")
                st.markdown("### Target Distribution")
                plot_target_distribution(df, target)
                st.image("outputs/target_dist.png")
    else:
        st.warning("Upload data to enable training.")

# --- Page: Insights ---
elif "Insights" in page:
    if st.session_state.model_trained:
        st.header("📊 Model Explainability")
        if os.path.exists("outputs/shap_plot.png"):
            st.image("outputs/shap_plot.png", caption="Feature Importance (SHAP values)")
        else:
            st.info("Run training to see importance plots.")
    else:
        st.error("No trained model found.")

# --- Page: Management ---
elif "Management" in page:
    st.header("⚙️ Deployment & Export")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Cloud Deployment")
        if st.button("🔌 Launch Predictor UI"):
            if not st.session_state.deploy_clicked:
                subprocess.Popen(["streamlit", "run", "predictor_ui.py"])
                st.session_state.deploy_clicked = True
                st.toast("Deployment active!")
            else:
                st.info("Predictor is already live.")
                
    with c2:
        st.subheader("Local Export")
        if os.path.exists("trained_model.pkl"):
            with open("trained_model.pkl", "rb") as f:
                st.download_button("⬇️ Download .PKL Model", f, "model.pkl")
