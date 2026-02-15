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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --- Modern Dark Theme CSS ---
def apply_modern_style():
    st.markdown("""
        <style>
        /* Main Background and Text */
        .stApp {
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #ffffff;
        }

        /* Glassmorphism Effect for Cards */
        div.stButton > button {
            width: 100%;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            background: rgba(255, 255, 255, 0.05);
            color: white;
            padding: 10px;
            transition: all 0.3s ease;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        div.stButton > button:hover {
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid #00d2ff;
            box-shadow: 0px 0px 15px rgba(0, 210, 255, 0.5);
            color: #00d2ff;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: rgba(0, 0, 0, 0.4) !important;
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Input Fields */
        .stSelectbox div[data-baseweb="select"], .stTextInput input {
            background-color: rgba(255, 255, 255, 0.05) !important;
            color: white !important;
            border-radius: 10px !important;
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
        }

        /* Metric/Card Styling */
        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.05);
            padding: 15px;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Custom Titles */
        h1, h2, h3 {
            font-family: 'Inter', sans-serif;
            font-weight: 700;
            background: -webkit-linear-gradient(#00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        /* Dataframe background fix */
        .stDataFrame {
            background: rgba(255, 255, 255, 0.02);
            border-radius: 10px;
        }

        /* Footer Custom */
        .footer {
            position: fixed;
            bottom: 10px;
            right: 20px;
            font-size: 12px;
            color: rgba(255,255,255,0.4);
        }
        </style>
        <div class="footer">🚀 AutoML Agent v2.0 | sairamanmathivelan@gmail.com</div>
    """, unsafe_allow_html=True)

# --- Configuration ---
st.set_page_config(layout="wide", page_title="AutoML Deployment Agent", page_icon="🤖")
apply_modern_style()

# --- Session State ---
for key in ["df", "model_trained", "deploy_clicked", "target_col", "selected_page"]:
    if key not in st.session_state:
        st.session_state[key] = pages[0] if key == "selected_page" else (None if key == "df" else False)

# --- Sidebar Navigation ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=80) # Use a generic AI icon
    st.title("Navigation")
    pages = ["Upload Dataset", "Explore Dataset", "Run ML Agent", "Training Status", "Retrain Model"]
    
    for p in pages:
        if st.button(p, key=f"nav_{p}"):
            st.session_state.selected_page = p

page = st.session_state.selected_page

# --- Main Logic Sections ---

if page == "Upload Dataset":
    st.title("📂 Data Acquisition")
    uploaded_file = st.file_uploader("Drop your CSV here", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ Dataset Linked Successfully")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(df.head(10), use_container_width=True)
        
        with col2:
            st.markdown("### ✨ Smart Cleaning")
            if st.button("Generate AI Suggestions"):
                agent = AutoMLAgent()
                with st.spinner("Analyzing data structure..."):
                    suggestion = agent.get_cleaning_suggestion(df)
                    code = agent.get_cleaning_code(df)
                    st.session_state.cleaning_code = code
                    st.info(suggestion)
            
            if "cleaning_code" in st.session_state:
                if st.button("Apply Code & Clean"):
                    try:
                        local_vars = {}
                        exec(st.session_state.cleaning_code, globals(), local_vars)
                        st.session_state.df = local_vars["clean_data"](df)
                        st.success("Data Refined!")
                    except Exception as e:
                        st.error(f"Error: {e}")

elif page == "Explore Dataset":
    if st.session_state.df is not None:
        st.title("🔎 Deep Insight EDA")
        df = st.session_state.df
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Rows", df.shape[0])
        m2.metric("Total Columns", df.shape[1])
        m3.metric("Missing Values", df.isna().sum().sum())

        tab1, tab2, tab3 = st.tabs(["📊 Distributions", "🔗 Correlations", "📋 Summary"])
        
        with tab1:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select Feature to Visualize", numeric_cols)
                fig = px.histogram(df, x=selected_col, marginal="box", template="plotly_dark", color_discrete_sequence=['#00d2ff'])
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r', template="plotly_dark")
                st.plotly_chart(fig_corr, use_container_width=True)
        
        with tab3:
            st.dataframe(df.describe().T, use_container_width=True)
    else:
        st.warning("Please upload a dataset first.")

elif page == "Run ML Agent":
    if st.session_state.df is not None:
        st.title("🧠 Model Factory")
        df = st.session_state.df
        
        target = st.selectbox("Choose Target Variable", df.columns)
        st.session_state.target_col = target

        if st.button("🚀 Execute AutoML Training"):
            agent = AutoMLAgent()
            with st.status("Training in progress...", expanded=True) as status:
                st.write("Detecting task type...")
                task_type = agent.get_task_type(df)
                st.write(f"Task: {task_type}")
                
                st.write("Optimizing Hyperparameters...")
                model, X = run_automl(df, target)
                
                with open("trained_model.pkl", "wb") as f:
                    pickle.dump(model, f)
                
                feature_types = X.dtypes.apply(lambda dt: dt.name).to_dict()
                with open("feature_types.pkl", "wb") as f:
                    pickle.dump(feature_types, f)
                
                st.session_state.model_trained = True
                status.update(label="Training Complete!", state="complete", expanded=False)

            st.balloons()
            
            # Results UI
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Model Performance")
                st.code(f"Algorithm: {model.estimator}")
            with c2:
                st.subheader("Explainability")
                generate_shap(model, X)
                st.image("outputs/shap_plot.png")
    else:
        st.warning("Upload data to start training.")

# --- Shared Deployment Footer ---
if st.session_state.model_trained:
    st.divider()
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.info("💡 Model is ready for production deployment.")
    with col_b:
        if st.button("🔌 Launch Predictor UI"):
            subprocess.Popen(["streamlit", "run", "predictor_ui.py"])
            st.toast("Predictor UI is live!", icon="🔥")
