import streamlit as st
import pandas as pd
import os
import pickle
import subprocess
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from agent import AutoMLAgent
from pipeline import run_automl, generate_shap, plot_target_distribution

# --- Strict Monochromatic Dark Theme ---
def apply_minimal_dark_style():
    st.markdown("""
        <style>
        /* Base Colors: Black (#000000), Gray (#1E1E1E), White (#FFFFFF) */
        .stApp {
            background-color: #000000;
            color: #FFFFFF;
        }

        /* Top Navigation Bar */
        .nav-container {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            background-color: #1E1E1E;
            padding: 10px 50px;
            display: flex;
            justify-content: center;
            border-bottom: 1px solid #333333;
            z-index: 1000;
        }

        /* Fix Content Padding for Top Nav */
        .main .block-container {
            padding-top: 100px;
        }

        /* Button Styling (Gray & White) */
        div.stButton > button {
            background-color: #1E1E1E;
            color: #FFFFFF;
            border: 1px solid #333333;
            border-radius: 4px;
            transition: all 0.2s ease;
        }

        div.stButton > button:hover {
            background-color: #FFFFFF;
            color: #000000;
            border: 1px solid #FFFFFF;
        }

        /* Sidebar Removal (Optional - Hiding default sidebar elements) */
        [data-testid="stSidebar"] {
            display: none;
        }

        /* Input Customization */
        input, select, .stSelectbox div[data-baseweb="select"] {
            background-color: #1E1E1E !important;
            color: white !important;
            border: 1px solid #333333 !important;
        }

        /* Metric Cards */
        div[data-testid="metric-container"] {
            background-color: #1E1E1E;
            border: 1px solid #333333;
            padding: 20px;
            border-radius: 5px;
        }

        /* Minimal Footer */
        .footer {
            text-align: center;
            padding: 20px;
            font-size: 12px;
            color: #555555;
            border-top: 1px solid #1E1E1E;
        }
        </style>
    """, unsafe_allow_html=True)

# --- App Config ---
st.set_page_config(layout="wide", page_title="AutoML Agent", page_icon="🤖")
apply_minimal_dark_style()

# --- Top Navigation Logic ---
# Since Streamlit doesn't have a native 'Top Nav', we use columns to simulate it.
pages = ["Upload", "Explore", "Train", "Status"]
if "current_page" not in st.session_state:
    st.session_state.current_page = "Upload"

# Render Navigation Bar
nav_cols = st.columns(len(pages))
for i, p in enumerate(pages):
    if nav_cols[i].button(p, use_container_width=True):
        st.session_state.current_page = p

st.divider() # Line separator under navigation
page = st.session_state.current_page

# --- Logic Sections ---

if page == "Upload":
    st.header("📂 Data Upload")
    uploaded_file = st.file_uploader("", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("Dataset Loaded.")
        st.dataframe(df.head(10), use_container_width=True)

elif page == "Explore":
    if "df" in st.session_state:
        st.header("🔎 Analysis")
        df = st.session_state.df
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Features", df.shape[1])
        c3.metric("Missing", df.isna().sum().sum())

        # Dark Plotly Charts
        fig = px.histogram(df, x=df.columns[0], template="plotly_dark", color_discrete_sequence=['#FFFFFF'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data found. Go to Upload.")

elif page == "Train":
    if "df" in st.session_state:
        st.header("🧠 Model Training")
        target = st.selectbox("Select Target Column", st.session_state.df.columns)
        
        if st.button("Start Training"):
            with st.status("Processing...", expanded=True):
                # Placeholder for your actual AutoML logic
                st.write("Initializing Agent...")
                model, X = run_automl(st.session_state.df, target)
                st.session_state.model_trained = True
                st.success("Model Complete")
    else:
        st.warning("Upload a dataset first.")

elif page == "Status":
    st.header("📈 System Status")
    if os.path.exists("trained_model.pkl"):
        st.write("✅ Primary Model: `Ready` (trained_model.pkl)")
        if st.button("Launch Predictor UI"):
            subprocess.Popen(["streamlit", "run", "predictor_ui.py"])
    else:
        st.write("⚪ Status: `No Model Found`")

st.markdown('<div class="footer">MSR AUTOML AGENT | 2026</div>', unsafe_allow_html=True)
