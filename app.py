import streamlit as st
import pandas as pd
import os
import pickle
import subprocess
import numpy as np
import plotly.express as px
from agent import AutoMLAgent
from pipeline import run_automl, generate_shap

# --- Strict Minimalist Dark Theme (Black, Gray, White) ---
def apply_clean_dark_theme():
    st.markdown("""
        <style>
        /* Base Palette */
        .stApp {
            background-color: #000000;
            color: #FFFFFF;
        }

        /* Top Navigation Styling */
        div[data-testid="stHorizontalBlock"] {
            background-color: #000000;
            padding: 10px;
        }

        /* Style the Radio buttons to look like a Nav Bar */
        div[data-testid="stWidgetLabel"] { display: none; }
        
        .stRadio > div {
            flex-direction: row;
            justify-content: center;
            background-color: #000000;
            gap: 20px;
        }

        .stRadio label {
            background-color: #1E1E1E !important;
            color: #FFFFFF !important;
            border: 1px solid #333333 !important;
            padding: 8px 20px !important;
            border-radius: 0px !important;
            font-size: 14px;
            font-weight: 500;
            transition: 0.3s;
        }

        .stRadio label:hover {
            border-color: #FFFFFF !important;
        }

        /* Selected State */
        div[data-testid="stMarkdownContainer"] p { color: #FFFFFF; }
        
        /* Buttons */
        div.stButton > button {
            background-color: #1E1E1E;
            color: #FFFFFF;
            border: 1px solid #333333;
            border-radius: 0px;
            width: 100%;
            height: 45px;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        div.stButton > button:hover {
            background-color: #FFFFFF;
            color: #000000;
            border: 1px solid #FFFFFF;
        }

        /* Inputs */
        .stTextInput input, .stSelectbox div[data-baseweb="select"] {
            background-color: #000000 !important;
            color: #FFFFFF !important;
            border: 1px solid #333333 !important;
            border-radius: 0px !important;
        }

        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}

        /* Horizontal Divider */
        hr {
            border: 0;
            border-top: 1px solid #333333;
            margin: 20px 0;
        }
        </style>
    """, unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="AutoML Agent")
apply_clean_dark_theme()

# --- Top Navigation ---
# Using a radio button forced into a horizontal row to act as a Nav Bar
nav_choice = st.radio(
    "Navigation",
    ["DATASET", "EXPLORATION", "TRAINING", "DEPLOYMENT"],
    horizontal=True
)

st.markdown("---")

# --- Logic Segments ---

if nav_choice == "DATASET":
    st.title("01 // DATASET UPLOAD")
    uploaded_file = st.file_uploader("", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("CSV DATA LINKED")
        st.dataframe(df.head(20), use_container_width=True)

elif nav_choice == "EXPLORATION":
    if "df" in st.session_state:
        st.title("02 // SYSTEM ANALYSIS")
        df = st.session_state.df
        
        # Grid layout for metrics
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("RECORDS", df.shape[0])
        with c2: st.metric("FEATURES", df.shape[1])
        with c3: st.metric("NULLS", df.isna().sum().sum())
        
        # Plotly chart in monochrome
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if numeric_cols:
            fig = px.histogram(
                df, x=numeric_cols[0], 
                template="plotly_dark", 
                color_discrete_sequence=['#FFFFFF']
            )
            fig.update_layout(paper_bgcolor="#000000", plot_bgcolor="#000000")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("AWAITING DATASET...")

elif nav_choice == "TRAINING":
    if "df" in st.session_state:
        st.title("03 // ENGINE TRAINING")
        target = st.selectbox("TARGET VARIABLE", st.session_state.df.columns)
        
        if st.button("EXECUTE AUTOML"):
            with st.spinner("TRAINING..."):
                model, X = run_automl(st.session_state.df, target)
                st.session_state.model_trained = True
                st.success("PROCESS COMPLETE")
    else:
        st.info("AWAITING DATASET...")

elif nav_choice == "DEPLOYMENT":
    st.title("04 // DEPLOYMENT STATUS")
    if os.path.exists("trained_model.pkl"):
        st.code("STATUS: LOCAL_MODEL_ACTIVE", language="bash")
        if st.button("LAUNCH PREDICTOR INTERFACE"):
            subprocess.Popen(["streamlit", "run", "predictor_ui.py"])
            st.toast("Predictor UI Launched")
    else:
        st.warning("NO ACTIVE MODEL DETECTED")
