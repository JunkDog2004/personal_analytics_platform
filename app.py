import streamlit as st
import pandas as pd
import os
import pickle
import subprocess
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
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

        /* Nav Bar Radio Buttons */
        div[data-testid="stWidgetLabel"] { display: none; }
        
        .stRadio > div {
            flex-direction: row;
            justify-content: center;
            background-color: #000000;
            gap: 10px;
        }

        .stRadio label {
            background-color: #000000 !important;
            color: #888888 !important;
            border: 1px solid #1E1E1E !important;
            padding: 10px 30px !important;
            border-radius: 0px !important;
            font-size: 13px;
            font-weight: 600;
            letter-spacing: 1px;
            transition: 0.2s;
        }

        .stRadio label:hover {
            color: #FFFFFF !important;
            border-color: #444444 !important;
        }

        /* Selected Tab State */
        .stRadio div[role="radiogroup"] input:checked + label {
            background-color: #1E1E1E !important;
            color: #FFFFFF !important;
            border-color: #FFFFFF !important;
        }
        
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
            font-size: 12px;
        }

        div.stButton > button:hover {
            background-color: #FFFFFF;
            color: #000000;
            border: 1px solid #FFFFFF;
        }

        /* Inputs & Selectboxes */
        .stSelectbox div[data-baseweb="select"], .stTextInput input {
            background-color: #000000 !important;
            color: #FFFFFF !important;
            border: 1px solid #333333 !important;
            border-radius: 0px !important;
        }

        /* Hide Streamlit elements */
        footer {visibility: hidden;}
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}

        /* Divider */
        hr { border-top: 1px solid #1E1E1E; margin: 20px 0; }
        </style>
    """, unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="AutoML")
apply_clean_dark_theme()

# --- Top Navigation ---
nav_choice = st.radio(
    "Navigation",
    ["DATASET", "ANALYSIS", "ENGINE", "DEPLOY"],
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
        st.success("DATA SOURCE LINKED")
        st.dataframe(df.head(15), use_container_width=True)

elif nav_choice == "ANALYSIS":
    if "df" in st.session_state:
        st.title("02 // SYSTEM ANALYSIS")
        df = st.session_state.df
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RECORDS", df.shape[0])
        m2.metric("FEATURES", df.shape[1])
        m3.metric("NULL_VALS", df.isna().sum().sum())
        m4.metric("DUPLICATES", df.duplicated().sum())

        col1, col2 = st.columns(2)

        with col1:
            st.write("### NULL MATRIX")
            # Graph 1: Missing Values Heatmap (Matplotlib/Seaborn)
            fig_null, ax_null = plt.subplots(figsize=(10, 5))
            fig_null.patch.set_facecolor('#000000')
            ax_null.set_facecolor('#000000')
            sns.heatmap(df.isnull(), cbar=False, cmap=['#1E1E1E', '#FFFFFF'], ax=ax_null)
            ax_null.tick_params(colors='#FFFFFF', labelsize=8)
            st.pyplot(fig_null)

        with col2:
            st.write("### CORRELATION MATRIX")
            # Graph 2: Feature Correlation (Plotly)
            numeric_df = df.select_dtypes(include=[np.number])
            if not numeric_df.empty:
                corr = numeric_df.corr()
                fig_corr = px.imshow(
                    corr, 
                    text_auto=True, 
                    aspect="auto", 
                    color_continuous_scale=['#000000', '#333333', '#FFFFFF'],
                    template="plotly_dark"
                )
                fig_corr.update_layout(
                    paper_bgcolor="#000000", 
                    plot_bgcolor="#000000",
                    margin=dict(l=20, r=20, t=20, b=20)
                )
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.warning("NO NUMERIC DATA FOR CORRELATION")
    else:
        st.info("AWAITING DATASET...")

elif nav_choice == "ENGINE":
    if "df" in st.session_state:
        st.title("03 // ML ENGINE")
        target = st.selectbox("TARGET COLUMN", st.session_state.df.columns)
        
        if st.button("START TRAIN"):
            with st.status("EXECUTING...", expanded=True):
                model, X = run_automl(st.session_state.df, target)
                st.session_state.model_trained = True
                st.success("ENGINE READY")
    else:
        st.info("AWAITING DATASET...")

elif nav_choice == "DEPLOY":
    st.title("04 // DEPLOYMENT")
    if os.path.exists("trained_model.pkl"):
        st.code("STATUS: MODEL_LOADED", language="bash")
        if st.button("RUN PREDICTOR UI"):
            subprocess.Popen(["streamlit", "run", "predictor_ui.py"])
            st.toast("Predictor UI Online")
    else:
        st.warning("SYSTEM OFFLINE: NO MODEL")
