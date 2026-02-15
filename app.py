import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
from agent import AutoMLAgent

# --- STYLING: BLACK, GRAY, WHITE ---
def apply_theme():
    st.markdown("""
        <style>
        .stApp { background-color: #000000; color: #FFFFFF; }
        header, footer, #MainMenu { visibility: hidden; }
        
        /* Top Navigation */
        .stRadio > div { flex-direction: row; justify-content: center; gap: 15px; }
        .stRadio label {
            background-color: #000000 !important; color: #888888 !important;
            border: 1px solid #1E1E1E !important; padding: 10px 30px !important;
            border-radius: 0px !important; font-size: 13px; font-weight: 600;
        }
        .stRadio div[role="radiogroup"] input:checked + label {
            background-color: #1E1E1E !important; color: #FFFFFF !important;
            border-color: #FFFFFF !important;
        }

        /* Buttons & Inputs */
        div.stButton > button {
            background-color: #1E1E1E; color: #FFFFFF; border: 1px solid #333333;
            border-radius: 0px; text-transform: uppercase; letter-spacing: 2px;
        }
        div.stButton > button:hover { background-color: #FFFFFF; color: #000000; }
        .stSelectbox div[data-baseweb="select"], .stTextInput input {
            background-color: #000000 !important; color: #FFFFFF !important;
            border: 1px solid #333333 !important; border-radius: 0px !important;
        }
        hr { border-top: 1px solid #1E1E1E; }
        </style>
    """, unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="AutoML Platform")
apply_theme()

# --- Top Navigation ---
nav = st.radio("NAV", ["DATASET", "ANALYSIS", "ENGINE", "DEPLOY"], horizontal=True, label_visibility="collapsed")
st.markdown("---")

if "df" not in st.session_state: st.session_state.df = None

# --- 01 // DATASET ---
if nav == "DATASET":
    st.title("01 // DATASET UPLOAD")
    file = st.file_uploader("", type="csv")
    if file:
        st.session_state.df = pd.read_csv(file)
        st.success("DATA SOURCE LINKED")
        st.dataframe(st.session_state.df.head(15), use_container_width=True)

# --- 02 // ANALYSIS ---
elif nav == "ANALYSIS":
    if st.session_state.df is not None:
        df = st.session_state.df
        st.title("02 // SYSTEM ANALYSIS")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("RECORDS", df.shape[0])
        m2.metric("FEATURES", df.shape[1])
        m3.metric("NULL_VALS", df.isna().sum().sum())

        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### FEATURE TYPE DISTRIBUTION")
            # FIXED: Type Conversion to prevent TypeError
            type_counts = df.dtypes.value_counts().reset_index()
            type_counts.columns = ['Type', 'Count']
            type_counts['Type'] = type_counts['Type'].astype(str)
            
            fig1 = px.bar(type_counts, x='Type', y='Count', template="plotly_dark", color_discrete_sequence=['#FFFFFF'])
            fig1.update_layout(paper_bgcolor="#000000", plot_bgcolor="#000000")
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.write("### DATA SKEWNESS")
            num_cols = df.select_dtypes(include=[np.number])
            if not num_cols.empty:
                skew_df = num_cols.skew().reset_index()
                skew_df.columns = ['Feature', 'Skew']
                skew_df['Feature'] = skew_df['Feature'].astype(str)
                fig2 = px.line(skew_df, x='Feature', y='Skew', template="plotly_dark", color_discrete_sequence=['#FFFFFF'], markers=True)
                fig2.update_layout(paper_bgcolor="#000000", plot_bgcolor="#000000")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("NO NUMERIC DATA")

        st.markdown("---")
        
        col3, col4 = st.columns(2)
        with col3:
            st.write("### NULL MATRIX")
            fig_n, ax_n = plt.subplots(figsize=(10, 4))
            fig_n.patch.set_facecolor('#000000')
            ax_n.set_facecolor('#000000')
            sns.heatmap(df.isnull(), cbar=False, cmap=['#1E1E1E', '#FFFFFF'], ax=ax_n)
            ax_n.tick_params(colors='#FFFFFF', labelsize=7)
            st.pyplot(fig_n)

        with col4:
            st.write("### CORRELATION")
            if not num_cols.empty:
                fig_c = px.imshow(num_cols.corr(), template="plotly_dark", color_continuous_scale=['#000000', '#FFFFFF'])
                fig_c.update_layout(paper_bgcolor="#000000", plot_bgcolor="#000000")
                st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("AWAITING DATASET...")

# --- 03 // ENGINE ---
elif nav == "ENGINE":
    if st.session_state.df is not None:
        st.title("03 // ML ENGINE")
        target = st.selectbox("TARGET COLUMN", st.session_state.df.columns)
        if st.button("RUN AUTOML"):
            with st.status("TRAINING..."):
                # Ensure you have run_automl defined in your pipeline.py
                from pipeline import run_automl
                model, X = run_automl(st.session_state.df, target)
                st.success("ENGINE READY")
    else:
        st.info("AWAITING DATASET...")

# --- 04 // DEPLOY ---
elif nav == "DEPLOY":
    st.title("04 // DEPLOYMENT")
    if os.path.exists("trained_model.pkl"):
        st.code("STATUS: MODEL_ACTIVE", language="bash")
        if st.button("LAUNCH PREDICTOR"):
            subprocess.Popen(["streamlit", "run", "predictor_ui.py"])
            st.toast("Predictor UI Online")
    else:
        st.warning("SYSTEM OFFLINE")
