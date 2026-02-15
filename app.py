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
        .stApp { background-color: #000000; color: #FFFFFF; }
        
        /* Top Navigation */
        div[data-testid="stHorizontalBlock"] { background-color: #000000; padding: 10px; }
        div[data-testid="stWidgetLabel"] { display: none; }
        .stRadio > div { flex-direction: row; justify-content: center; gap: 10px; }
        .stRadio label {
            background-color: #000000 !important;
            color: #888888 !important;
            border: 1px solid #1E1E1E !important;
            padding: 10px 30px !important;
            border-radius: 0px !important;
            font-size: 13px;
            font-weight: 600;
        }
        .stRadio div[role="radiogroup"] input:checked + label {
            background-color: #1E1E1E !important;
            color: #FFFFFF !important;
            border-color: #FFFFFF !important;
        }
        
        /* UI Components */
        div.stButton > button {
            background-color: #1E1E1E; color: #FFFFFF;
            border: 1px solid #333333; border-radius: 0px;
            text-transform: uppercase; letter-spacing: 2px;
        }
        div.stButton > button:hover { background-color: #FFFFFF; color: #000000; }
        .stSelectbox div[data-baseweb="select"], .stTextInput input {
            background-color: #000000 !important; color: #FFFFFF !important;
            border: 1px solid #333333 !important; border-radius: 0px !important;
        }
        
        /* Hide Branding */
        footer {visibility: hidden;}
        header {visibility: hidden;}
        #MainMenu {visibility: hidden;}
        hr { border-top: 1px solid #1E1E1E; }
        </style>
    """, unsafe_allow_html=True)

st.set_page_config(layout="wide", page_title="AutoML")
apply_clean_dark_theme()

nav_choice = st.radio("Navigation", ["DATASET", "ANALYSIS", "ENGINE", "DEPLOY"], horizontal=True)
st.markdown("---")

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
        
        # Row 1: Key Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RECORDS", df.shape[0])
        m2.metric("FEATURES", df.shape[1])
        m3.metric("NULL_VALS", df.isna().sum().sum())
        m4.metric("DATATYPES", len(df.dtypes.unique()))

        st.markdown("<br>", unsafe_allow_html=True)

        # Row 2: Charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### FEATURE TYPE DISTRIBUTION")
            # Bar Chart: Count of Data Types
            type_counts = df.dtypes.value_counts().astype(str).reset_index()
            type_counts.columns = ['Type', 'Count']
            fig_bar = px.bar(
                type_counts, x='Type', y='Count',
                template="plotly_dark",
                color_discrete_sequence=['#FFFFFF']
            )
            fig_bar.update_layout(paper_bgcolor="#000000", plot_bgcolor="#000000", showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            st.write("### DATA SKEWNESS")
            # Graph: Skewness of Numeric Features
            numeric_cols = df.select_dtypes(include=[np.number])
            if not numeric_cols.empty:
                skew_values = numeric_cols.skew().reset_index()
                skew_values.columns = ['Feature', 'Skew']
                fig_skew = px.line(
                    skew_values, x='Feature', y='Skew',
                    template="plotly_dark",
                    color_discrete_sequence=['#FFFFFF'],
                    markers=True
                )
                fig_skew.update_layout(paper_bgcolor="#000000", plot_bgcolor="#000000")
                st.plotly_chart(fig_skew, use_container_width=True)
            else:
                st.info("NO NUMERIC DATA TO ANALYZE SKEW")

        st.markdown("---")
        
        # Row 3: Heatmaps
        col3, col4 = st.columns(2)
        with col3:
            st.write("### MISSING DATA ARCHITECTURE")
            fig_null, ax_null = plt.subplots(figsize=(10, 4))
            fig_null.patch.set_facecolor('#000000')
            ax_null.set_facecolor('#000000')
            sns.heatmap(df.isnull(), cbar=False, cmap=['#1E1E1E', '#FFFFFF'], ax=ax_null)
            ax_null.tick_params(colors='#FFFFFF', labelsize=7)
            st.pyplot(fig_null)

        with col4:
            st.write("### CORRELATION HEATMAP")
            if not numeric_cols.empty:
                corr = numeric_cols.corr()
                fig_corr = px.imshow(
                    corr, template="plotly_dark",
                    color_continuous_scale=['#000000', '#FFFFFF']
                )
                fig_corr.update_layout(paper_bgcolor="#000000", plot_bgcolor="#000000")
                st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("AWAITING DATASET...")

elif nav_choice == "ENGINE":
    st.title("03 // ML ENGINE")
    # ... Training Logic ...

elif nav_choice == "DEPLOY":
    st.title("04 // DEPLOYMENT")
    # ... Deployment Logic ...
