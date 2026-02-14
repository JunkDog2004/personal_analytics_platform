import streamlit as st
import pickle
import pandas as pd
import os
import joblib
import base64

st.set_page_config(layout="centered", page_title="Prediction UI", page_icon="🤖")

# --- Custom CSS for background and professional UI ---
def get_base64(bin_file):
    with open(bin_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_bg_from_local(image_file):
    ext = image_file.split('.')[-1]
    bin_str = get_base64(image_file)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/{ext};base64,{bin_str}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image (make sure the file exists in your directory)
set_bg_from_local("background.jpg")

# --- Custom CSS for professional UI ---
st.markdown(
    """
    <style>
    .main > div {
        background: rgba(255, 255, 255, 0.92);
        border-radius: 18px;
        padding: 2.5rem 2rem 2rem 2rem;
        margin-top: 2rem;
        box-shadow: 0 4px 24px 0 rgba(44, 62, 80, 0.12);
        max-width: 520px;
        margin-left: auto;
        margin-right: auto;
    }
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1e272e;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #2980b9;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6em 1.2em;
        font-weight: 600;
        font-size: 1.1em;
        margin-top: 1em;
        margin-bottom: 1em;
        transition: background 0.2s;
    }
    .stButton>button:hover {
        background-color: #1e6fa6;
    }
    .stTextInput>div>div>input, .stNumberInput>div>input, .stTextArea>div>textarea {
        background-color: #f5f5f5 !important;
        color: #333;
        border-radius: 6px;
        border: 1px solid #d0d0d0;
    }
    #MainMenu, footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center;'>📡 Deployed Prediction UI</h1>", unsafe_allow_html=True)

if st.button("Logout / Close Predictor UI"):
    st.session_state.clear()
    st.success("You have been logged out. You can now close this tab.")
    st.stop()
    
# --- Model Loading ---
if not os.path.exists("trained_model.pkl"):
    st.error("❌ Trained model not found!")
    st.stop()

with open("trained_model.pkl", "rb") as f:
    model = pickle.load(f)

if not os.path.exists("feature_types.pkl"):
    st.error("❌ Feature types not found!")
    st.stop()

with open("feature_types.pkl", "rb") as f:
    feature_types = pickle.load(f)

model_features = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else list(feature_types.keys())

# --- Input Form ---
with st.form("prediction_form"):
    st.markdown("### ✍️ Enter Input Values")
    inputs = {}
    for feature in model_features:
        dtype = feature_types.get(feature, "float64")
        if dtype in ["float64", "float32"]:
            value = st.number_input(f"{feature}", value=0.0, format="%.4f")
        elif dtype in ["int64", "int32"]:
            value = st.number_input(f"{feature}", value=0, step=1)
        elif dtype == "bool":
            value = st.checkbox(f"{feature}")
        else:
            value = st.text_input(f"{feature}")
        inputs[feature] = value

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        input_df = pd.DataFrame([inputs])
        input_df = input_df[model_features]  # Ensure correct column order

        for feature in model_features:
            dtype = feature_types.get(feature, "float64")
            if dtype in ["int64", "int32"]:
                input_df[feature] = pd.to_numeric(input_df[feature], errors="coerce").fillna(0).astype(int)
            elif dtype in ["float64", "float32"]:
                input_df[feature] = pd.to_numeric(input_df[feature], errors="coerce").fillna(0.0).astype(float)
            elif dtype == "bool":
                input_df[feature] = input_df[feature].astype(bool)
            else:
                input_df[feature] = input_df[feature].astype("category")

        prediction = model.predict(input_df)

        label_path = "label_encoder.pkl"
        if os.path.exists(label_path):
            label_encoder = joblib.load(label_path)
            decoded = label_encoder.inverse_transform(prediction)
            st.success("🎯 Prediction:")
            st.markdown(f"<span style='color:#2980b9;font-size:1.2em'><b>{decoded[0]}</b></span>", unsafe_allow_html=True)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_df)
                st.markdown("#### 📊 Probabilities:")
                st.json({label_encoder.classes_[i]: float(prob) for i, prob in enumerate(proba[0])})
        else:
            st.success("🎯 Prediction:")
            st.markdown(f"<span style='color:#2980b9;font-size:1.2em'><b>{prediction[0]}</b></span>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# --- Optional: Add a footer ---
st.markdown(
    """
    <div style='text-align:right; color: #888; font-size: 13px; margin-top: 2em;'>
        Need help? Contact: <a href="mailto:sairamanmathivelan@gmail.com">sairamanmathivelan@gmail.com</a>
    </div>
    """,
    unsafe_allow_html=True
)
