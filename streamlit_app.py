import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import time
import json 
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
from streamlit.components.v1 import html
import requests

# ✅ Page Config
st.set_page_config(
    page_title="InSightAI",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- Custom CSS ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    h1 {
        color: #f0f8ff;
        text-align: center;
        font-size: 3rem;
    }
    .stButton > button {
        background-color: #00c6ff;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stFileUploader {
        background-color: #1f1f1f;
        border-radius: 8px;
    }
    .result-box {
        background-color: #ffffff22;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    .fade-in {
        animation: fadeIn 2s;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    /* Fix white block around Lottie animation */
    div[data-testid="st_lottie"] {
        background-color: transparent !important;
        padding: 0 !important;
        margin: 0 auto !important;
        box-shadow: none !important;
        border: none !important;
        max-width: none !important;
    }
    div[data-testid="st_lottie"] * {
        background-color: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }
    </style>
""", unsafe_allow_html=True)


# Load Lottie animation from local JSON file
@st.cache_resource
def load_lottie_local(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

lottie_ai = load_lottie_local("animations/ai_loader.json")

st_lottie(lottie_ai, height=200)


# --- Load Model Once ---
@st.cache_resource
def load_cnn_model():
    return load_model("cifar10_cnn_model.h5")

model = load_cnn_model()

# CIFAR-10 Labels
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# --- App Header ---
st.markdown("<div class='fade-in'><h1>🧠 InSightAI : Upload.Predict.Done.</h1></div>", unsafe_allow_html=True)
st.write("Upload any image and the model will predict its class with confidence.")

# Theme Switch
theme = st.sidebar.radio("Choose Theme", ["Dark", "Light"])
if theme == "Light":
    st.markdown("""
        <style>
        .stApp { background: Grey; color: white; }
        h1 { color: #003366; }
        </style>
    """, unsafe_allow_html=True)

# --- Upload Image ---
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], accept_multiple_files=False)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.markdown("### 📷 Uploaded Image:")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    image = image.resize((32, 32))
    img_array = np.array(image) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # Loading animation
    with st.spinner('Processing with InSightAI...'):
        time.sleep(1.5)
        prediction = model.predict(img_batch)[0]

    predicted_index = np.argmax(prediction)
    confidence = prediction[predicted_index]

    # --- Display Result ---
    st.markdown("---")
    st.markdown("### 🎯 Prediction Result")
    st.markdown(f"<div class='result-box'><h3>{class_names[predicted_index].capitalize()}</h3></div>", unsafe_allow_html=True)

    # Gauge Chart for Confidence
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Confidence"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#d40f0f"},
        }
    ))
    st.plotly_chart(fig)
    st.write(f"**Confidence:** {confidence * 100:.2f}%")

    # --- Expander: See All Probabilities ---
    with st.expander("🔍 See all class probabilities"):
        card_html = """
        <style>
        .card-grid {
            display: flex;
            flex-wrap: wrap;
            justify-content: center;
            gap: 1rem;
            padding: 10px;
        }
        .prob-card {
            background-color: #ffffff22;
            border-radius: 12px;
            padding: 1rem;
            width: 150px;
            color: white;
            text-align: center;
            transition: transform 0.2s, background-color 0.3s;
        }
        .prob-card:hover {
            transform: scale(1.05);
            background-color: #00c6ff44;
        }
        </style>
        <div class="card-grid">
        """

        for idx, score in enumerate(prediction):
            class_name = class_names[idx].capitalize()
            confidence_pct = f"{score * 100:.2f}%"
            card_html += f"""
            <div class="prob-card">
                <strong>{class_name}</strong><br/>
                <span>{confidence_pct}</span>
            </div>
            """

        card_html += "</div>"

        html(card_html, height=350)

    # --- History Log ---
    if 'history' not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        'image': uploaded_file,
        'class': class_names[predicted_index].capitalize(),
        'confidence': f"{confidence * 100:.2f}%"
    })

    st.markdown("---")
    st.markdown("### 🕒 Prediction History")
    for item in st.session_state.history[-3:]:  # last 3
        st.image(item['image'], width=100)
        st.write(f"Class: {item['class']}, Confidence: {item['confidence']}")
