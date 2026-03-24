import os
import pickle
import json
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(
    page_title="CropSense - AI Crop Recommendation System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<head>
<meta name="description" content="CropSense - Free AI crop recommendation using KNN. Enter soil and climate data to find the best crop.">
<meta name="keywords" content="crop recommendation, KNN algorithm, soil analysis, AI farming, precision agriculture, best crop prediction">
</head>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #f8fdf8; }
.hero {
    background: linear-gradient(135deg, #1a5c20 0%, #2e7d32 60%, #388e3c 100%);
    padding: 70px 40px; border-radius: 24px; text-align: center;
    margin-bottom: 36px; color: white;
    box-shadow: 0 10px 40px rgba(46,125,50,0.3);
}
.hero h1 { font-size: 3.2rem; font-weight: 800; margin: 0; }
.hero p  { font-size: 1.15rem; opacity: 0.88; margin-top: 12px; }
.badge {
    display: inline-block; background: rgba(255,255,255,0.2);
    padding: 6px 18px; border-radius: 50px; font-size: 0.85rem;
    margin-top: 16px; letter-spacing: 1px;
}
.card {
    background: white; border-radius: 18px; padding: 28px 32px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07); margin-bottom: 20px;
}
.section-title {
    font-size: 0.95rem; font-weight: 700; color: #2e7d32;
    text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 18px;
}
.metric-card {
    background: linear-gradient(135deg, #f1f8e9, #e8f5e9);
    border-radius: 14px; padding: 20px; text-align: center;
    border: 1px solid #c8e6c9;
}
.metric-val { font-size: 2rem; font-weight: 800; color: #1b5e20; }
.metric-lbl { font-size: 0.8rem; color: #666; margin-top: 4px; font-weight: 500; }
.result-box {
    background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
    border-left: 6px solid #2e7d32; border-radius: 16px;
    padding: 36px; text-align: center; margin-top: 24px;
    box-shadow: 0 4px 20px rgba(46,125,50,0.15);
}
.result-crop { font-size: 2.8rem; font-weight: 800; color: #1b5e20; }
.result-sub  { font-size: 1rem; color: #555; margin-top: 8px; }
.confidence  { font-size: 1.1rem; color: #2e7d32; font-weight: 600; margin-top: 12px; }
.stButton>button {
    background: linear-gradient(135deg, #2e7d32, #43a047);
    color: white; font-size: 1.1rem; font-weight: 700;
    padding: 16px 40px; border-radius: 50px; border: none;
    width: 100%; letter-spacing: 0.5px;
    box-shadow: 0 4px 15px rgba(46,125,50,0.35);
}
.how-step {
    background: #f9fbe7; border-radius: 12px; padding: 16px 20px;
    margin-bottom: 10px; border-left: 4px solid #8bc34a; font-size: 0.95rem;
}
footer {
    text-align: center; color: #aaa; font-size: 0.82rem;
    margin-top: 50px; padding: 24px; border-top: 1px solid #e0e0e0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training KNN model...")
def get_model():
    base = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base, "models", "knn_crop.pkl")
    os.makedirs(os.path.join(base, "models"), exist_ok=True)
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    config_path = os.path.join(base, "CONFIG.json")
    with open(config_path) as f:
        config = json.load(f)
    data = pd.read_csv(config["raw_data_path"])
    x, y = data.drop("label", axis=1), data["label"]
    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.10, random_state=0)
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)
    knn.fit(x_train, y_train)
    with open(model_path, "wb") as f:
        pickle.dump(knn, f)
    return knn


model = get_model()

st.markdown("""
<div class="hero">
    <h1>🌱 CropSense</h1>
    <p>AI-Powered Crop Recommendation System</p>
    <p style="opacity:0.75; font-size:0.95rem;">
        Enter your soil nutrients and climate conditions to discover the best crop for your land
    </p>
    <span class="badge">K-Nearest Neighbors Algorithm</span>
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in zip(
    [c1, c2, c3, c4],
    ["22+", "KNN", "7", "Free"],
    ["Crop Types", "Algorithm", "Input Features", "To Use"]
):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-val">{val}</div>
            <div class="metric-lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧪 Soil Nutrients (NPK Ratio)</div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    N = st.slider("Nitrogen (N)", 0, 140, 90)
with col2:
    P = st.slider("Phosphorus (P)", 5, 145, 42)
with col3:
    K = st.slider("Potassium (K)", 5, 205, 43)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🌤 Climate Conditions</div>', unsafe_allow_html=True)
col4, col5, col6, col7 = st.columns(4)
with col4:
    temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 20.87, step=0.1)
with col5:
    humidity = st.number_input("Humidity (%)", 0.0, 100.0, 82.00, step=0.1)
with col6:
    ph = st.number_input("Soil pH", 0.0, 14.0, 6.50, step=0.1)
with col7:
    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 202.93, step=0.1)
st.markdown("</div>", unsafe_allow_html=True)

_, mid, _ = st.columns([1, 2, 1])
with mid:
    predict = st.button("🌾 Get Crop Recommendation")

if predict:
    inp = np.array([N, P, K, temperature, humidity, ph, rainfall]).reshape(1, -1)
    crop = model.predict(inp)[0]
    confidence = round(max(model.predict_proba(inp)[0]) * 100, 1)
    st.markdown(f"""
    <div class="result-box">
        <div class="result-crop">🌾 {crop.upper()}</div>
        <div class="result-sub">Best crop for your soil and climate conditions</div>
        <div class="confidence">Model Confidence: {confidence}%</div>
        <div style="margin-top:10px; font-size:0.85rem; color:#777;">
            Predicted using K-Nearest Neighbors (k=5, Euclidean distance)
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.balloons()

st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">⚙️ How It Works</div>', unsafe_allow_html=True)
for step in [
    "1. Enter your soil nutrient values — Nitrogen, Phosphorus, Potassium",
    "2. Provide your local climate data — temperature, humidity, pH, rainfall",
    "3. The KNN algorithm finds the 5 most similar soil-climate profiles in the dataset",
    "4. The most common crop among those neighbors is recommended to you"
]:
    st.markdown(f'<div class="how-step">{step}</div>', unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<footer>
    <strong>CropSense</strong> &mdash; AI Crop Recommendation System &bull; KNN Algorithm &bull; Powered by Streamlit
</footer>
""", unsafe_allow_html=True)
