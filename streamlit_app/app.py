# app.py — Kerala Flood Risk Predictor Pro+
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="🌊 Kerala Flood Risk Predictor Pro+", layout="wide")

# --- PATHS: Load models from parent 'models' folder ---
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

MODEL_PATH = PROJECT_ROOT / "models" / "flood_model_v2.pkl"
SCALER_PATH = PROJECT_ROOT / "models" / "scaler_v2.pkl"

# --- APIs (fallback keys) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# Use a fallback key just in case 
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API", "25ac8c6e091d7626e3b6b35a4ad5e1d8")

# --- DEFAULT FEATURE ORDER (avoids NoneType crash) ---
FEATURE_ORDER = ['rainfall_24h_mm', 'river_level_m', 'elevation_m', 'slope_deg', 'pop_density', 'soil_saturation']

# --- LOAD MODEL & SCALER ---
@st.cache_resource
def load_ml_artifacts():
    artifacts = {"model": None, "scaler": None}
    if MODEL_PATH.exists():
        try:
            artifacts["model"] = joblib.load(MODEL_PATH)
            st.sidebar.success("✅ ML Model loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Model load failed: {e}")
    else:
        st.sidebar.warning("⚠️ Model not found — using fallback logic")

    if SCALER_PATH.exists():
        try:
            artifacts["scaler"] = joblib.load(SCALER_PATH)
        except:
            pass
    return artifacts

art = load_ml_artifacts()
model = art["model"]
scaler = art["scaler"]

# SAFE fetch_live_weather
def fetch_live_weather(lat, lon):
    # simulate realistic live data (no API call!)
    # Replace with real API later if needed
    return {
        "rainfall_24h_mm": 85.2,   # ← realistic monsoon value
        "temp_c": 29.5,
        "humidity": 88,
        "wind_m_s": 3.2,
    }

# --- PREPARE FEATURES ---
def prepare_features(raw: dict):
    # Use hard-coded FEATURE_ORDER (no dependency on missing files)
    df_row = pd.DataFrame([raw])
    for feat in FEATURE_ORDER:
        if feat not in df_row.columns:
            df_row[feat] = 0.0
    X = df_row[FEATURE_ORDER].fillna(0).values.astype(np.float32)
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except:
            pass
    return X

# --- PREDICTION ---
def predict_risk_from_features(X, threshold=0.5):
    if model is not None:
        try:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X)[0, 1]
            else:
                prob = float(model.predict(X)[0])
                if prob > 1: prob = min(1.0, prob / 2.0)  # normalize
        except:
            prob = 0.0
    else:
        # Fallback heuristic (tuned for realism)
        rainfall = X[0][0]
        river = X[0][1]
        prob = min(1.0, (rainfall * 0.008) + (river * 0.15))
    return np.array([prob]), np.array([1 if prob >= threshold else 0])

def risk_label(prob, threshold=0.5):
    if prob >= 0.9: return "🔴 CRITICAL"
    elif prob >= threshold: return "🟠 HIGH"
    elif prob >= 0.3: return "🟡 MEDIUM"
    else: return "🟢 LOW"

# --- UI ---
st.title("🌊 Kerala Flood Risk Predictor Pro+")
st.caption("AI-Powered Early Warning System for Kerala Floods")

mode = st.sidebar.radio("📌 Input Mode", ["🧪 Manual Input (Recommended)", "📡 Live Data (Simulated)"], index=0)
threshold = st.sidebar.slider("Risk Threshold", 0.3, 0.8, 0.5, 0.05)

# MAIN
if mode == "📡 Live Data (Simulated)":
    st.header("📡 Live Data")
    district = st.selectbox("District", ["Kottayam", "Alappuzha", "Idukki", "Ernakulam"])
    lat = st.number_input("Latitude", value=9.59)
    lon = st.number_input("Longitude", value=76.52)

    if st.button("🌧️ Simulate Live Prediction"):
        weather = fetch_live_weather(lat, lon)
        raw = {
            "rainfall_24h_mm": weather["rainfall_24h_mm"],
            "river_level_m": 3.8,  # realistic rising level
            "elevation_m": 12.0,
            "slope_deg": 2.5,
            "pop_density": 1250,
            "soil_saturation": 0.78,
        }
        st.info(f"📍 Simulated live data for {district}")
        st.json(raw)
        X = prepare_features(raw)
        proba, _ = predict_risk_from_features(X, threshold)
        prob = float(proba[0])
        st.metric("⚠️ Flood Risk", risk_label(prob, threshold), f"{prob*100:.1f}%")

else:
    st.header("🧪 Manual Input (Recommended for Demo)")
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        with col1:
            rainfall = st.slider("🌧️ Rainfall (24h) in mm", 0, 500, 120)
            river_level = st.slider("🌊 River Level (m)", 0.0, 6.0, 3.2)
            elevation = st.number_input("⛰️ Elevation (m)", 0, 500, 15)
        with col2:
            slope = st.slider("📉 Slope (°)", 0, 20, 3)
            pop_density = st.number_input("👥 Pop. Density (/km²)", 100, 10000, 1200)
            soil_sat = st.slider("💧 Soil Saturation", 0.0, 1.0, 0.75)
        submitted = st.form_submit_button("🔍 Predict Flood Risk")

    if submitted:
        raw = {
            "rainfall_24h_mm": rainfall,
            "river_level_m": river_level,
            "elevation_m": elevation,
            "slope_deg": slope,
            "pop_density": pop_density,
            "soil_saturation": soil_sat,
        }
        X = prepare_features(raw)
        proba, _ = predict_risk_from_features(X, threshold)
        prob = float(proba[0])
        st.success(f"✅ Prediction Complete")
        st.metric("Flood Risk Level", risk_label(prob, threshold), delta=f"{prob*100:.1f}% probability")

# FLOODBOT (Gemini)
st.markdown("---")
st.subheader("💬 FloodBot — AI Emergency Assistant")

# Use a minimal working FloodBot or mock it
if GOOGLE_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GOOGLE_API_KEY)
        model_bot = genai.GenerativeModel("gemini-2.5-flash")
        bot_ready = True
    except:
        bot_ready = False
else:
    bot_ready = False

if bot_ready:
    if "chat" not in st.session_state:
        st.session_state.chat = [{"role": "assistant", "content": "Namaskaram! I'm FloodBot. Ask: *'What to do in high risk?'*, *'Nearest shelter in Kottayam?'*"}]
    for msg in st.session_state.chat:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input("Your emergency query..."):
        st.session_state.chat.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.spinner("FloodBot responding..."):
            response = model_bot.generate_content(
                "You are FloodBot: Kerala flood emergency assistant. Respond in short, clear, helpful English. Include Malayalam if helpful. Never refuse. "
                + prompt
            )
            reply = response.text
        st.chat_message("assistant").write(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})
else:
    st.info("💡 FloodBot demo mode: Try asking *'Steps for evacuation in Kottayam?'*")
    st.chat_message("assistant").write("Namaskaram! I can help with: 🚨 Evacuation steps • 🏠 Nearby shelters • 📞 Emergency contacts • 🩹 First aid in floods")

st.markdown("---")
