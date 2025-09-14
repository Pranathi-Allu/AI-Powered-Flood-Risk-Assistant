import streamlit as st
import joblib
import numpy as np
import pandas as pd
import google.generativeai as genai
import os
import folium
from streamlit_folium import st_folium
import hashlib

# PAGE CONFIG — MUST BE FIRST
st.set_page_config(
    page_title="🌊 Kerala Flood Risk Predictor Pro",
    page_icon="🌊",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "map_object" not in st.session_state:
    st.session_state.map_object = None
if "last_district" not in st.session_state:
    st.session_state.last_district = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "model",
            "parts": "Hello! I’m FloodBot 🌊 — your AI assistant for Kerala flood risk. Ask me anything!"
        }
    ]

# CONFIGURE GEMINI

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCSfYxifd7yVmTp9Vg3-HpFyypuiGq5XQc")

try:
    genai.configure(api_key=GEMINI_API_KEY)
    model_gemini = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    st.error(f"⚠️ Gemini setup failed: {e}")
    st.stop()

# 🌍 HIDDEN DISTRICT DATA — FOR GEMINI ONLY (NOT SHOWN TO USER)
DISTRICT_COORDS = {
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Kollam": (8.8932, 76.6141),
    "Pathanamthitta": (9.2660, 76.7855),
    "Alappuzha": (9.4981, 76.3388),
    "Kottayam": (9.5916, 76.5225),
    "Idukki": (9.6805, 76.8184),
    "Ernakulam": (10.0250, 76.3388),
    "Thrissur": (10.5276, 76.2144),
    "Palakkad": (10.7868, 76.6544),
    "Malappuram": (11.0757, 76.0743),
    "Kozhikode": (11.2532, 75.7754),
    "Wayanad": (11.6535, 75.9504),
    "Kannur": (11.8745, 75.3704),
    "Kasaragod": (12.5167, 74.9700)
}

DISTRICT_INFO = {
    "Alappuzha": "High flood risk — coastal, backwaters, low elevation.",
    "Ernakulam": "Moderate-High — urban flooding, Periyar river influence.",
    "Idukki": "Moderate — hilly, landslides and dam releases increase risk.",
    "Kottayam": "High flood risk — near Vembanad Lake, heavy monsoon rainfall, low-lying areas.",
    "Kozhikode": "Moderate — coastal, but well-drained in most areas.",
    "Thrissur": "High — river networks, urban density, historical floods.",
    "Thiruvananthapuram": "Moderate — southern tip, less prone than central districts.",
    "Palakkad": "Moderate — gap in Western Ghats brings heavy rain.",
    "Malappuram": "Moderate-High — river valleys, dense population.",
    "Pathanamthitta": "High — Pamba river, forested hills, flash floods.",
    "Kasaragod": "Low-Moderate — northernmost, less rainfall impact.",
    "Wayanad": "Moderate — hilly, landslides more common than floods.",
    "Kollam": "High — coastal, Ashtamudi Lake, low-lying.",
    "Kannur": "Moderate — northern, less rainfall impact."
}

# DISTRICT DATA WITH PHYSICAL PROPERTIES (used for auto-filling)
DISTRICT_DATA = {
    "Thiruvananthapuram": {
        "coastal": True,
        "hilly": False,
        "riverine": True,
        "avg_rainfall": 2800,
        "avg_river_dist": 8,
        "avg_population": 400
    },
    "Kollam": {
        "coastal": True,
        "hilly": False,
        "riverine": True,
        "avg_rainfall": 3800,
        "avg_river_dist": 3,
        "avg_population": 600
    },
    "Pathanamthitta": {
        "coastal": False,
        "hilly": True,
        "riverine": True,
        "avg_rainfall": 4100,
        "avg_river_dist": 2,
        "avg_population": 600
    },
    "Alappuzha": {
        "coastal": True,
        "hilly": False,
        "riverine": True,
        "avg_rainfall": 4200,
        "avg_river_dist": 1,
        "avg_population": 1200
    },
    "Kottayam": {
        "coastal": False,
        "hilly": False,
        "riverine": True,
        "avg_rainfall": 3900,
        "avg_river_dist": 0.5,
        "avg_population": 900
    },
    "Idukki": {
        "coastal": False,
        "hilly": True,
        "riverine": True,
        "avg_rainfall": 3200,
        "avg_river_dist": 15,
        "avg_population": 200
    },
    "Ernakulam": {
        "coastal": False,
        "hilly": False,
        "riverine": True,
        "avg_rainfall": 3500,
        "avg_river_dist": 2,
        "avg_population": 1500
    },
    "Thrissur": {
        "coastal": False,
        "hilly": False,
        "riverine": True,
        "avg_rainfall": 3700,
        "avg_river_dist": 1,
        "avg_population": 800
    },
    "Palakkad": {
        "coastal": False,
        "hilly": True,
        "riverine": True,
        "avg_rainfall": 3400,
        "avg_river_dist": 20,
        "avg_population": 400
    },
    "Malappuram": {
        "coastal": False,
        "hilly": True,
        "riverine": True,
        "avg_rainfall": 3600,
        "avg_river_dist": 5,
        "avg_population": 800
    },
    "Kozhikode": {
        "coastal": True,
        "hilly": False,
        "riverine": True,
        "avg_rainfall": 3300,
        "avg_river_dist": 10,
        "avg_population": 500
    },
    "Wayanad": {
        "coastal": False,
        "hilly": True,
        "riverine": False,
        "avg_rainfall": 3100,
        "avg_river_dist": 10,
        "avg_population": 300
    },
    "Kannur": {
        "coastal": True,
        "hilly": False,
        "riverine": False,
        "avg_rainfall": 2900,
        "avg_river_dist": 25,
        "avg_population": 300
    },
    "Kasaragod": {
        "coastal": True,
        "hilly": False,
        "riverine": False,
        "avg_rainfall": 900,
        "avg_river_dist": 50,
        "avg_population": 150
    }
}

# POWERFUL SYSTEM CONTEXT — FLOODBOT IS NOW A LIFE-SAVING ASSISTANT
SYSTEM_CONTEXT = (
    "You are 'FloodBot', an AI assistant in the Kerala Flood Risk Predictor app. "
    "You have access to real-time user input: the selected district, rainfall, river distance, population density, and whether it's coastal/hilly/riverine. "
    "You also know the model's prediction: HIGH or LOW risk, with confidence level.\n\n"

    "Here is your internal knowledge of all 14 Kerala districts:\n"
)

for district, info in DISTRICT_DATA.items():
    risk_desc = DISTRICT_INFO[district]
    SYSTEM_CONTEXT += f"- {district}: {risk_desc} (Avg rainfall: {info['avg_rainfall']} mm, Distance to river: {info['avg_river_dist']} km, Population: ~{info['avg_population']}/km²)\n"

SYSTEM_CONTEXT += (
    "\nRULES FOR RESPONSES:\n"
    "- ALWAYS start by acknowledging the user’s concern. Say: 'I understand you're worried about flooding.'\n"
    "- NEVER say 'I don't know' or 'I can't help'. You always have enough information.\n"
    "- If the user asks about safety, ALWAYS use the current prediction and district info to give tailored advice.\n"
    "- If prediction is HIGH RISK: \n"
    "   • Say: 'Based on your location and conditions, there is HIGH flood risk. Take action NOW.'\n"
    "   • Suggest moving to higher ground: 'Go to the nearest school, church, community center, or panchayat office — they are usually built on elevated land.'\n"
    "   • Warn: 'DO NOT walk or drive through floodwater — even 6 inches can sweep away a car or person.'\n"
    "   • Recommend: 'If you have elderly people, children, or pets, leave immediately. Do not wait for official orders.'\n"
    "   • Suggest local landmarks: \n"
    "       - In Alappuzha: 'Head to St. Mary’s Church, District Hospital, or Alappuzha Railway Station.'\n"
    "       - In Kottayam: 'Go to the Catholic College campus, Kottayam Municipal Office, or the Government Medical College.'\n"
    "       - In Pathanamthitta: 'Move to the Pathanamthitta Town Hall or nearby schools on higher ground.'\n"
    "       - In Ernakulam: 'Seek shelter at the Cochin University campus or any government building on raised ground.'\n"
    "       - In Thrissur: 'Head to the Thrissur Pooram Ground or the District Collectorate.'\n"
    "       - In Wayanad: 'Go to the Government Hospital or any school on the hilltop.'\n"
    "       - In Kasaragod: 'Move to the Panchayat office, bus stand, or nearby temple on raised platforms.'\n"
    "- If prediction is LOW RISK: \n"
    "   • Say: 'Your area has LOW flood risk based on current conditions.'\n"
    "   • Still warn: 'Stay alert — monsoon rains can change quickly. Keep your emergency kit ready.'\n"
    "   • Advise: 'Keep drains clear. Avoid going near rivers, backwaters, or low-lying areas.'\n"
    "   • Reassure: 'If water starts rising, move to higher ground — even if it’s just the first floor of a sturdy building.'\n"
    "- If asked 'Where is the safest place near me?':\n"
    "   • Use district-specific knowledge above — never guess.\n"
    "- If asked 'Should I evacuate?':\n"
    "   • If HIGH risk: 'YES — evacuate now. Don’t wait for official orders. Lives are more important than property.'\n"
    "   • If LOW risk: 'Not yet — but prepare your bag and listen to local announcements.'\n"
    "- If asked 'Can I cross the road?':\n"
    "   • Say: 'Never cross flooded roads. Water may be deeper than it looks — and currents can be deadly. Wait until it recedes or get help.'\n"
    "- NEVER mention coordinates, latitude, longitude, or model details like '92% confidence'.\n"
    "- Speak calmly, clearly, and urgently — like a trained emergency officer.\n"
    "- Use simple words. Assume the user may not speak English well.\n"
    "- End every response with: 'Stay safe. Help is nearby.'\n"
    "- Use emojis sparingly — only one at the end if appropriate. 🌊"
)

# LOAD ML MODEL & TOOLS
MODEL_PATH = 'C:/ML Projects/AI Project/AI-Powered-Flood-Risk-Assistant/models/flood_model_v2.pkl'
SCALER_PATH = 'C:/ML Projects/AI Project/AI-Powered-Flood-Risk-Assistant/models/scaler_v2.pkl'
FEATURES_PATH = 'C:/ML Projects/AI Project/AI-Powered-Flood-Risk-Assistant/models/feature_names.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    
    # Validate feature count (6 features)
    expected_features = 6  # distance_to_river_km, avg_annual_rainfall_mm, population_density_log, is_coastal, is_hilly, is_riverine
    if len(feature_names) != expected_features:
        st.error(f"❌ Model expects {len(feature_names)} features, but app expects {expected_features}. Please retrain model with correct features.")
        st.stop()
        
    st.sidebar.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# MAIN PAGE TITLE
st.title("🌊 Kerala Flood Risk Predictor")
st.subheader("AI-powered flood risk assessment for Kerala districts")

# Display prediction result permanently
if st.session_state.prediction_result:
    result = st.session_state.prediction_result
    if result["type"] == "error":
        st.error(result["message"])
    elif result["type"] == "success":
        st.success(result["message"])
    if result["advice"]:
        st.info(result["advice"])

# Difference between prediction and map
st.markdown("""
### 📌 Note:
- 🗺️ The **heatmap** shows **historical flood risk** based on past events.  
- 🤖 The **prediction** shows **current risk** based on your inputs.  
- ⚠️ A low prediction does *not* mean zero risk — only that conditions are currently favorable.
""")

st.markdown("""
Enter location details below to predict flood risk — powered by geography, rainfall, rivers, and population.
""")

# Create a dedicated placeholder for the heatmap
map_placeholder = st.empty()

# SIDEBAR INPUTS — USER-FRIENDLY (AUTO-FILLED FROM DISTRICT)
st.sidebar.header("📍 Enter Location Details")

# District selection — source of truth for auto-filling
districts = sorted(DISTRICT_INFO.keys())
district_name = st.sidebar.selectbox(
    "Select District", 
    options=districts,
    index=0,
    help="Choose the district for better context — inputs will auto-fill based on historical patterns."
)

# Auto-fill values from DISTRICT_DATA
selected_data = DISTRICT_DATA[district_name]

rainfall = st.sidebar.number_input(
    "Avg Annual Rainfall (mm)", 
    min_value=0, 
    max_value=5000, 
    value=int(selected_data["avg_rainfall"]),
    help="Average yearly rainfall in millimeters"
)

river_dist = st.sidebar.number_input(
    "Distance to Nearest River (km)", 
    min_value=0.0, 
    max_value=100.0, 
    value=float(selected_data["avg_river_dist"]),
    step=0.1,
    help="How far is the location from a major river?"
)

population = st.sidebar.number_input(
    "Population Density (people/km²)", 
    min_value=0, 
    max_value=10000, 
    value=int(selected_data["avg_population"]),
    help="Average number of people per square kilometer"
)

# These are fixed based on district — checkboxes for transparency
is_coastal = selected_data["coastal"]
is_hilly = selected_data["hilly"]
is_riverine = selected_data["riverine"]

# Show summary of auto-filled values
st.sidebar.markdown(f"### 📍 Summary for {district_name}")
st.sidebar.write(f"- **Rainfall**: {rainfall} mm/year")
st.sidebar.write(f"- **River distance**: {river_dist:.1f} km")
st.sidebar.write(f"- **Population density**: {population:,}/km²")
st.sidebar.write(f"- **Coastal**: {'✅ Yes' if is_coastal else '❌ No'}")
st.sidebar.write(f"- **Hilly**: {'✅ Yes' if is_hilly else '❌ No'}")
st.sidebar.write(f"- **Riverine**: {'✅ Yes' if is_riverine else '❌ No'}")

# Optional: Allow manual override of flags (for advanced users)
st.sidebar.markdown("#### Adjust Risk Factors (Optional)")
is_coastal = st.sidebar.checkbox("Is Coastal?", value=is_coastal, help="Check if area is near sea (e.g., Alappuzha, Kasaragod)")
is_hilly = st.sidebar.checkbox("Is Hilly?", value=is_hilly, help="Check if area is mountainous (e.g., Idukki, Wayanad)")
is_riverine = st.sidebar.checkbox("Is Riverine?", value=is_riverine, help="Check if area is near major river network (e.g., Alappuzha, Kottayam)")

# 🔮 PREDICT BUTTON + RESULT DISPLAY
if st.sidebar.button("🔮 Predict Flood Risk"):
    log_pop = np.log1p(population)

    input_data = np.array([[
        river_dist,
        rainfall,
        log_pop,
        int(is_coastal),
        int(is_hilly),
        int(is_riverine)
    ]])

    st.write("### 🧪 DEBUG: Input Data Before Scaling")
    st.write(f"Feature names expected: {feature_names}")
    for i, name in enumerate(feature_names):
        st.write(f"  {name}: {input_data[0][i]:.3f}")

    try:
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]

        # ALWAYS: prob[0] = P(High Risk), prob[1] = P(Low Risk)
        if pred == 1:
            # Model says LOW risk → show low risk message with prob[1]
            st.session_state.prediction_result = {
                "type": "success",
                "message": f"✅ LOW FLOOD RISK — {prob[1]:.1%} confidence",
                "advice": "🟢 Area is relatively safe — but stay alert during heavy monsoon!"
            }
        else:
            # Model says HIGH risk → show high risk message with prob[0]
            st.session_state.prediction_result = {
                "type": "error",
                "message": f"🚨 HIGH FLOOD RISK — {prob[0]:.1%} confidence",
                "advice": "⚠️ Consider evacuation plans, sandbagging, and monitoring water levels."
            }

        st.session_state.prediction_done = True

    except Exception as e:
        st.session_state.prediction_result = {
            "type": "error",
            "message": f"❌ Prediction failed: {e}",
            "advice": ""
        }
        st.session_state.prediction_done = False

# 🗺️ ADD HEATMAP: KERALA FLOOD RISK MAP (DYNAMIC & FLICKER-FREE) + LEGEND
def create_kerala_map(selected_district=None, prediction_done=False):
    kerala_center = [10.5, 76.0]
    m = folium.Map(location=kerala_center, zoom_start=8, tiles="OpenStreetMap")

    colors = {
        "High": "#e74c3c",
        "Moderate-High": "#f39c12",
        "Moderate": "#f1c40f",
        "Low-Moderate": "#2ecc71",
        "Low": "#27ae60"
    }

    # Add base markers for all districts
    for district, (lat, lon) in DISTRICT_COORDS.items():
        risk_desc = DISTRICT_INFO[district]
        risk_level = None

        if "High" in risk_desc:
            risk_level = "High"
        elif "Moderate-High" in risk_desc:
            risk_level = "Moderate-High"
        elif "Moderate" in risk_desc:
            risk_level = "Moderate"
        elif "Low-Moderate" in risk_desc:
            risk_level = "Low-Moderate"
        else:
            risk_level = "Low"

        color = colors.get(risk_level, "#95a5a6")

        folium.CircleMarker(
            location=[lat, lon],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"<b>{district}</b><br>Flood Risk: {risk_level}<br>{risk_desc}",
            tooltip=district
        ).add_to(m)

    # Highlight selected district if prediction was made
    if prediction_done and selected_district:
        try:
            lat, lon = DISTRICT_COORDS[selected_district]
            folium.CircleMarker(
                location=[lat, lon],
                radius=12,
                color="#ffffff",
                fill=True,
                fill_color="#3498db",
                fill_opacity=0.8,
                popup=f"<b>{selected_district}</b><br>Selected Location",
                tooltip="Your Input"
            ).add_to(m)
        except Exception as e:
            st.warning(f"⚠️ Failed to highlight {selected_district}: {e}")

    return m

# Render map
st.markdown("---")
st.markdown("<h3 style='color:#FF6B6B;'>📊 Kerala Flood Risk Heatmap</h3>", unsafe_allow_html=True)
st.caption("Interactive map showing flood risk levels by district (based on historical patterns)")

try:
    if st.session_state.map_object is None:
        st.session_state.map_object = create_kerala_map()

    if st.session_state.last_district != district_name or st.session_state.prediction_done != st.session_state.get('prev_prediction_done', False):
        st.session_state.map_object = create_kerala_map(
            selected_district=district_name,
            prediction_done=st.session_state.prediction_done
        )
        st.session_state.last_district = district_name
        st.session_state.prev_prediction_done = st.session_state.prediction_done

    # Render the map
    st_folium(st.session_state.map_object, width=700, height=400)

    # HIGH-CONTRAST, THEME-SAFE LEGEND
    st.markdown("""
    <div style="
        display: flex; 
        justify-content: center; 
        gap: 12px; 
        margin: 15px auto; 
        padding: 10px 15px; 
        background-color: rgba(255, 255, 255, 0.9); 
        border-radius: 12px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        font-family: 'Segoe UI', sans-serif;
        font-size: 14px;
        color: #222;
        max-width: 90%;
        flex-wrap: wrap;
    ">
        <span style="display: flex; align-items: center;">
            <span style="display: inline-block; width: 14px; height: 14px; background-color: #e74c3c; border-radius: 50%; margin-right: 6px;"></span>
            <strong>High</strong>
        </span>
        <span style="display: flex; align-items: center;">
            <span style="display: inline-block; width: 14px; height: 14px; background-color: #f39c12; border-radius: 50%; margin-right: 6px;"></span>
            <strong>Moderate-High</strong>
        </span>
        <span style="display: flex; align-items: center;">
            <span style="display: inline-block; width: 14px; height: 14px; background-color: #f1c40f; border-radius: 50%; margin-right: 6px;"></span>
            <strong>Moderate</strong>
        </span>
        <span style="display: flex; align-items: center;">
            <span style="display: inline-block; width: 14px; height: 14px; background-color: #2ecc71; border-radius: 50%; margin-right: 6px;"></span>
            <strong>Low-Moderate</strong>
        </span>
        <span style="display: flex; align-items: center;">
            <span style="display: inline-block; width: 14px; height: 14px; background-color: #27ae60; border-radius: 50%; margin-right: 6px;"></span>
            <strong>Low</strong>
        </span>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"❌ Map failed to load: {e}")
    st.info("Try refreshing the page or check your internet connection.")

# GEMINI CHATBOT — SMART, HIDDEN CONTEXT
st.markdown("---")
st.markdown("<h3 style='color:#007BFF;'>💬 Ask FloodBot AI Assistant</h3>", unsafe_allow_html=True)
st.caption("Ask about flood risk, safety tips, districts, or general questions — I’m here to help!")

for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["parts"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["parts"])

#DYNAMIC CONTEXT INJECTION FOR FLOODBOT
if user_input := st.chat_input("Type your question here..."):
    st.session_state.chat_history.append({
        "role": "user",
        "parts": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🧠 FloodBot is thinking..."):
            try:
                # Get current context for dynamic personalization
                current_district = district_name
                current_risk = "HIGH" if st.session_state.prediction_result and st.session_state.prediction_result["type"] == "error" else "LOW"
                current_confidence = ""
                if st.session_state.prediction_result:
                    if st.session_state.prediction_result["type"] == "error":
                        current_confidence = f" ({st.session_state.prediction_result['message'].split('—')[1].strip()})"
                    else:
                        current_confidence = f" ({st.session_state.prediction_result['message'].split('—')[1].strip()})"

                # Build dynamic system context with live context
                contextual_system = SYSTEM_CONTEXT + (
                    f"\n\nUSER CONTEXT: Current district: {current_district}. "
                    f"Current prediction: {current_risk} risk{current_confidence}."
                )

                chat = model_gemini.start_chat(history=[
                    {"role": "user", "parts": contextual_system},
                    {"role": "model", "parts": "Understood. I am FloodBot and will assist without revealing internal data."},
                ] + st.session_state.chat_history)

                response = chat.send_message(user_input)
                st.markdown(response.text)
                st.session_state.chat_history.append({
                    "role": "model",
                    "parts": response.text
                })

            except Exception as e:
                st.error(f"⚠️ Error: {e}")
                st.info("Try rephrasing or check your API key.")

# 🆘 EMERGENCY MODE BUTTON — ONE-CLICK LIFE-SAVING GUIDE
st.markdown("---")
if st.button("🆘 EMERGENCY MODE — Give me immediate instructions"):
    st.session_state.chat_history.append({
        "role": "user",
        "parts": "I am in danger. My area is flooding. Tell me exactly what to do right now."
    })
    with st.chat_message("user"):
        st.markdown("I am in danger. My area is flooding. Tell me exactly what to do right now.")
    
    with st.chat_message("assistant"):
        with st.spinner("🧠 FloodBot is assessing your situation..."):
            try:
                current_district = district_name
                current_risk = "HIGH" if st.session_state.prediction_result and st.session_state.prediction_result["type"] == "error" else "LOW"
                contextual_system = SYSTEM_CONTEXT + (
                    f"\n\nUSER CONTEXT: Current district: {current_district}. "
                    f"Current prediction: {current_risk} risk."
                )
                chat = model_gemini.start_chat(history=[
                    {"role": "user", "parts": contextual_system},
                    {"role": "model", "parts": "Understood. I am FloodBot and will assist without revealing internal data."},
                ] + st.session_state.chat_history)
                response = chat.send_message("I am in danger. My area is flooding. Tell me exactly what to do right now.")
                st.markdown(response.text)
                st.session_state.chat_history.append({
                    "role": "model",
                    "parts": response.text
                })
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

# FOOTER
st.markdown("---")
st.caption("💡 Powered by ML model trained on Kerala historical flood data | Features: rainfall, rivers, population, terrain")
st.caption("© 2025 Kerala Flood Risk Predictor Pro | For educational and emergency preparedness use")

# Clear prediction when inputs change
def get_input_hash():
    return hashlib.md5(
        f"{district_name}{rainfall}{river_dist}{population}{is_coastal}{is_hilly}{is_riverine}".encode()
    ).hexdigest()

current_hash = get_input_hash()
if "last_input_hash" not in st.session_state:
    st.session_state.last_input_hash = current_hash

if st.session_state.last_input_hash != current_hash:
    st.session_state.prediction_result = None
    st.session_state.prediction_done = False
    st.session_state.last_input_hash = current_hash
