# 🌊 AI-Powered Flood Risk Assistant for Kerala
“Predicting flood risk before it strikes — using data, not guesswork.” 

This project is a complete end-to-end AI solution designed to empower communities in Kerala, India, with early, accurate, and actionable flood risk assessments. Built using geospatial data, machine learning, and an intuitive web interface, this tool transforms raw environmental data into life-saving insights — without requiring technical expertise.

🎯 Project Goal
To build a user-friendly, AI-driven web application that predicts localized flood risk across Kerala using real-time or manual input features such as rainfall, river level, terrain slope, and population density.
It helps residents, local authorities, and NGOs take timely and informed action during the monsoon season.


## 📂 Datasets
The datasets used in this project include:
- 🌧️ Annual Rainfall (source: IMD)
- 🏔️ Elevation data  (source: SRTM, NASA)
- 🗺️ Administrative boundaries (GADM)  
- 🌊 River network data (source: HydroRIVERS (NASA))  
- 👥 Population distribution data (source: censes of india)
- 🌐 Kerala past flood dataset (Flood / No Flood samples) 
  (I got this kerala past flood dataset from NASA Earth data. I already included the code for getting the dataset "Kerala_flood_data.ipynb" (refer that file)).

📥 All datasets are available on **Kaggle** (beacuse all the datasets which I used are too large):  
👉 [Download Datasets Here](https://www.kaggle.com/datasets/allupranathi/ai-powered-flood-risk-assistant)

💡 All datasets were preprocessed and merged into a single unified CSV:
kerala_flood_final_withRainfall_WITH_LABELS.csv
(Available in /data/preprocessed/) 

🚫 No latitude/longitude used in modeling — to prevent spatial leakage. Only derived physical features are used. 


## ⚙️ Key Features
1. Smart Data Integration
   - Merged 6 disparate datasets into one clean, labeled dataset.
   - Created domain-specific features:
        - is_coastal, is_hilly, is_riverine (binary flags)
        - distance_to_river_km
        - population_density_log (log-transformed for skew handling)
          
2. Robust Machine Learning Model

   - Trained using multi-source Kerala environmental datasets
   - Achieved ~87% balanced accuracy (5-fold CV)
   - Automatically scales inputs via pre-trained scaler_v2.pkl
   - Works even without model files (fallback heuristic enabled)

3. Interactive Streamlit Web App
   - Two input modes:
      - Manual Input (Recommended for demo)
      - Live Data (Simulated) – generates realistic monsoon conditions
   - Real-time visual feedback using risk-level badges:
     🔴 CRITICAL | 🟠 HIGH | 🟡 MEDIUM | 🟢 LOW
   - No dependency on GPS or external API calls
   - Fully functional offline after deployment

4. FloodBot — AI Emergency Assistant
   - Powered by Google Gemini 2.5 Flash (with .env API key)
   - Provides instant, short, and bilingual (English + Malayalam) emergency help
   - Suggests evacuation steps, nearest shelters, and safety guidance
   - Works in demo mode if Gemini API key is not provided

## 🛠️ Tech Stack
- **Python**  
- **Pandas / GeoPandas**  
- **Matplotlib / Seaborn**  
- **Rasterio** (for raster datasets)  
- **Jupyter Notebook**
- **Google Gemini API (Gemini 2.5 Flash)** (AI assistant)

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/Pranathi-Allu/AI-Powered-Flood-Risk-Assistant.git
   cd AI-Powered-Flood-Risk-Assistant
2. Set up environment
   Create a .env file in the project root:
   ```bash
     GOOGLE_API_KEY=your_gemini_api_key_here
     OPENWEATHER_API=your_openweather_api_key_here
(Optional — app runs even without these in demo mode.)

3. Place your trained model files (flood_model_v2.pkl, scaler_v2.pkl, feature_names.pkl) in /models/
4. Run the Streamlit app:
     ```bash
     streamlit run streamlit_app/app.py


##Future Enhancements
- Integration with live rainfall APIs (IMD / OpenWeather)
- Addition of NDWI (Normalized Difference Water Index) using Sentinel-2 data
- District-wise 3D terrain visualization
- Integration with Twilio alerts for real-time SMS notifications
