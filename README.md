# 🌊 AI-Powered Flood Risk Assistant for Kerala
“Predicting flood risk before it strikes — using data, not guesswork.” 

This project is a complete end-to-end AI solution designed to empower communities in Kerala, India, with early, accurate, and actionable flood risk assessments. Built using geospatial data, machine learning, and an intuitive web interface, this tool transforms raw environmental data into life-saving insights — without requiring technical expertise.

🎯 Project Goal
To build a user-friendly, AI-driven web application that predicts localized flood risk in Kerala based on real-time user inputs (rainfall, river proximity, population density, terrain) — helping residents, local authorities, and NGOs make informed decisions during monsoon season.

✅ No GPS needed
✅ Works offline after deployment
✅ Delivers personalized safety advice via AI chatbot 



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
   - Trained a Random Forest Classifier on 14 districts of Kerala.
   - Achieved 87% balanced accuracy using 5-fold cross-validation.

3. Interactive Web App (Streamlit)

4. FloodBot AI Assistant – Your Emergency Guide
    - Powered by Google Gemini API, with custom system prompt.

5. Interactive Flood Risk Heatmap
    - Visualizes historical flood zones across Kerala (based on 2018 events).
    - Color-coded legend: Red = High Risk, Green = Low Risk.
    - Highlights selected district when prediction is made.

## 🛠️ Tech Stack
- **Python**  
- **Pandas / GeoPandas**  
- **Matplotlib / Seaborn**  
- **Rasterio** (for raster datasets)  
- **Jupyter Notebook**
- **Folium + Leaflet.js** (for mapping)
- **Google Gemini API (Gemini 1.5 Flash)** (AI assistant)

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/Pranathi-Allu/AI-Powered-Flood-Risk-Assistant.git
   cd AI-Powered-Flood-Risk-Assistant

2. Place your trained model files (flood_model_v2.pkl, scaler_v2.pkl, feature_names.pkl) in /models/
3. Run the Streamlit app:
     ```bash
     streamlit run streamlit_app/app.py
     
