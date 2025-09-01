# AI-Powered-Flood-Risk-Assistant

## 📌 Project Overview
This project is part of my **AI & ML learning journey**.  
The AI-Powered Flood Risk Assistant is designed to analyze flood-prone areas by integrating multiple datasets such as precipitation, elevation, river networks, administrative boundaries, and population exposure. The goal is to support early flood risk assessment and decision-making using geospatial and AI-driven approaches.

## 📂 Datasets
The datasets used in this project include:
- 🌧️ Precipitation data (GPM IMERG)  
- 🏔️ Elevation data  
- 🗺️ Administrative boundaries (GADM)  
- 🌊 River network data (HydroRIVERS)  
- 👥 Population distribution data

  (Note: Need to add some more datasets like Historical flood events and soil moisture to make predictions)

📥 All datasets are available on **Kaggle** (beacuse all the datasets which I used are too large):  
👉 [Download Datasets Here](https://www.kaggle.com/datasets/allupranathi/ai-powered-flood-risk-assistant)


## ⚙️ Features
- Load and preprocess geospatial datasets (GeoJSON, Shapefile, GeoPackage, TIFF, TXT).  
- Perform **data exploration** using Pandas & GeoPandas.  
- Check for **missing values** and perform statistical summaries.  
- Foundation for building an **AI-powered risk assistant**.

## 🛠️ Tech Stack
- **Python**  
- **Pandas / GeoPandas**  
- **Matplotlib / Seaborn**  
- **Rasterio** (for raster datasets)  
- **Jupyter Notebook**

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/Pranathi-Allu/AI-Powered-Flood-Risk-Assistant.git
   cd AI-Powered-Flood-Risk-Assistant
