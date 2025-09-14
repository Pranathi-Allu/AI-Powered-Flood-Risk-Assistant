import pandas as pd
import json

# Load flood data
df = pd.read_excel(r"C:\ML Projects\AI Project\AI-Powered-Flood-Risk-Assistant\data\raw\Past Flood data\Kerala_Flood_NoFlood_GSW.xlsx")

# Extract lat/long from .geo (it's JSON string)
def extract_geo(geo_str):
    try:
        geo = json.loads(geo_str)
        return geo['coordinates'][1], geo['coordinates'][0]  # lat, long
    except:
        return None, None

df['latitude'], df['longitude'] = zip(*df['.geo'].apply(extract_geo))

# Drop rows with invalid coordinates
df = df.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)

# Save enhanced dataset
df.to_csv(r"C:\ML Projects\AI Project\AI-Powered-Flood-Risk-Assistant\data\preprocessed\kerala_Flood_enhanced.csv", index=False)
print("Saved enhanced flood data with latitude & longitude!")
print(df[['occurrence', 'label', 'latitude', 'longitude']].head())