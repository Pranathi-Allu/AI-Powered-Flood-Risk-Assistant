# Adding rainfall data to past flood data
import pandas as pd

# Load full flood data
df = pd.read_csv(r"C:\ML Projects\AI Project\AI-Powered-Flood-Risk-Assistant\data\preprocessed\kerala_flood_full.csv")

# Load rainfall
rain_df = pd.read_csv(r"C:\ML Projects\AI Project\AI-Powered-Flood-Risk-Assistant\data\raw\precipitation\daily-rainfall-at-state-level.csv")

# Filter for Kerala (find state_code or state_name for Kerala)
kerala_rain = rain_df[rain_df['state_name'] == 'Kerala'] 

# Compute average actual rainfall
avg_rainfall = kerala_rain['actual'].mean()

# Add to all rows
df['avg_annual_rainfall_mm'] = avg_rainfall

# Save final dataset
df.to_csv(r"C:\ML Projects\AI Project\AI-Powered-Flood-Risk-Assistant\data\preprocessed\kerala_flood_final_withRainfall.csv", index=False)
print("Added average annual rainfall for Kerala!")
print(df[['latitude', 'longitude', 'avg_annual_rainfall_mm']].head())