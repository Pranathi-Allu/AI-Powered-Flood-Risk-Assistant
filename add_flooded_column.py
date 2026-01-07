import pandas as pd

# Path to original CSV
input_csv = r"C:\ML Projects\AI Project\AI-Powered-Flood-Risk-Assistant\data\preprocessed\kerala_flood_final_withRainfall.csv"
output_csv = r"C:\ML Projects\AI Project\AI-Powered-Flood-Risk-Assistant\data\preprocessed\kerala_flood_final_withRainfall_WITH_LABELS.csv"

# Load the data
df = pd.read_csv(input_csv)

# Define flood labels by district (only for known districts)
flooded_2018_map = {
    "Alappuzha": 1,
    "Kottayam": 1,
    "Pathanamthitta": 1,
    "Ernakulam": 1,
    "Thrissur": 1,
    "Malappuram": 1,
    "Palakkad": 1,
    "Kozhikode": 1,
    "Kollam": 1,
    "Kannur": 0,
    "Kasaragod": 0,
    "Wayanad": 0,
    "Idukki": 0,
    "Thiruvananthapuram": 0
}

# Map district to flooded_2018 — if district is NaN or unknown, set to 0 (safe default)
df['flooded_2018'] = df['district'].map(flooded_2018_map).fillna(0)

nan_count = df['flooded_2018'].isna().sum()
print(f"Filled {nan_count} NaN values in 'flooded_2018' with 0")

# Check for unmapped districts (if any remain)
unmapped = df[df['district'].isin(flooded_2018_map.keys()) == False]['district'].dropna().unique()
if len(unmapped) > 0:
    print("Warning: Unmapped districts found:", unmapped)
else:
    print("All district names matched successfully!")

# Save updated file
df.to_csv(output_csv, index=False)

print(f"Saved updated file to: {output_csv}")
