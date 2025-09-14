# add_district

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Load flood data
df = pd.read_csv(r"C:\ML Projects\AI Project\AI-Powered-Flood-Risk-Assistant\data\preprocessed\kerala_Flood_enhanced.csv")

# Create GeoDataFrame
gdf_flood = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

# Load GADM districts (Level 2)
gdf_districts = gpd.read_file(r"C:\ML Projects\AI Project\AI-Powered-Flood-Risk-Assistant\data\raw\boundaries\gadm41_IND_2.json")  # or .shp if .json fails

# Filter for Kerala
# Usually: NAME_1 = state, NAME_2 = district
kerala_districts = gdf_districts[gdf_districts['NAME_1'] == 'Kerala'].copy()

kerala_districts = kerala_districts.to_crs(gdf_flood.crs)

# Spatial join: which district is each point in
gdf_joined = gpd.sjoin(gdf_flood, kerala_districts[['NAME_2', 'geometry']], how='left', predicate='within')

# Add district column
df['district'] = gdf_joined['NAME_2']

# Save
df.to_csv(r"C:\ML Projects\AI Project\AI-Powered-Flood-Risk-Assistant\data\preprocessed\kerala_flood_with_district.csv", index=False)
print("Added district info!")
print(df[['latitude', 'longitude', 'district']].head())