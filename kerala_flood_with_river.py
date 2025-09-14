import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from shapely.geometry import Point 

# 1. Load flood data and convert to GeoDataFrame
df = pd.read_csv(r"C:\ML Projects\AI Project\AI-Powered-Flood-Risk-Assistant\data\preprocessed\kerala_flood_with_district.csv")

# Create GeoDataFrame with geometry from longitude/latitude (WGS84)
gdf_flood = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"  # WGS84
)

# 2. Load rivers and filter by Kerala bounding box
gdf_rivers = gpd.read_file(r"C:\ML Projects\AI Project\AI-Powered-Flood-Risk-Assistant\data\raw\hydro\HydroRIVERS_v10_as.shp")

# Get bounding box of flood points (in WGS84)
kerala_bbox = gdf_flood.total_bounds  # [minx, miny, maxx, maxy]
print(f"📍 Kerala Bounding Box: {kerala_bbox}")

# Spatially filter rivers to bounding box (speeds up processing)
gdf_rivers = gdf_rivers.cx[kerala_bbox[0]:kerala_bbox[2], kerala_bbox[1]:kerala_bbox[3]]
print(f"🌊 Rivers in bbox: {len(gdf_rivers)}")

# 3. Project to UTM Zone 43N (EPSG:32643) for accurate distance in meters
gdf_flood = gdf_flood.to_crs("EPSG:32643")
gdf_rivers = gdf_rivers.to_crs("EPSG:32643")

# 4. Convert river geometries to sampled points
river_points = []
for geom in gdf_rivers.geometry:
    if geom.geom_type == 'LineString':
        river_points.extend([Point(x, y) for x, y in geom.coords])
    elif geom.geom_type == 'MultiLineString':
        for line in geom.geoms:
            river_points.extend([Point(x, y) for x, y in line.coords])

# Filter valid points and extract coordinates
river_coords = np.array([[p.x, p.y] for p in river_points if p.is_valid])
print(f"💧 Sampled river points: {len(river_coords)}")

# 5. Extract flood point coordinates (in meters, UTM)
flood_coords = np.array([[geom.x, geom.y] for geom in gdf_flood.geometry])

# 6. Compute nearest river distance using NearestNeighbors
if len(river_coords) == 0:
    raise ValueError("No valid river points found. Check river data or bounding box.")

nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(river_coords)
distances, _ = nbrs.kneighbors(flood_coords)

# Convert meters to kilometers
df['distance_to_river_km'] = distances.flatten() / 1000.0

output_path = r"C:\ML Projects\AI Project\AI-Powered-Flood-Risk-Assistant\data\preprocessed\kerala_flood_with_river.csv"
df.to_csv(output_path, index=False)

print("Added distance to nearest river (in km)!")
print(df[['latitude', 'longitude', 'distance_to_river_km']].head())