import rioxarray as rxr
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point
import rasterio
from pyproj import Transformer

# Load flood data
df = pd.read_csv(r"C:\ML Projects\AI Project\AI-Powered-Flood-Risk-Assistant\data\preprocessed\kerala_flood_with_river.csv")

# METHOD: rasterio directly for sampling

raster_path = r"C:\ML Projects\AI Project\AI-Powered-Flood-Risk-Assistant\data\raw\ind_ppp_2020_constrained.tif"

with rasterio.open(raster_path) as src:
    # Create transformer: from EPSG:4326 (lon/lat) to raster CRS
    transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)

    # Transform flood points to raster CRS
    xs, ys = transformer.transform(df.longitude.values, df.latitude.values)

    # Sample raster at each point
    sampled = np.array(list(src.sample(zip(xs, ys))))
    # Flatten and handle multi-band (if any)
    if sampled.ndim == 2 and sampled.shape[1] == 1:
        sampled = sampled.flatten()
    elif sampled.ndim == 1:
        pass  # already flat
    else:
        sampled = sampled[:, 0]  # take first band

    df['population_density'] = sampled

# Handle NaN or NoData
nodata = src.nodata
if nodata is not None:
    df['population_density'] = df['population_density'].replace(nodata, np.nan)

df['population_density'] = df['population_density'].fillna(0)

print("Raster CRS:", src.crs)
print("Sampled first 5 values:", df['population_density'].head().values)
print("Number of NaNs (before fill):", df['population_density'].isna().sum())

# Save
output_path = r"C:\ML Projects\AI Project\AI-Powered-Flood-Risk-Assistant\data\preprocessed\kerala_flood_full.csv"
df.to_csv(output_path, index=False)

print("Added population density!")
print(df[['latitude', 'longitude', 'population_density']].head())