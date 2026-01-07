import h5py
import requests
import datetime
import numpy as np
import pandas as pd
import os
import socket
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from netrc import netrc

# Kerala bounding box (approx)
lat_min, lat_max = 8.0, 12.8
lon_min, lon_max = 74.8, 77.5

# Time range
start_date = datetime.date(2000, 6, 1)
end_date = datetime.date(2000, 6, 5)  # Example; expand as needed

#Earthdata Credentials
# For better security, create a .netrc file in your home directory
# with the following content:
# machine urs.earthdata.nasa.gov login YOUR_USERNAME password YOUR_PASSWORD
# If you don't use a .netrc file, replace the placeholders below.
USERNAME = ""  # NASA Earthdata username
PASSWORD = ""  # NASA Earthdata password

# Base URL templates for GPM IMERG Final Daily (V07)
url_template = "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/GPM_3IMERGDF.07/{year}/{month}/3B-DAY.MS.MRG.3IMERG.{ymd}-S000000-E235959.V07B.HDF5"

# Output CSV path
output_csv = r"kerala_precipitation_daily.csv"

# Ensure output directory exists
output_dir = os.path.dirname(output_csv)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)
    print(f"Created output directory: {output_dir}")

# FUNCTION TO CHECK NETWORK CONNECTIVITY
def check_network(hostname="gpm1.gesdisc.eosdis.nasa.gov"):
    """Checks for a connection to the data server."""
    try:
        socket.create_connection((hostname, 443), timeout=5)
        print(f"Successfully connected to {hostname}")
        return True
    except (socket.gaierror, socket.timeout) as e:
        print(f"Network error: Cannot connect to {hostname} - {e}")
        return False

# FUNCTION TO DOWNLOAD AND PROCESS HDF5 DATA
def get_precip_from_hdf(url, date_str, session):
    """Downloads an HDF5 file, extracts precipitation data for the bounding box, and returns the mean."""
    temp_file = None
    try:
        # Download HDF5 file with a timeout
        with session.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            # Save to a temporary file
            temp_file = f"temp_{date_str}.h5"
            with open(temp_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Read the HDF5 file
            with h5py.File(temp_file, 'r') as f:
                # The correct dataset path for V07 is 'Grid/precipitation'
                if 'Grid/precipitation' not in f:
                    print(f" Dataset 'Grid/precipitation' not found in {url}")
                    return None
                
                precip_data = f['Grid/precipitation'][:]
                lats = f['Grid/lat'][:]
                lons = f['Grid/lon'][:]

                # Find indices for the Kerala bounding box
                lat_idx = np.where((lats >= lat_min) & (lats <= lat_max))[0]
                lon_idx = np.where((lons >= lon_min) & (lons <= lon_max))[0]

                if len(lat_idx) == 0 or len(lon_idx) == 0:
                    print(f"No data within bounding box for {date_str}")
                    return None

                # Use np.ix_ to correctly select a rectangular subset (grid).
                # The original code `precip_data[0, lon_idx, lat_idx]` was incorrect.
                # The shape of the GPM data is (time, lon, lat).
                kerala_precip = precip_data[0, np.ix_(lon_idx, lat_idx)]
                
                # Calculate the mean, ignoring any potential fill values (NaNs)
                mean_precip = np.nanmean(kerala_precip)
                
                return mean_precip
                
    except requests.exceptions.HTTPError as he:
        # Handle 404 errors for dates where data might not exist
        if he.response.status_code == 404:
            print(f" Data not found for {date_str} (HTTP 404). This is normal for some dates.")
        else:
            print(f"HTTP error for {date_str}: {he}")
        return None
    except requests.exceptions.RequestException as re:
        print(f" Request error for {date_str}: {re}")
        return None
    except (OSError, h5py.Error) as oe:
        print(f" HDF5 or file error for {date_str}: {oe}")
        return None
    except Exception as e:
        print(f" An unexpected error occurred for {date_str}: {e}")
        return None
    finally:
        # Clean up the temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except OSError as e:
                print(f"Error deleting temporary file {temp_file}: {e}")

# Check network first
if not check_network():
    print(" Exiting due to network connectivity issues.")
    exit(1)

# Set up a session with automatic retries for temporary server issues
session = requests.Session()
retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# Authenticate using .netrc file if it exists, otherwise use hardcoded credentials
try:
    info = netrc()
    login, _, password = info.authenticators("urs.earthdata.nasa.gov")
    session.auth = (login, password)
    print(" Successfully using credentials from .netrc file.")
except (FileNotFoundError, TypeError):
    print(" .netrc file not found. Using credentials from script.")
    session.auth = (USERNAME, PASSWORD)


# --- Main Loop ---
records = []
current_date = start_date
while current_date <= end_date:
    ymd = current_date.strftime("%Y%m%d")
    year = current_date.strftime("%Y")
    # GPM data uses day-of-year for directory structure in some cases, but month for this product
    month = current_date.strftime("%m")
    
    # Construct the URL for the current date
    url = url_template.format(year=year, month=month, ymd=ymd)
    
    # Get and process the precipitation data
    precip = get_precip_from_hdf(url, ymd, session)
    
    if precip is not None and not np.isnan(precip):
        print(f"{ymd} : {precip:.2f} mm/day")
        records.append({"date": current_date, "precip_mm": precip})
    else:
        print(f" Failed to retrieve valid data for {ymd}")

    current_date += datetime.timedelta(days=1)

# SAVE RESULTS TO CSV
if records:
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['date'])
    
    try:
        df.to_csv(output_csv, index=False)
        print(f"\nSuccess! Kerala daily precipitation data saved to:\n{os.path.abspath(output_csv)}")
    except Exception as e:
        print(f"Error saving the final CSV file: {e}")
else:

    print("\nNo data was collected. The CSV file was not created.")
