"""
new_data.py

Purpose:
Combine burn severity, fuel treatment, vegetation, topography, and climate datasets into a 
unified testing dataset for northern California.

Inputs:
- MTBS burn severity rasters (contained in fires_mtbs dictionary)
- CAL FIRE fuel treatment geodatabase (CALFIRE_FuelReductionProjects.gdb, CMDash_ProjectTreatments layer)
- LANDFIRE vegetation and topography rasters (contained in landfire dictionary)
- PRISM climate rasters (downloaded through API based on fire ignition month)

Output:
- new_fires_dataset.csv

Notes:
- Sampled every 10th pixel to reduce dataset size for testing, gridding not necessary since this is only for testing purposes
- Used LF2024_EVT.csv lookup table to group LANDFIRE vegetation types into broader categories
- Grouped LANDFIRE fuel models into broader categories based on standard classifications
"""

import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, Point
import requests
import zipfile
import os

#-----------------------
# LOAD DATA
#-----------------------

# dictionary of MTBS fires and their corresponding burn severity raster paths
fires_mtbs = {
    "McCash_2021": "SourceDatasets/mtbs_new_fires/mtbs/McCash_2021/mtbs_ca4156412340420210801_10024164_20210718_20220720_rdnbr.tif",
    "Park_2024": "SourceDatasets/mtbs_new_fires/mtbs/Park_2024/mtbs_ca3981912180320240724_10033184_20230827_20240826_rdnbr.tif"
}

# LANDFIRE raster paths for vegetation and topography variables
landfire = {
    "fuel_model": "SourceDatasets/landfire/LF2020_FBFM40_200_CONUS/Tif/LC20_F40_200.tif",
    "vegetation_type": "SourceDatasets/landfire/LF2016_EVT_200_CONUS/Tif/LC16_EVT_200.tif",
    "canopy_cover": "SourceDatasets/landfire/LF2020_CC_200_CONUS/Tif/LC20_CC_200.tif",
    "slope_deg": "SourceDatasets/landfire/LF2020_SlpD_CONUS/Tif/LF2020_SlpD_CONUS.tif",
    "aspect_deg": "SourceDatasets/landfire/LF2020_Asp_CONUS/Tif/LF2020_Asp_CONUS.tif",
    "elevation_m": "SourceDatasets/landfire/LF2020_Elev_CONUS/Tif/LF2020_Elev_CONUS.tif"
}

# get raster CRS from one of the burn severity rasters
raster_path = fires_mtbs["McCash_2021"]
with rasterio.open(raster_path) as src:
    raster_crs = src.crs

# load CAL FIRE fuel treatment data and reproject to raster CRS
gdb_path = "SourceDatasets/CALFIRE_FuelReductionProjects.gdb"
calfire = gpd.read_file(gdb_path, layer = "CMDash_ProjectTreatments")
calfire = calfire.to_crs(raster_crs)


#------------------------
# HELPER FUNCTIONS
#------------------------

def load_fire_raster(raster_path):
    """
    Load a fire raster and return the array, transform, and CRS

    Inputs:
    - raster_path: file path to the raster
    
    Outputs:
    - 2D numpy array of raster values
    - affine transform for converting between pixel and spatial coordinates
    - coordinate reference system of the raster
    """
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs

    return arr, transform, crs


def raster_to_df(arr, transform):
    """
    Convert a raster array to a pandas DataFrame with spatial coordinates

    Inputs:
    - arr: 2D numpy array of raster values
    - transform: affine transform for converting between pixel and spatial coordinates

    Outputs:
    - pandas DataFrame with columns for raster value, x coordinate, and y coordinate
    """
    # get row and column indices of valid pixels (assuming -9999 is nodata) and convert to spatial coordinates
    rows, cols = np.where(arr != -9999)
    xs, ys = rasterio.transform.xy(transform, rows, cols)

    # create DataFrame
    df = pd.DataFrame({
        "severity": arr[rows, cols],
        "x": xs,
        "y": ys
    })

    return df


def add_treated(gdf, calfire_clip):
    """
    Add a binary "treated" variable to the GeoDataFrame indicating whether each pixel was treated by CAL FIRE

    Inputs:
    - gdf: GeoDataFrame of pixels with geometry column
    - calfire_clip: GeoDataFrame of CAL FIRE treatments clipped to raster extent

    Outputs:
    - GeoDataFrame with new "treated" column (1 if treated, 0 if not)
    """
    # reset index so grouping works cleanly and reproject pixels to same CRS as CAL FIRE data
    gdf = gdf.reset_index(drop=True)
    gdf = gdf.to_crs(calfire_clip.crs)

    # spatial join and collapse duplicates
    join = gpd.sjoin(
        gdf,
        calfire_clip[['PROJECT_ID','geometry']],
        how='left',
        predicate='intersects'
    )
    treated = join.groupby(join.index)["PROJECT_ID"].apply(lambda x: x.notnull().any())

    # add treated variable to original GeoDataFrame
    gdf["treated"] = treated.astype(int)

    return gdf

def sample_raster(gdf, raster_path, colname):
    """
    Sample raster values at the locations of a GeoDataFrame

    Inputs:
    - gdf: GeoDataFrame with geometry column
    - raster_path: file path to the raster
    - colname: name of the column to store raster values

    Outputs:
    - GeoDataFrame with new column containing raster values
    """
    # open raster and reproject GeoDataFrame if necessary
    with rasterio.open(raster_path) as src:
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        # get coordinates of pixel centroids and sample raster values at those locations
        coords = [(p.x, p.y) for p in gdf.geometry]
        values = [v[0] for v in src.sample(coords)]

        # add sampled values to GeoDataFrame
        gdf[colname] = values

    return gdf


def download_prism(variable, yyyymm, save_folder = "SourceDatasets/prism_data"):
    """
    Download PRISM climate data for a given variable and month

    Inputs:
    - variable: PRISM variable to download (e.g. "ppt", "tmax", "vpdmax")
    - yyyymm: year and month to download in YYYYMM format (e.g. "202307")
    - save_folder: folder to save downloaded data (default "SourceDatasets/prism_data")

    Outputs:
    - file path to the downloaded raster
    """
    # create save folder if it doesn't exist and define zip file path
    os.makedirs(save_folder, exist_ok=True)
    zip_path = f"{save_folder}/{variable}_{yyyymm}.zip"

    # download and extract data if not already downloaded
    if not os.path.exists(zip_path):
        url = f"https://services.nacse.org/prism/data/get/us/4km/{variable}/{yyyymm}"
        r = requests.get(url)

        with open(zip_path, "wb") as f:
            f.write(r.content)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(save_folder)

    # find extracted .tif file and return its path
    for f in os.listdir(save_folder):
        if variable in f and yyyymm in f and f.endswith(".tif"):
            return os.path.join(save_folder, f)


#------------------------
# MAIN CODE
#------------------------

# get bounding box of all fires to clip CAL FIRE data and speed up spatial joins
all_bounds = gpd.GeoDataFrame(
    geometry=[box(*gpd.GeoSeries([box(*rasterio.open(p).bounds) for p in fires_mtbs.values()]).total_bounds)],
    crs=raster_crs
)
calfire_clip = gpd.clip(calfire, all_bounds)

# loop through each fire, convert raster to GeoDataFrame, sample rasters, add treatment variable, and combine into final dataset
all_cells = []
for fire_name, raster_path in fires_mtbs.items():
    arr, transform, crs = load_fire_raster(raster_path)
    df = raster_to_df(arr, transform)

    # sample every 10th pixel to reduce dataset size for testing
    df = df.iloc[::10]

    df["fire"] = fire_name

    # build geometry column from x and y coordinates
    df["geometry"] = [Point(xy) for xy in zip(df["x"], df["y"])]
    gdf = gpd.GeoDataFrame(df, geometry = "geometry", crs = crs)

    # add prism data
    ppt = download_prism("ppt", "202307")
    tmax = download_prism("tmax", "202307")
    vpd = download_prism("vpdmax", "202307")
    gdf = sample_raster(gdf, ppt, "ppt")
    gdf = sample_raster(gdf, tmax, "tmax")
    gdf = sample_raster(gdf, vpd, "vpdmax")

    # add landfire data
    for var, path in landfire.items():
        gdf = sample_raster(gdf, path, var)

    # add CAL FIRE data
    gdf = add_treated(gdf, calfire_clip)

    # append to list of all cells
    all_cells.append(gdf)

# combine all cells into final DataFrame and drop geometry
final_df = pd.concat(all_cells, ignore_index=True)
final_df = final_df.drop(columns="geometry")

# group fuel models into broader categories
final_df["fuel_model_group"] = "Other"
final_df.loc[final_df["fuel_model"].isin([91,92,93,98,99]), "fuel_model_group"] = "Non-Burnable"
final_df.loc[final_df["fuel_model"].between(101,109), "fuel_model_group"] = "Grass"
final_df.loc[final_df["fuel_model"].between(121,124), "fuel_model_group"] = "Grass-Shrub"
final_df.loc[final_df["fuel_model"].between(141,149), "fuel_model_group"] = "Shrub"
final_df.loc[final_df["fuel_model"].between(161,165), "fuel_model_group"] = "Timber-Understory"
final_df.loc[final_df["fuel_model"].between(181,189), "fuel_model_group"] = "Timber Litter"
final_df.loc[final_df["fuel_model"].between(201,204), "fuel_model_group"] = "Slash-Blowdown"

# group vegetation types into broader categories using LF2024_EVT lookup table
evt_lookup = pd.read_csv("SourceDatasets/LF2024_EVT.csv")
lf_map = dict(zip(evt_lookup["VALUE"], evt_lookup["EVT_LF"]))
final_df["vegetation_type_group"] = final_df["vegetation_type"].map(lf_map)

# save final dataset to csv
final_df.to_csv("AnalysisDatasets/new_fires_dataset.csv", index=False)



