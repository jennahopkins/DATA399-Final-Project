"""
join_dataset.py

Purpose:
Combine burn severity, fuel treatment, vegetation, topography, and climate datasets into a 
unified modeling dataset for northern California.

Inputs:
- MTBS burn severity rasters (contained in fires_mtbs dictionary)
- CAL FIRE fuel treatment geodatabase (CALFIRE_FuelReductionProjects.gdb, CMDash_ProjectTreatments layer)
- LANDFIRE vegetation and topography rasters (contained in landfire_data dictionary)
- PRISM climate rasters (downloaded through API based on fire ignition month)

Output:
- combined_dataset.csv

Notes:
- Used a 3 km grid to reduce spacial autocorrelation and sampled rasters at grid centroids
    - function plot_fire_with_grid() can be used to visualize the grid overlaid on the burn severity raster
- Used LF2024_EVT.csv lookup table to group LANDFIRE vegetation types into broader categories
- Grouped LANDFIRE fuel models into broader categories based on standard classifications
"""

import rasterio
from rasterio.plot import show
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import requests
import zipfile
import os
from esda.moran import Moran
from libpysal.weights import KNN


#-----------------------
# LOAD DATA
#-----------------------

# dictionary of MTBS fires and their corresponding burn severity raster paths
fires_mtbs = {
    "Antelope_2021": "SourceDatasets/mtbs/Antelope_2021/ca4150012192920210801_20210718_20220721_rdnbr.tif",
    "Bald_2014": "SourceDatasets/mtbs/Bald_2014/ca4090112136820140731_20140724_20140825_rdnbr.tif",
    "Camp_2018": "SourceDatasets/mtbs/Camp_2018/ca3982012144020181108_20180719_20190722_rdnbr.tif", 
    "Carr_2018": "SourceDatasets/mtbs/Carr_2018/ca4065012263020180723_20180710_20190729_rdnbr.tif",
    "Dixie_2021": "SourceDatasets/mtbs/Dixie_2021/ca3987612137920210714_20200708_20220714_rdnbr.tif",
    "Hat_2018": "SourceDatasets/mtbs/Hat_2018/ca4099012153020180809_20180804_20180820_rdnbr.tif",
    "King_2014": "SourceDatasets/mtbs/King_2014/ca3878212060420140913_20130730_20150805_rdnbr.tif",
    "McKinney_2022": "SourceDatasets/mtbs/McKinney_2022/ca4183012289520220729_20210830_20230815_rdnbr.tif",
    "Monument_2021": "SourceDatasets/mtbs/Monument_2021/ca4075212333720210731_20210718_20220721_rdnbr.tif",
    "NorthComplex_2020": "SourceDatasets/mtbs/NorthComplex_2020/ca4009112093120200817_20200809_20210711_rdnbr.tif"
}

# getting the crs from one of the fire rasters to use for all spatial data
raster_path = fires_mtbs["Antelope_2021"]
with rasterio.open(raster_path) as src:
    raster_crs = src.crs

# load CAL FIRE fuel treatment data and reproject to raster CRS
gdb_path = "SourceDatasets/CALFIRE_FuelReductionProjects.gdb"
calfire_gdf = gpd.read_file(gdb_path, layer = "CMDash_ProjectTreatments")
calfire_gdf = calfire_gdf.to_crs(raster_crs)

# dictionary of LANDFIRE rasters for vegetation and topography variables
landfire_data = {
    "fuel_model": "SourceDatasets/landfire/LF2020_FBFM40_200_CONUS/Tif/LC20_F40_200.tif",
    "vegetation_type": "SourceDatasets/landfire/LF2016_EVT_200_CONUS/Tif/LC16_EVT_200.tif",
    "canopy_cover": "SourceDatasets/landfire/LF2020_CC_200_CONUS/Tif/LC20_CC_200.tif",
    "slope_deg": "SourceDatasets/landfire/LF2020_SlpD_CONUS/Tif/LF2020_SlpD_CONUS.tif",
    "aspect_deg": "SourceDatasets/landfire/LF2020_Asp_CONUS/Tif/LF2020_Asp_CONUS.tif",
    "elevation_m": "SourceDatasets/landfire/LF2020_Elev_CONUS/Tif/LF2020_Elev_CONUS.tif"
}


#-----------------------
# HELPER FUNCTIONS
#-----------------------

def create_grid_from_raster(raster_path, cell_size = 3000):
    """
    Create a grid of square polygons covering the extent of the raster with specified cell size

    Inputs:
    - raster_path: path to the raster file to get bounds and CRS
    - cell_size: size of grid cells in raster units (default 3000 km)

    Output:
    - GeoDataFrame with grid cells as geometry and same CRS as raster
    """
    # open raster to get bounds and CRS
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs

    # create grid cells
    xmin, ymin, xmax, ymax = bounds
    x_coords = np.arange(xmin, xmax, cell_size)
    y_coords = np.arange(ymin, ymax, cell_size)

    # loop through coordinates to create grid cells as shapely boxes
    grid_cells = []
    for x in x_coords:
        for y in y_coords:
            grid_cells.append(box(x, y, x + cell_size, y + cell_size))

    # create GeoDataFrame from grid cells
    grid = gpd.GeoDataFrame({"geometry": grid_cells}, crs=crs)

    return grid


def sample_raster_at_centroids(grid, raster_path, colname):
    """
    Sample raster values at the centroids of the grid cells and add them as a new column

    Inputs:
    - grid: GeoDataFrame with grid cells as geometry
    - raster_path: path to the raster file to sample
    - colname: name of the column to store sampled values

    Output:
    - GeoDataFrame with sampled raster values added as a new column
    """
    # open raster and reproject grid to raster CRS if needed
    with rasterio.open(raster_path) as src:
        if grid.crs != src.crs:
            grid = grid.to_crs(src.crs)

        # get centroid coordinates
        coords = [(geom.centroid.x, geom.centroid.y) for geom in grid.geometry]
        values = [val[0] for val in src.sample(coords)]

        # add sampled values to grid
        grid[colname] = values

    return grid


def add_treatment_variable(grid, calfire_gdf, raster_path):
    """
    Add a binary "treated" variable to the grid indicating whether each cell 
    intersects with any CAL FIRE fuel treatment project

    Inputs:
    - grid: GeoDataFrame with grid cells as geometry
    - calfire_gdf: GeoDataFrame of CAL FIRE fuel treatment projects
    - raster_path: path to the raster file to get bounds and CRS for clipping

    Output:
    - GeoDataFrame with "treated" column added (1 if intersects with any treatment, 0 otherwise)
    """
    # create raster bounding box
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        raster_bbox = gpd.GeoDataFrame(
            geometry=[box(bounds.left, bounds.bottom, bounds.right, bounds.top)],
            crs=src.crs
        )

    # reproject CAL FIRE to raster CRS, clip to raster bounds, and reproject grid to same CRS
    calfire_gdf = calfire_gdf.to_crs(raster_bbox.crs)
    calfire_clip = gpd.clip(calfire_gdf, raster_bbox)
    grid = grid.to_crs(calfire_clip.crs)

    # spatial join and collapse duplicates
    join = gpd.sjoin(grid, calfire_clip[['PROJECT_ID','geometry']],
                     how='left', predicate='intersects')
    treated = join.groupby(join.index)["PROJECT_ID"].apply(lambda x: x.notnull().any())

    # assign treated variable back to grid
    grid["treated"] = treated.astype(int)

    return grid

def get_fire_month_from_filename(raster_path):
    """
    Extract the fire month (YYYYMM) from the raster filename

    Inputs:
    - raster_path: path to the raster file

    Output:
    - fire month in YYYYMM format
    """
    # extract filename from path
    fname = os.path.basename(raster_path)
    
    # split by "_" to get components
    parts = fname.split("_")
    
    # ignition date is second element
    ignition = parts[1]
    
    return ignition[:6]  # return YYYYMM



def download_prism_month(variable, yyyymm, save_folder = "SourceDatasets/prism_data"):
    """
    Download PRISM data for a specific variable and month, and return the path to the downloaded raster

    Inputs:
    - variable: PRISM variable to download (e.g. "ppt", "tmax", "vpdmax")
    - yyyymm: year and month to download in YYYYMM format
    - save_folder: folder to save downloaded data (default "SourceDatasets/prism_data")

    Output:
    - path to the downloaded PRISM raster file
    """
    # create save folder if it doesn't exist and construct zip file path
    os.makedirs(save_folder, exist_ok = True)
    zip_path = f"{save_folder}/{variable}_{yyyymm}.zip"
    
    # download and extract if zip file doesn't already exist
    if not os.path.exists(zip_path):
        url = f"https://services.nacse.org/prism/data/get/us/4km/{variable}/{yyyymm}"
        r = requests.get(url)
        
        with open(zip_path, "wb") as f:
            f.write(r.content)
        
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(save_folder)
    
    # Find extracted tif to return
    for file in os.listdir(save_folder):
        if variable in file and yyyymm in file and file.endswith(".tif"):
            return os.path.join(save_folder, file)
        

def plot_fire_with_grid(raster_path, grid_gdf):
    """
    Plot fire severity raster with overlaid grid

    Inputs:
    - raster_path: path to the fire severity raster
    - grid_gdf: GeoDataFrame with grid cells

    Output:
    - None (displays plot)
    """
    # open raster and plot with grid overlay
    with rasterio.open(raster_path) as src:
        fig, ax = plt.subplots(figsize=(10,10))

        # plot severity raster
        show(src, ax=ax, cmap="inferno")

        # plot grid
        grid_gdf.boundary.plot(ax=ax, color="cyan", linewidth=0.5)

        ax.set_title("Fire Severity with 1.5 km Grid")
        plt.show()

def morans_i(grid, value_col = "severity"):
    """
    Calculate and print Moran's I for spatial autocorrelation of a given variable in a grid

    Inputs:
    - grid: GeoDataFrame with grid cells and variable to analyze
    - value_col: name of the column containing the variable to analyze (default "severity")

    Output:
    - None (prints Moran's I and p-value)
    """
    # compute centroids in projected CRS
    coords = list(zip(grid["x"], grid["y"]))

    # create spatial weights based on k-nearest neighbors, using 8 neighbors and row-standardized weights
    w = KNN.from_array(coords, k = 8)
    w.transform = "r"

    # calculate Moran's I for the specified variable
    mi = Moran(grid[value_col], w)

    print(f"Moran's I: {mi.I}")
    print(f"P-value: {mi.p_sim}")


#-----------------------
# MAIN CODE
#-----------------------

# create grid, sample rasters, and add treatment variable for each fire
all_cells = []
for fire_name, raster_path in fires_mtbs.items():
    grid = create_grid_from_raster(raster_path)
    grid = sample_raster_at_centroids(grid, raster_path, "severity")
    fire_month = get_fire_month_from_filename(raster_path)

    # add prism data
    ppt_path = download_prism_month("ppt", fire_month)
    tmax_path = download_prism_month("tmax", fire_month)
    vpd_path = download_prism_month("vpdmax", fire_month)
    grid = sample_raster_at_centroids(grid, ppt_path, "ppt")
    grid = sample_raster_at_centroids(grid, tmax_path, "tmax")
    grid = sample_raster_at_centroids(grid, vpd_path, "vpdmax")

    # add landfire data
    for var, path in landfire_data.items():
        grid = sample_raster_at_centroids(grid, path, var)

    # add CAL FIRE data
    grid = add_treatment_variable(grid, calfire_gdf, raster_path)

    grid["fire"] = fire_name

    # compute centroids in projected CRS, convert to lat/lon, and extract coordinates for sampling rasters
    centroids = grid.geometry.centroid
    centroids_gdf = gpd.GeoDataFrame(geometry = centroids, crs = grid.crs)
    centroids_latlon = centroids_gdf.to_crs("EPSG:4326")
    grid["x"] = centroids_latlon.geometry.x
    grid["y"] = centroids_latlon.geometry.y

    # append grid for this fire to list of all cells
    all_cells.append(grid)

# concatenate all fire grids into a single GeoDataFrame and drop geomerty column for modeling
final_geometry_df = pd.concat(all_cells, ignore_index = True)
final_df = final_geometry_df.drop(columns="geometry")

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
final_df.to_csv("AnalysisDatasets/combined_dataset.csv", index=False)

