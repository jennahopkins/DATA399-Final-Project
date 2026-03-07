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


fires_mtbs = {
    "Antelope_2021": "Datasets/mtbs/Antelope_2021/ca4150012192920210801_20210718_20220721_rdnbr.tif",
    "Bald_2014": "Datasets/mtbs/Bald_2014/ca4090112136820140731_20140724_20140825_rdnbr.tif",
    "Camp_2018": "Datasets/mtbs/Camp_2018/ca3982012144020181108_20180719_20190722_rdnbr.tif", 
    "Carr_2018": "Datasets/mtbs/Carr_2018/ca4065012263020180723_20180710_20190729_rdnbr.tif",
    "Dixie_2021": "Datasets/mtbs/Dixie_2021/ca3987612137920210714_20200708_20220714_rdnbr.tif",
    "Hat_2018": "Datasets/mtbs/Hat_2018/ca4099012153020180809_20180804_20180820_rdnbr.tif",
    "King_2014": "Datasets/mtbs/King_2014/ca3878212060420140913_20130730_20150805_rdnbr.tif",
    "McKinney_2022": "Datasets/mtbs/McKinney_2022/ca4183012289520220729_20210830_20230815_rdnbr.tif",
    "Monument_2021": "Datasets/mtbs/Monument_2021/ca4075212333720210731_20210718_20220721_rdnbr.tif",
    "NorthComplex_2020": "Datasets/mtbs/NorthComplex_2020/ca4009112093120200817_20200809_20210711_rdnbr.tif"
}

raster_path = fires_mtbs["Antelope_2021"]
with rasterio.open(raster_path) as src:
    raster_crs = src.crs

gdb_path = "Datasets/CALFIRE_FuelReductionProjects.gdb"
calfire_gdf = gpd.read_file(gdb_path, layer = "CMDash_ProjectTreatments")
calfire_gdf = calfire_gdf.to_crs(raster_crs)

landfire_data = {
    "fuel_model": "Datasets/landfire/LF2020_FBFM40_200_CONUS/Tif/LC20_F40_200.tif",
    "vegetation_type": "Datasets/landfire/LF2016_EVT_200_CONUS/Tif/LC16_EVT_200.tif",
    "canopy_cover": "Datasets/landfire/LF2020_CC_200_CONUS/Tif/LC20_CC_200.tif",
    "slope_deg": "Datasets/landfire/LF2020_SlpD_CONUS/Tif/LF2020_SlpD_CONUS.tif",
    "aspect_deg": "Datasets/landfire/LF2020_Asp_CONUS/Tif/LF2020_Asp_CONUS.tif",
    "elevation_m": "Datasets/landfire/LF2020_Elev_CONUS/Tif/LF2020_Elev_CONUS.tif"
}


def create_grid_from_raster(raster_path, cell_size=3000):

    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs

    xmin, ymin, xmax, ymax = bounds

    x_coords = np.arange(xmin, xmax, cell_size)
    y_coords = np.arange(ymin, ymax, cell_size)

    grid_cells = []

    for x in x_coords:
        for y in y_coords:
            grid_cells.append(box(x, y, x + cell_size, y + cell_size))

    grid = gpd.GeoDataFrame({"geometry": grid_cells}, crs=crs)

    return grid


def sample_raster_at_centroids(grid, raster_path, colname):

    with rasterio.open(raster_path) as src:

        # reproject grid to raster CRS
        if grid.crs != src.crs:
            grid = grid.to_crs(src.crs)

        coords = [(geom.centroid.x, geom.centroid.y) for geom in grid.geometry]

        values = [val[0] for val in src.sample(coords)]

        grid[colname] = values

    return grid


def add_treatment_variable(grid, calfire_gdf, raster_path):

    import rasterio
    from shapely.geometry import box

    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        raster_bbox = gpd.GeoDataFrame(
            geometry=[box(bounds.left, bounds.bottom, bounds.right, bounds.top)],
            crs=src.crs
        )

    # reproject CAL FIRE to raster CRS
    calfire_gdf = calfire_gdf.to_crs(raster_bbox.crs)

    # clip to raster bounds
    calfire_clip = gpd.clip(calfire_gdf, raster_bbox)

    # reproject grid to same CRS
    grid = grid.to_crs(calfire_clip.crs)

    # spatial join
    join = gpd.sjoin(grid, calfire_clip[['PROJECT_ID','geometry']],
                     how='left', predicate='intersects')

    # collapse duplicates
    treated = join.groupby(join.index)["PROJECT_ID"].apply(lambda x: x.notnull().any())

    grid["treated"] = treated.astype(int)

    return grid

def get_fire_month_from_filename(raster_path):
    fname = os.path.basename(raster_path)
    
    # split by "_"
    parts = fname.split("_")
    
    # ignition date is second element (YYYYMMDD)
    ignition = parts[1]
    
    return ignition[:6]  # return YYYYMM



def download_prism_month(variable, yyyymm, save_folder="Datasets/prism_data"):
    
    os.makedirs(save_folder, exist_ok=True)
    
    zip_path = f"{save_folder}/{variable}_{yyyymm}.zip"
    
    if not os.path.exists(zip_path):
        url = f"https://services.nacse.org/prism/data/get/us/4km/{variable}/{yyyymm}"
        r = requests.get(url)
        
        with open(zip_path, "wb") as f:
            f.write(r.content)
        
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(save_folder)
    
    # Find extracted tif
    for file in os.listdir(save_folder):
        if variable in file and yyyymm in file and file.endswith(".tif"):
            return os.path.join(save_folder, file)
        

def plot_fire_with_grid(raster_path, grid_gdf):

    with rasterio.open(raster_path) as src:
        fig, ax = plt.subplots(figsize=(10,10))

        # plot severity raster
        show(src, ax=ax, cmap="inferno")

        # plot grid
        grid_gdf.boundary.plot(ax=ax, color="cyan", linewidth=0.5)

        ax.set_title("Fire Severity with 1.5 km Grid")
        plt.show()



all_cells = []

for fire_name, raster_path in fires_mtbs.items():
    grid = create_grid_from_raster(raster_path)
    grid = sample_raster_at_centroids(grid, raster_path, "severity")
    fire_month = get_fire_month_from_filename(raster_path)

    ppt_path = download_prism_month("ppt", fire_month)
    tmax_path = download_prism_month("tmax", fire_month)
    vpd_path = download_prism_month("vpdmax", fire_month)

    grid = sample_raster_at_centroids(grid, ppt_path, "ppt")
    grid = sample_raster_at_centroids(grid, tmax_path, "tmax")
    grid = sample_raster_at_centroids(grid, vpd_path, "vpdmax")

    for var, path in landfire_data.items():
        grid = sample_raster_at_centroids(grid, path, var)

    grid = add_treatment_variable(grid, calfire_gdf, raster_path)

    grid["fire"] = fire_name

    # compute centroids in projected CRS
    centroids = grid.geometry.centroid

    centroids_gdf = gpd.GeoDataFrame(geometry=centroids, crs=grid.crs)

    # convert centroids to lat/lon
    centroids_latlon = centroids_gdf.to_crs("EPSG:4326")

    # extract coordinates
    grid["x"] = centroids_latlon.geometry.x
    grid["y"] = centroids_latlon.geometry.y

    all_cells.append(grid)

final_geometry_df = pd.concat(all_cells, ignore_index=True)

final_df = final_geometry_df.drop(columns="geometry")

final_df["fuel_model_group"] = "Other"

final_df.loc[final_df["fuel_model"].isin([91,92,93,98,99]), "fuel_model_group"] = "Non-Burnable"
final_df.loc[final_df["fuel_model"].between(101,109), "fuel_model_group"] = "Grass"
final_df.loc[final_df["fuel_model"].between(121,124), "fuel_model_group"] = "Grass-Shrub"
final_df.loc[final_df["fuel_model"].between(141,149), "fuel_model_group"] = "Shrub"
final_df.loc[final_df["fuel_model"].between(161,165), "fuel_model_group"] = "Timber-Understory"
final_df.loc[final_df["fuel_model"].between(181,189), "fuel_model_group"] = "Timber Litter"
final_df.loc[final_df["fuel_model"].between(201,204), "fuel_model_group"] = "Slash-Blowdown"

evt_lookup = pd.read_csv("LF2024_EVT.csv")
lf_map = dict(zip(evt_lookup["VALUE"], evt_lookup["EVT_LF"]))
final_df["vegetation_type_group"] = final_df["vegetation_type"].map(lf_map)

final_df.to_csv("combined_dataset.csv", index=False)

"""
from esda.moran import Moran
from libpysal.weights import KNN

coords = list(zip(grid["x"], grid["y"]))

w = KNN.from_array(coords, k=8)
w.transform = "r"

mi = Moran(grid["severity"], w)

print(mi.I)
print(mi.p_sim)
print(final_df.shape)
"""
