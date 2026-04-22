import rasterio
from rasterio.plot import show
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box, Point
import requests
import zipfile
import os

fires_mtbs = {
    "McCash_2021": "mtbs_new_fires/mtbs/McCash_2021/mtbs_ca4156412340420210801_10024164_20210718_20220720_rdnbr.tif",
    "Deep_2023": "mtbs_new_fires/mtbs/Deep_2023/mtbs_ca4093612292720230815_10030918_20230716_20240718_rdnbr.tif",
    "Park_2024": "mtbs_new_fires/mtbs/Park_2024/mtbs_ca3981912180320240724_10033184_20230827_20240826_rdnbr.tif"
}

landfire = {
    "fuel_model": "Datasets/landfire/LF2020_FBFM40_200_CONUS/Tif/LC20_F40_200.tif",
    "vegetation_type": "Datasets/landfire/LF2016_EVT_200_CONUS/Tif/LC16_EVT_200.tif",
    "canopy_cover": "Datasets/landfire/LF2020_CC_200_CONUS/Tif/LC20_CC_200.tif",
    "slope_deg": "Datasets/landfire/LF2020_SlpD_CONUS/Tif/LF2020_SlpD_CONUS.tif",
    "aspect_deg": "Datasets/landfire/LF2020_Asp_CONUS/Tif/LF2020_Asp_CONUS.tif",
    "elevation_m": "Datasets/landfire/LF2020_Elev_CONUS/Tif/LF2020_Elev_CONUS.tif"
}

raster_path = fires_mtbs["McCash_2021"]
with rasterio.open(raster_path) as src:
    raster_crs = src.crs

gdb_path = "Datasets/CALFIRE_FuelReductionProjects.gdb"
calfire = gpd.read_file(gdb_path, layer = "CMDash_ProjectTreatments")
calfire = calfire.to_crs(raster_crs)


def load_fire_raster(raster_path):
    with rasterio.open(raster_path) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs

    return arr, transform, crs


def raster_to_df(arr, transform):

    rows, cols = np.where(arr != -9999)

    xs, ys = rasterio.transform.xy(transform, rows, cols)

    df = pd.DataFrame({
        "severity": arr[rows, cols],
        "x": xs,
        "y": ys
    })

    return df


def add_treated(gdf, calfire_clip, raster_path):

    # 🔥 reset index so grouping works cleanly
    gdf = gdf.reset_index(drop=True)

    """
    # create raster bounding box
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        raster_bbox = gpd.GeoDataFrame(
            geometry=[box(bounds.left, bounds.bottom, bounds.right, bounds.top)],
            crs=src.crs
        )

    # reproject CAL FIRE
    calfire_gdf = calfire_gdf.to_crs(raster_bbox.crs)

    # clip to fire extent (huge performance boost)
    calfire_clip = gpd.clip(calfire_gdf, raster_bbox)
    """
    
    # reproject pixels
    gdf = gdf.to_crs(calfire_clip.crs)

    # spatial join
    join = gpd.sjoin(
        gdf,
        calfire_clip[['PROJECT_ID','geometry']],
        how='left',
        predicate='intersects'
    )

    # 🔥 CRITICAL: collapse duplicates
    treated = join.groupby(join.index)["PROJECT_ID"].apply(lambda x: x.notnull().any())

    # assign back
    gdf["treated"] = treated.astype(int)

    return gdf

def sample_raster(gdf, raster_path, colname):

    with rasterio.open(raster_path) as src:

        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)

        coords = [(p.x, p.y) for p in gdf.geometry]

        values = [v[0] for v in src.sample(coords)]

        gdf[colname] = values

    return gdf


def download_prism(variable, yyyymm, save_folder="prism"):
    os.makedirs(save_folder, exist_ok=True)

    zip_path = f"{save_folder}/{variable}_{yyyymm}.zip"

    if not os.path.exists(zip_path):
        url = f"https://services.nacse.org/prism/data/get/us/4km/{variable}/{yyyymm}"
        r = requests.get(url)

        with open(zip_path, "wb") as f:
            f.write(r.content)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(save_folder)

    for f in os.listdir(save_folder):
        if variable in f and yyyymm in f and f.endswith(".tif"):
            return os.path.join(save_folder, f)



all_bounds = gpd.GeoDataFrame(
    geometry=[box(*gpd.GeoSeries([box(*rasterio.open(p).bounds) for p in fires_mtbs.values()]).total_bounds)],
    crs=raster_crs
)

calfire_clip = gpd.clip(calfire, all_bounds)

all_cells = []

for fire_name, raster_path in fires_mtbs.items():

    arr, transform, crs = load_fire_raster(raster_path)

    df = raster_to_df(arr, transform)
    df = df.iloc[::10]

    df["fire"] = fire_name

    # build geometry (NEW STEP)
    df["geometry"] = [Point(xy) for xy in zip(df["x"], df["y"])]
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)

    # add environmental variables (same as before)
    ppt = download_prism("ppt", "202307")
    tmax = download_prism("tmax", "202307")
    vpd = download_prism("vpdmax", "202307")

    gdf = sample_raster(gdf, ppt, "ppt")
    gdf = sample_raster(gdf, tmax, "tmax")
    gdf = sample_raster(gdf, vpd, "vpdmax")

    for var, path in landfire.items():
        gdf = sample_raster(gdf, path, var)

    gdf = add_treated(gdf, calfire_clip, raster_path)

    all_cells.append(gdf)

final_df = pd.concat(all_cells, ignore_index=True)


final_df = final_df.drop(columns="geometry")

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

final_df.to_csv("new_fires_dataset.csv", index=False)



