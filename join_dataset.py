import rasterio
from rasterio.plot import show
from rasterio.features import rasterize
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, box
from datetime import datetime, timedelta
from shapely.ops import nearest_points
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


def sample_and_plot_fire(fire_name, raster_path, calfire_gdf, n_points=5000, sample_plot=2000, plot = False):
    """
    Samples points from a raster, determines if they fall in CAL FIRE polygons,
    and plots the fire raster, clipped polygons, and treated/untreated points.
    """
    # Open raster
    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        data = src.read(1)
        mask = data != src.nodata
        valid_idx = np.argwhere(mask)

        # sample points
        if n_points > len(valid_idx):
            n_points = len(valid_idx)
        sampled_idx = valid_idx[np.random.choice(len(valid_idx), n_points, replace=False)]

        xs, ys = rasterio.transform.xy(src.transform, sampled_idx[:,0], sampled_idx[:,1])
        values = data[sampled_idx[:,0], sampled_idx[:,1]]

        points_gdf = gpd.GeoDataFrame({
            'severity': values
        }, geometry=[Point(x,y) for x,y in zip(xs, ys)], crs=raster_crs)

        # get raster bounds for clipping polygons
        bounds = src.bounds
        raster_bbox = gpd.GeoDataFrame(
            geometry=[box(bounds.left, bounds.bottom, bounds.right, bounds.top)],
            crs=raster_crs
        )

    # Clip CAL FIRE polygons to raster bounds
    calfire_clip = gpd.clip(calfire_gdf, raster_bbox)

    # Spatial join to assign treated=1 if inside polygon
    points_gdf = gpd.sjoin(points_gdf, calfire_clip[['PROJECT_ID', 'geometry']], how='left', predicate='intersects')
    points_gdf['treated'] = points_gdf['PROJECT_ID'].notnull().astype(int)
    points_gdf = points_gdf.drop(columns=['PROJECT_ID', 'index_right'])

    # Add fire name
    points_gdf['fire'] = fire_name

    if plot:
        # Plot for sanity check
        fig, ax = plt.subplots(figsize=(10,10))

        # Raster background
        with rasterio.open(raster_path) as src:
            show(src, ax=ax, cmap='Greys', title=f'{fire_name} rDNBR with Treatments')

        # Clipped polygons
        calfire_clip.plot(ax=ax, facecolor='green', edgecolor='darkgreen', alpha=0.3, label='Fuel treatment polygons')

        # Sample points for plotting
        plot_points = points_gdf
        if len(plot_points) > sample_plot:
            plot_points = plot_points.sample(n=sample_plot, random_state=42)

        plot_points[plot_points['treated']==0].plot(ax=ax, color='red', markersize=5, label='Untreated')
        plot_points[plot_points['treated']==1].plot(ax=ax, color='blue', markersize=5, label='Treated')

        ax.legend()
        plt.show()

    return points_gdf



def extract_raster_values(points_gdf, raster_path, col_name):
    """
    Adds a column to points_gdf with values sampled from raster_path at each point.
    points_gdf: GeoDataFrame with geometry column
    raster_path: path to the raster (.tif)
    col_name: name of new column to create
    """
    with rasterio.open(raster_path) as src:
        # Reproject points if CRS mismatch
        if points_gdf.crs != src.crs:
            points_gdf = points_gdf.to_crs(src.crs)

        coords = [(x,y) for x,y in zip(points_gdf.geometry.x, points_gdf.geometry.y)]
        values = [val[0] for val in src.sample(coords)]
    
    points_gdf[col_name] = values
    return points_gdf

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



all_samples = []

for fire_name, raster_path in fires_mtbs.items():
    gdf_fire = sample_and_plot_fire(fire_name, raster_path, calfire_gdf, plot = False)

    fire_month = get_fire_month_from_filename(raster_path)
    
    # Download PRISM for this fire month
    ppt_path = download_prism_month("ppt", fire_month)
    tmax_path = download_prism_month("tmax", fire_month)
    vpd_path = download_prism_month("vpdmax", fire_month)
    
    # Extract values
    gdf_fire = extract_raster_values(gdf_fire, ppt_path, "ppt")
    gdf_fire = extract_raster_values(gdf_fire, tmax_path, "tmax")
    gdf_fire = extract_raster_values(gdf_fire, vpd_path, "vpdmax") # maximum vapor pressure deficit, atmospheric dryness indicator

    all_samples.append(gdf_fire)

severity_df = pd.concat(all_samples, ignore_index=True)

severity_df = extract_raster_values(severity_df, "Datasets/landfire/LF2020_FBFM40_200_CONUS/Tif/LC20_F40_200.tif", "fuel_model")
severity_df = extract_raster_values(severity_df, "Datasets/landfire/LF2016_EVT_200_CONUS/Tif/LC16_EVT_200.tif", "vegetation_type")
severity_df = extract_raster_values(severity_df, "Datasets/landfire/LF2020_CC_200_CONUS/Tif/LC20_CC_200.tif", "canopy_cover")
severity_df = extract_raster_values(severity_df, "Datasets/landfire/LF2020_SlpD_CONUS/Tif/LF2020_SlpD_CONUS.tif", "slope_deg")
severity_df = extract_raster_values(severity_df, "Datasets/landfire/LF2020_Asp_CONUS/Tif/LF2020_Asp_CONUS.tif", "aspect_deg")
severity_df = extract_raster_values(severity_df, "Datasets/landfire/LF2020_Elev_CONUS/Tif/LF2020_Elev_CONUS.tif", "elevation_m")


final_df = severity_df.drop(columns="geometry")

final_df.to_csv("combined_dataset.csv", index=False)


"""

1) Treatment coverage vs burn severity
    Compute % of treated vs untreated points.
    Compare mean/high-severity proportions in treated vs untreated.
    Identify mismatches (high severity pixels that were treated, low severity untreated).

2) Severity vs environmental variables
    Slope, aspect, canopy cover, fuel type, drought indices.
    Look at relationships visually (boxplots, scatterplots) and with summary stats.

3) Fire-specific summaries
    For each fire, quantify % of area treated and severity distribution (helps show variability across fires)
    Descriptive stats that justifies the first solution (better placement of treatments) and establish “baseline conditions” in white paper.

4) Feature preparation
    Add climate/weather data from raws or somewhere else (prism?)
    Convert fuel type, vegetation cover, project type, etc., to dummy variables or ordinal codes.
    Normalize or scale continuous predictors for modeling if needed.

5) Predictive modeling
    Goal: predict likelihood of high-severity fire so we can test and optimize treatment placement.
    a) Define outcome
        High-severity burn: 1 (RDNBR threshold), low/moderate severity: 0
    b) Candidate models
        Logistic regression — interpretable, good for showing which variables drive severity.
        Random forest / gradient boosting — handles nonlinear relationships, rank feature importance, robust to correlated predictors.
        Spatial models (geographically weighted regression) if you want to account for location effects.
    c) Validation
        Split points per fire into train/test or use k-fold cross-validation.
        Metrics: accuracy, ROC-AUC, or % correctly predicted high-severity areas.

6) Translating model outputs to solutions
    Severity-based prioritization (Solution 1)
        Use model predictions to highlight pixels or areas with highest predicted severity.
        Compare these areas to existing CAL FIRE treatments - quantify improvement if placement changes.
    Dynamic, climate-responsive planning (Solution 2)
        Use seasonal drought/fuel conditions to modify severity predictions.
        Map high-risk zones under different scenarios.

    Visualizations
        Maps showing predicted severity risk vs current treatments.
        Histograms of severity distributions per treatment type.
        Tables of % improvement under new prioritization.








"""