import os
from datetime import datetime
import ee
import json
import numpy as np
import geemap.foliumap as gee_folium
import leafmap.foliumap as leaf_folium
import gradio as gr
import pandas as pd
import geopandas as gpd
import plotly.express as px
import branca.colormap as cm
from shapely.ops import transform
import pyproj
from io import BytesIO
import requests
import kml2geojson
import folium
import xml.etree.ElementTree as ET
from fastapi import FastAPI, HTTPException, Response
from urllib.parse import unquote
from pydantic import BaseModel, HttpUrl


app = FastAPI()

# --- Helper Functions ---

def one_time_setup():
    """Initializes the Earth Engine API."""
    try:
        # Attempt to initialize with default credentials
        ee.Initialize()
    except Exception:
        try:
            # Fallback to service account credentials if default init fails
            credentials_path = os.path.expanduser("~/.config/earthengine/credentials.json")
            ee_credentials = os.environ.get("EE_GRAD")
            if ee_credentials:
                os.makedirs(os.path.dirname(credentials_path), exist_ok=True)
                with open(credentials_path, "w") as f:
                    f.write(ee_credentials)
                credentials = ee.ServiceAccountCredentials('ujjwal@ee-ujjwaliitd.iam.gserviceaccount.com', credentials_path)
                ee.Initialize(credentials, project='ee-ujjwaliitd')
        except Exception as inner_e:
            # If the fallback also fails, print the error
            print(f"Earth Engine initialization failed: {inner_e}")


def _process_spatial_data(data_bytes: BytesIO) -> gpd.GeoDataFrame:
    """Core function to process bytes of a KML or GeoJSON file."""
    # Read the first few bytes to determine file type without consuming the stream
    start_of_file = data_bytes.read(100)
    data_bytes.seek(0)  # Reset stream position

    # Check if the file is KML (XML-based)
    if start_of_file.strip().lower().startswith(b'<?xml'):
        try:
            geojson_data = kml2geojson.convert(data_bytes)
            if not geojson_data or not geojson_data[0].get("features"):
                raise ValueError("KML file is empty or has no features.")
            features = geojson_data[0]["features"]
            input_gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        except Exception as e:
            raise ValueError(f"Failed to process KML data: {e}")
    # Otherwise, assume it's a format geopandas can read from databytes
    else:
        try:
            geojson_str = data_bytes.read().decode('utf-8')
            input_gdf = gpd.read_file(geojson_str)
        except Exception as e:
            raise ValueError(f"Failed to read GeoJSON or other vector data: {e}")
    return input_gdf

def get_gdf_from_file(file_obj):
    """Reads a KML or GeoJSON file from a Gradio file object and returns a GeoDataFrame."""
    if file_obj is None:
        return None
    with open(file_obj.name, 'rb') as f:
        data_bytes = BytesIO(f.read())
    return _process_spatial_data(data_bytes)

def get_gdf_from_url(url: str) -> gpd.GeoDataFrame:
    """Downloads and reads a KML/GeoJSON from a URL."""
    if not url or not url.strip():
        return None

    # Handle Google Drive URLs
    if "drive.google.com" in url:
        if "/file/d/" in url:
            file_id = url.split('/d/')[1].split('/')[0]
        elif "open?id=" in url:
            file_id = url.split('open?id=')[1].split('&')[0]
        else:
            raise ValueError("Unsupported Google Drive URL format. Please provide a direct link or a shareable link with 'open?id=' or '/file/d/'.")
        download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    else:
        download_url = url

    try:
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()
        data_bytes = BytesIO(response.content)
        return _process_spatial_data(data_bytes)
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to download file from URL: {e}")


def find_best_epsg(geometry) -> int:
    """Finds the most suitable EPSG code for a given geometry based on its centroid."""
    if geometry.geom_type == "Polygon":
        centroid = geometry.centroid
    else:
        raise ValueError("Geometry is not a Polygon.")

    common_epsg_codes = [
        7761,  # Gujarat
        7774,  # Rajasthan
        7766,  # MadhyaPradesh
        7767,  # Maharastra
        7755,  # India
        # Add other relevant state/country EPSG codes here
    ]

    for epsg in common_epsg_codes:
        try:
            crs = pyproj.CRS.from_epsg(epsg)
            area_of_use = crs.area_of_use.bounds
            if (area_of_use[0] <= centroid.x <= area_of_use[2]) and \
               (area_of_use[1] <= centroid.y <= area_of_use[3]):
                return epsg
        except pyproj.exceptions.CRSError:
            continue
    return 4326 # Default to WGS84 if no suitable projection is found

def shape_3d_to_2d(shape):
    """Converts a 3D geometry to 2D."""
    if shape.has_z:
        return transform(lambda x, y, z: (x, y), shape)
    return shape

def preprocess_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Preprocesses a GeoDataFrame by converting geometries to 2D and fixing invalid ones."""
    gdf["geometry"] = gdf["geometry"].apply(shape_3d_to_2d)
    gdf["geometry"] = gdf.buffer(0)
    return gdf

def to_best_crs(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Converts a GeoDataFrame to the most suitable CRS."""
    if not gdf.empty and gdf["geometry"].iloc[0] is not None:
        best_epsg_code = find_best_epsg(gdf.to_crs(epsg=4326)["geometry"].iloc[0])
        return gdf.to_crs(epsg=best_epsg_code)
    return gdf

def is_valid_polygon(geometry_gdf):
    """Checks if the geometry in a GeoDataFrame is a valid, non-empty Polygon."""
    if geometry_gdf.empty:
        return False
    geometry = geometry_gdf.geometry.item()
    return (geometry.type == 'Polygon') and (not geometry.is_empty)

def add_geometry_to_map(m, geometry_gdf, buffer_geometry_gdf, opacity=0.3):
    """Adds geometry and its buffer to a folium map."""
    if buffer_geometry_gdf is not None and not buffer_geometry_gdf.empty:
        folium.GeoJson(
            buffer_geometry_gdf.to_crs(epsg=4326),
            name="Geometry Buffer",
            style_function=lambda x: {"color": "red", "fillOpacity": opacity, "fillColor": "red"}
        ).add_to(m)
    if geometry_gdf is not None and not geometry_gdf.empty:
        folium.GeoJson(
            geometry_gdf.to_crs(epsg=4326),
            name="Geometry",
            style_function=lambda x: {"color": "blue", "fillOpacity": opacity, "fillColor": "blue"}
        ).add_to(m)
    return m

def get_wayback_data():
    """Fetches and parses Wayback imagery data from ArcGIS."""
    try:
        url = "https://wayback.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/WMTS/1.0.0/WMTSCapabilities.xml"
        response = requests.get(url)
        response.raise_for_status()  # Ensure request was successful

        # Parse XML
        root = ET.fromstring(response.content)

        ns = {
            "wmts": "https://www.opengis.net/wmts/1.0",
            "ows": "https://www.opengis.net/ows/1.1",
            "xlink": "https://www.w3.org/1999/xlink",
        }

        # Use a robust XPath to find all 'Layer' elements anywhere in the document.
        # This is less brittle than specifying the full path.
        layers = root.findall(".//wmts:Contents/wmts:Layer", ns)

        layer_data = []
        for layer in layers:
            title = layer.find("ows:Title", ns)
            identifier = layer.find("ows:Identifier", ns)
            resource = layer.find("wmts:ResourceURL", ns)  # Tile URL template

            title_text = title.text if title is not None else "N/A"
            identifier_text = identifier.text if identifier is not None else "N/A"
            url_template = resource.get("template") if resource is not None else "N/A"

            layer_data.append({"Title": title_text, "ResourceURL_Template": url_template})

        wayback_df = pd.DataFrame(layer_data)
        wayback_df["date"] = pd.to_datetime(wayback_df["Title"].str.extract(r"(\d{4}-\d{2}-\d{2})").squeeze(), errors="coerce")
        wayback_df.set_index("date", inplace=True)
        return wayback_df.sort_index(ascending=False)

    except Exception as e:
        print(f"Could not fetch or parse Wayback data: {e}")
        return pd.DataFrame()


def get_dem_slope_maps(ee_geometry, map_bounds, wayback_url, wayback_title, zoom=12,):
    """Creates DEM and Slope maps from SRTM data, using wayback tiles as a basemap if available."""

    print(wayback_url, wayback_title)

    # --- DEM Map ---
    dem_map = gee_folium.Map(zoom_start=zoom)
    if wayback_url:
        dem_map.add_tile_layer(url=wayback_url, name=wayback_title, attribution="Esri")
    
    dem_map_html = "<div>No DEM data available for this area.</div>"
    try:
        dem_layer = ee.Image("USGS/SRTMGL1_003").resample("bilinear").reproject(crs="EPSG:4326", scale=30).clip(ee_geometry)
        stats = dem_layer.reduceRegion(reducer=ee.Reducer.minMax(), geometry=ee_geometry, scale=30, maxPixels=1e9).getInfo()
        print(stats)

        if stats and stats.get('elevation_min') is not None:
            min_val, max_val = stats['elevation_min'], stats['elevation_max']
            vis_params = {"min": min_val, "max": max_val, "palette": ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']}
            dem_map.addLayer(dem_layer, vis_params, "Elevation")
            dem_map.add_colorbar(vis_params=vis_params, label="Elevation (m)")

        dem_map.addLayerControl()
        dem_map.fit_bounds(map_bounds, padding=(10, 10))
        dem_map_html = dem_map._repr_html_()

    except Exception as e:
        print(f"Error creating DEM map: {e}")
        dem_map_html = f"<div>Error creating DEM map: {e}</div>"

    # ---Slope Map --- #
    slope_map = gee_folium.Map(zoom_start=zoom)
    if wayback_url:
            slope_map.add_tile_layer(url=wayback_url, name=wayback_title, attribution="Esri")
    
    slope_map_html = "<div>No Slope data available for this area.</div>"
    try:
        dem_for_slope = ee.Image("USGS/SRTMGL1_003")
        # Calculate slope. The result is an image with slope values in degrees.
        slope_layer = ee.Terrain.slope(dem_for_slope).clip(ee_geometry)

        stats = slope_layer.reduceRegion(reducer=ee.Reducer.minMax(), geometry=ee_geometry, scale=30, maxPixels=1e9).getInfo()
        print(stats)

        if stats and stats.get('slope_min') is not None:
            min_val, max_val = stats['slope_min'], stats['slope_max']
            vis_params = {"min": min_val, "max": max_val, "palette": ['#0000FF', '#00FF00', '#FFFF00', '#FF0000']}
            slope_map.addLayer(slope_layer, vis_params, "Slope")
            slope_map.add_colorbar(vis_params=vis_params, label="Slope (degrees)")

        slope_map.addLayerControl()
        slope_map.fit_bounds(map_bounds, padding=(10, 10))
        slope_map_html = slope_map._repr_html_()

    except Exception as e:
        print(f"Error creating Slope map: {e}")
        slope_map_html = f"<div>Error creating Slope map: {e}</div>"

    return dem_map_html, slope_map_html

def add_indices(image, nir_band, red_band, blue_band, green_band, swir_band, swir2_band, evi_vars):
    """Calculates and adds multiple vegetation indices to an Earth Engine image."""
    nir = image.select(nir_band).divide(10000)
    red = image.select(red_band).divide(10000)
    blue = image.select(blue_band).divide(10000)
    green = image.select(green_band).divide(10000)
    swir = image.select(swir_band).divide(10000)
    swir2 = image.select(swir2_band).divide(10000)

    # Previously existing indices
    ndvi = image.normalizedDifference([nir_band, red_band]).rename('NDVI')
    evi = image.expression(
        'G * ((NIR - RED) / (NIR + C1 * RED - C2 * BLUE + L))', {
            'NIR': nir, 'RED': red, 'BLUE': blue,
            'G': evi_vars['G'], 'C1': evi_vars['C1'], 'C2': evi_vars['C2'], 'L': evi_vars['L']
        }).rename('EVI')
    evi2 = image.expression(
        'G * (NIR - RED) / (NIR + L + C * RED)', {
            'NIR': nir, 'RED': red,
            'G': evi_vars['G'], 'L': evi_vars['L'], 'C': evi_vars['C']
        }).rename('EVI2')
    try:
        table = ee.FeatureCollection('projects/in793-aq-nb-24330048/assets/cleanedVDI').select(
            ["B2", "B4", "B8", "cVDI"], ["Blue", "Red", "NIR", 'cVDI'])
        classifier = ee.Classifier.smileRandomForest(50).train(
            features=table, classProperty='cVDI', inputProperties=['Blue', 'Red', 'NIR'])
        rf = image.classify(classifier).multiply(ee.Number(0.2)).add(ee.Number(0.1)).rename('RandomForest')
    except Exception as e:
        print(f"Random Forest calculation failed: {e}")
        rf = ee.Image.constant(0).rename('RandomForest')
    ci = image.expression(
        '(-3.98 * (BLUE/NIR) + 12.54 * (GREEN/NIR) - 5.49 * (RED/NIR) - 0.19) / ' +
        '(-21.87 * (BLUE/NIR) + 12.4 * (GREEN/NIR) + 19.98 * (RED/NIR) + 1) * 2.29', {
            'NIR': nir, 'RED': red, 'BLUE': blue, 'GREEN': green
        }).clamp(0, 1).rename('CI')
    gujvdi = image.expression(
        '0.5 * (NIR - RED) / (NIR + 6 * RED - 8.25 * BLUE - 0.01)', {
            'NIR': nir, 'RED': red, 'BLUE': blue
        }).rename('GujVDI')
    mndwi = image.normalizedDifference([green_band, swir_band]).rename('MNDWI')

    # Newly added indices
    savi = image.expression('(1 + L) * (NIR - RED) / (NIR + RED + L)', {
        'NIR': nir, 'RED': red, 'L': 0.5
    }).rename('SAVI')
    mvi = image.expression('(NIR - (GREEN + SWIR)) / (NIR + (GREEN + SWIR))', {
        'NIR': nir, 'GREEN': green, 'SWIR': swir
    }).rename('MVI')
    nbr = image.normalizedDifference([nir_band, swir2_band]).rename('NBR')
    gci = image.expression('(NIR - GREEN) / GREEN', {
        'NIR': nir, 'GREEN': green
    }).rename('GCI')

    return image.addBands([ndvi, evi, evi2, rf, ci, gujvdi, mndwi, savi, mvi, nbr, gci])


# --- Gradio App Logic ---

# Initialize GEE and fetch wayback data once at the start
one_time_setup()
WAYBACK_DF = get_wayback_data()

def process_and_display(file_obj, url_str, buffer_m, progress=gr.Progress()):
    """Main function to process the uploaded file or URL and generate initial outputs."""
    if file_obj is None and not (url_str and url_str.strip()):
        return None, "Please upload a file or provide a URL.", None, None, None, None, None


    progress(0, desc="Reading and processing geometry...")
    try:
        input_gdf = get_gdf_from_file(file_obj) if file_obj is not None else get_gdf_from_url(url_str)
        input_gdf = preprocess_gdf(input_gdf)
        geometry_gdf = next((input_gdf.iloc[[i]] for i in range(len(input_gdf)) if is_valid_polygon(input_gdf.iloc[[i]])), None)
        if geometry_gdf is None:
            return None, "No valid polygon found in the provided file.", None, None, None, None, None
        geometry_gdf = to_best_crs(geometry_gdf)
        outer_geometry_gdf = geometry_gdf.copy()
        outer_geometry_gdf["geometry"] = outer_geometry_gdf["geometry"].buffer(buffer_m)
        buffer_geometry_gdf = gpd.GeoDataFrame(
            geometry=[outer_geometry_gdf.unary_union.difference(geometry_gdf.unary_union)],
            crs=geometry_gdf.crs
        )
    except Exception as e:
        return None, f"Error processing file: {e}", None, None, None, None, None

    progress(0.5, desc="Generating maps and stats...")
    m = folium.Map()
    wayback_url = None
    wayback_title = "Esri Satellite"
    if not WAYBACK_DF.empty:
        latest_item = WAYBACK_DF.iloc[0]
        wayback_title = f"Esri Wayback ({latest_item.name.strftime('%Y-%m-%d')})"
        wayback_url = (
            latest_item["ResourceURL_Template"]
            .replace("{TileMatrixSet}", "GoogleMapsCompatible")
            .replace("{TileMatrix}", "{z}")
            .replace("{TileRow}", "{y}")
            .replace("{TileCol}", "{x}")
        )
        folium.TileLayer(tiles=wayback_url, attr="Esri", name=wayback_title).add_to(m)

    m = add_geometry_to_map(m, geometry_gdf, buffer_geometry_gdf, opacity=0.3)

    bounds = geometry_gdf.to_crs(epsg=4326).total_bounds
    map_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
    m.fit_bounds(map_bounds, padding=(10, 10))
    folium.LayerControl().add_to(m)

    ee_geometry = ee.Geometry(json.loads(geometry_gdf.to_crs(4326).to_json())['features'][0]['geometry'])
    dem_html, slope_html = get_dem_slope_maps(ee_geometry, map_bounds, wayback_url=wayback_url, wayback_title=wayback_title)

    stats_df = pd.DataFrame({
        "Area (ha)": [f"{geometry_gdf.area.item() / 10000:.2f}"],
        "Perimeter (m)": [f"{geometry_gdf.length.item():.2f}"],
        "Centroid (Lat, Lon)": [f"({geometry_gdf.to_crs(4326).centroid.y.iloc[0]:.6f}, {geometry_gdf.to_crs(4326).centroid.x.iloc[0]:.6f})"]
    })
    geometry_json = geometry_gdf.to_json()
    buffer_geometry_json = buffer_geometry_gdf.to_json()
    progress(1, desc="Done!")
    return m._repr_html_(), None, stats_df, dem_html, slope_html, geometry_json, buffer_geometry_json

@app.get("/api/geometry")
def calculate_geometry_metrics(file_url: str):
    """
    Accepts a URL to a KML/GeoJSON file, calculates the area and
    perimeter of the first valid polygon, and returns the results
    in a CSV format compatible with Google Sheets' IMPORTDATA.
    """
    try:
        decoded_url = unquote(file_url)
        input_gdf = get_gdf_from_url(decoded_url)

        if input_gdf is None or input_gdf.empty:
            raise ValueError("Could not read geometry from the provided URL.")

        input_gdf = preprocess_gdf(input_gdf)
        geometry_gdf = next((input_gdf.iloc[[i]] for i in range(len(input_gdf)) if is_valid_polygon(input_gdf.iloc[[i]])), None)

        if geometry_gdf is None:
            raise ValueError("No valid polygon found in the provided file.")

        projected_gdf = to_best_crs(geometry_gdf)
        area_hectares = projected_gdf.area.item() / 10000
        perimeter_meters = projected_gdf.length.item()

        centroid_gdf = projected_gdf.to_crs(epsg=4326)
        centroid_point = centroid_gdf.centroid.item()

        data_row = (
            f"area_hectares, {round(area_hectares, 4)},"
            f"perimeter_meters, {round(perimeter_meters, 4)},"
            f"latitude, {round(centroid_point.y, 4)},"
            f"longitude, {round(centroid_point.x, 4)}"
        )
        csv_output = f"{data_row}"
        return Response(content=csv_output, media_type="text/csv")

    except ValueError as e:
        # Handle specific errors with a 400 Bad Request
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Handle any other unexpected errors with a 500
        print(f"An unexpected error occurred in /api/geometry: {e}")
        raise HTTPException(status_code=500, detail="An unexpected server error occurred.")

            
def calculate_indices(
    geometry_json, buffer_geometry_json, veg_indices, evi_vars, date_range,
    min_year, max_year, progress=gr.Progress()
):
    """Calculates vegetation indices based on user inputs."""
    one_time_setup()

    if not all([geometry_json, buffer_geometry_json, veg_indices]):
        return "Please process a file and select at least one index first.", None, None, None

    try:
        geometry_gdf = gpd.read_file(geometry_json)
        buffer_geometry_gdf = gpd.read_file(buffer_geometry_json)
        ee_geometry = ee.Geometry(json.loads(geometry_gdf.to_crs(4326).to_json())['features'][0]['geometry'])
        buffer_ee_geometry = ee.Geometry(json.loads(buffer_geometry_gdf.to_crs(4326).to_json())['features'][0]['geometry'])

        start_day, start_month = date_range[0].day, date_range[0].month
        end_day, end_month = date_range[1].day, date_range[1].month
        dates = [
            (f"{year}-{start_month:02d}-{start_day:02d}", f"{year}-{end_month:02d}-{end_day:02d}")
            for year in range(min_year, max_year + 1)
        ]

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .select(
                ["B2", "B3", "B4", "B8", "B11", "B12", "MSK_CLDPRB"],
                ["Blue", "Green", "Red", "NIR", "SWIR", "SWIR2", "MSK_CLDPRB"]
            )
            .map(lambda img: add_indices(img, 'NIR', 'Red', 'Blue', 'Green', 'SWIR', 'SWIR2', evi_vars))
        )

        result_rows = []
        for i, (start_date, end_date) in enumerate(dates):
            progress((i + 1) / len(dates), desc=f"Processing {start_date} to {end_date}")
            filtered_collection = collection.filterDate(start_date, end_date).filterBounds(ee_geometry)
            if filtered_collection.size().getInfo() == 0:
                continue

            year_val = int(start_date.split('-')[0])
            row = {'Year': year_val, 'Date Range': f"{start_date}_to_{end_date}"}

            for veg_index in veg_indices:
                mosaic = filtered_collection.qualityMosaic(veg_index)
                mean_val = mosaic.reduceRegion(reducer=ee.Reducer.mean(), geometry=ee_geometry, scale=10, maxPixels=1e9).get(veg_index).getInfo()
                buffer_mean_val = mosaic.reduceRegion(reducer=ee.Reducer.mean(), geometry=buffer_ee_geometry, scale=10, maxPixels=1e9).get(veg_index).getInfo()

                row[veg_index] = mean_val
                row[f"{veg_index}_buffer"] = buffer_mean_val
                row[f"{veg_index}_ratio"] = (mean_val / buffer_mean_val) if buffer_mean_val and buffer_mean_val != 0 else np.nan
            result_rows.append(row)

        if not result_rows:
            return "No satellite imagery found for the selected dates.", None, None, None

        result_df = pd.DataFrame(result_rows)
        result_df = result_df.round(3)

        plots = []
        for veg_index in veg_indices:
            plot_cols = [veg_index, f"{veg_index}_buffer", f"{veg_index}_ratio"]
            existing_plot_cols = [col for col in plot_cols if col in result_df.columns]

            plot_df = result_df[['Year'] + existing_plot_cols].dropna()

            if not plot_df.empty:
                fig = px.line(plot_df, x='Year', y=existing_plot_cols, markers=True, title=f"{veg_index} Time Series")
                fig.update_layout(xaxis_title="Year", yaxis_title="Index Value")
                fig.update_xaxes(dtick=1)
                plots.append(fig)

        return None, result_df, plots, "Calculation complete."

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"An error occurred during calculation: {e}", None, None, None

def get_histogram(index_name, image, geometry, bins):
    """Calculates the histogram for an image within a given geometry using GEE."""
    try:
        # Request histogram data from Earth Engine
        hist_info = image.reduceRegion(
            reducer=ee.Reducer.fixedHistogram(min=bins[0], max=bins[-1], steps=len(bins)-1),
            geometry=geometry,
            scale=10,  # Scale in meters appropriate for Sentinel-2
            maxPixels=1e9
        ).get(index_name).getInfo()

        # Extract histogram counts
        if hist_info:
            histogram = [item[1] for item in hist_info]
            return np.array(histogram), bins
        else:
            # Return empty histogram if no data
            return np.array([0] * (len(bins) - 1)), bins
    except Exception as e:
        print(f"Could not compute histogram for {index_name}: {e}")
        return np.array([0] * (len(bins) - 1)), bins

def generate_comparison_maps(geometry_json, selected_index, selected_years, evi_vars, date_start_str, date_end_str, progress=gr.Progress()):
    """Generates side-by-side maps for a selected index and two selected years with a custom HTML legend."""
    if not geometry_json or not selected_index or not selected_years:
        return "Please process a file and select an index and years first.", "", ""
    if len(selected_years) != 2:
        return "Please select exactly two years to compare.", "", ""

    one_time_setup()
    geometry_gdf = gpd.read_file(geometry_json).to_crs(4326)
    ee_geometry = ee.Geometry(json.loads(geometry_gdf.to_json())['features'][0]['geometry'])
    bounds = geometry_gdf.total_bounds
    map_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

    start_month, start_day = map(int, date_start_str.split('-'))
    end_month, end_day = map(int, date_end_str.split('-'))

    maps_html = []
    for i, year in enumerate(selected_years):
        progress((i + 1) / 2, desc=f"Generating map for {year}")
        start_date = f"{year}-{start_month:02d}-{start_day:02d}"
        end_date = f"{year}-{end_month:02d}-{end_day:02d}"

        wayback_url = None
        wayback_title = "Default Satellite"
        if not WAYBACK_DF.empty:
            try:
                target_date = datetime(int(year), start_month, 15)
                nearest_idx = WAYBACK_DF.index.get_indexer([target_date], method='nearest')[0]
                wayback_item = WAYBACK_DF.iloc[nearest_idx]
                wayback_title = f"Esri Wayback ({wayback_item.name.strftime('%Y-%m-%d')})"
                wayback_url = (
                    wayback_item["ResourceURL_Template"]
                    .replace("{TileMatrixSet}", "GoogleMapsCompatible")
                    .replace("{TileMatrix}", "{z}")
                    .replace("{TileRow}", "{y}")
                    .replace("{TileCol}", "{x}")
                )
            except Exception as e:
                print(f"Could not find a suitable Wayback basemap for {year}: {e}")
                wayback_url = None
                wayback_title = "Default Satellite"
        
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterDate(start_date, end_date)
            .filterBounds(ee_geometry)
            .select(
                ["B2", "B3", "B4", "B8", "B11", "B12", "MSK_CLDPRB"],
                ["Blue", "Green", "Red", "NIR", "SWIR", "SWIR2", "MSK_CLDPRB"]
            )
            .map(lambda img: add_indices(img, 'NIR', 'Red', 'Blue', 'Green', 'SWIR', 'SWIR2', evi_vars))
        )

        if collection.size().getInfo() == 0:
            maps_html.append(f"<div style='text-align:center; padding-top: 50px;'>No data found for {year}.</div>")
            continue

        mosaic = collection.qualityMosaic(selected_index)
        m = gee_folium.Map(zoom_start=14)
        if wayback_url:
            m.add_tile_layer(wayback_url, name=wayback_title, attribution="Esri")
        else:
            m.add_basemap("SATELLITE")

        if selected_index in ["NDVI", "RandomForest", "GujVDI", "CI", "EVI", "EVI2"]:
            bins = [0, 0.2, 0.4, 0.6, 0.8, 1]
            histogram, _ = get_histogram(selected_index, mosaic.select(selected_index), ee_geometry, bins)
            
            total_pix = np.sum(histogram)
            formatted_histogram = ["0.00"] * len(histogram)
            if total_pix > 0:
                formatted_histogram = [f"{h*100/total_pix:.2f}" for h in histogram]
            
            # Define visualization parameters for the classified layer
            ind_vis_params = {
                "min": 0, "max": 1,
                "palette": ["#FF0000", "#FFFF00", "#FFA500", "#00FE00", "#00A400"],
            }
            m.addLayer(mosaic.select(selected_index).clip(ee_geometry), ind_vis_params, f"{selected_index} Classified Layer ({year})")

            legend_items = {
                f"0-0.2: Open/Sparse Vegetation ({formatted_histogram[0]}%)": "#FF0000",
                f"0.2-0.4: Low Vegetation ({formatted_histogram[1]}%)": "#FFFF00",
                f"0.4-0.6: Moderate Vegetation ({formatted_histogram[2]}%)": "#FFA500",
                f"0.6-0.8: Dense Vegetation ({formatted_histogram[3]}%)": "#00FE00",
                f"0.8-1: Very Dense Vegetation ({formatted_histogram[4]}%)": "#00A400",
            }

            legend_html = f'''
            <div style="
                position: fixed; 
                bottom: 20px; 
                right: 10px; 
                z-index:9999; 
                font-size:14px;
                background-color: rgba(255, 255, 255, 0.85);
                border:2px solid grey; 
                padding: 10px;
                border-radius: 6px;
                ">
                <b>{selected_index} Classification ({year})</b>
                <ul style="list-style-type: none; padding-left: 0; margin: 5px 0 0 0;">
            '''
            for text, color in legend_items.items():
                legend_html += f'<li><i style="background:{color}; width:20px; height:15px; display:inline-block; margin-right:5px; vertical-align:middle; border: 1px solid black;"></i> {text}</li>'
            legend_html += "</ul></div>"
            
            # Add the raw HTML to the map
            m.get_root().html.add_child(folium.Element(legend_html))
        
        else:  # Fallback for indices without a color bar
            vis_params = {"min": 0.0, "max": 1.0, "palette": ['FFFFFF', 'CE7E45', 'DF923D', 'F1B555', 'FCD163', '99B718', '74A901', '66A000', '529400', '3E8601', '207401', '056201', '004C00', '023B01', '012E01', '011D01', '011301']}
            clipped_image = mosaic.select(selected_index).clip(ee_geometry)
            m.addLayer(clipped_image, vis_params, f"{selected_index} {year}")
            m.add_colorbar(vis_params=vis_params, label=f"{selected_index} Value")

        folium.GeoJson(geometry_gdf, name="Geometry", style_function=lambda x: {"color": "yellow", "fillOpacity": 0, "weight": 2.5}).add_to(m)
        m.fit_bounds(map_bounds, padding=(10, 10))
        m.addLayerControl()
        maps_html.append(m._repr_html_())

    while len(maps_html) < 2:
        maps_html.append("")

    return f"Comparison generated for {selected_years[0]} and {selected_years[1]}.", maps_html[0], maps_html[1]

with gr.Blocks(title="Kamlan: KML Analyzer") as demo:
    # Hidden state to store data between steps
    geometry_data = gr.State()
    buffer_geometry_data = gr.State()
    timeseries_df_state = gr.State()

    gr.HTML("""
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <img src="https://huggingface.co/spaces/SustainabilityLabIITGN/NDVI_PERG/resolve/main/Final_IITGN-Logo-symmetric-Color.png" style="width: 10%; margin-right: auto;">
            <h1 style="text-align: center;">Kamlan: KML Analyzer</h1>
            <img src="https://huggingface.co/spaces/SustainabilityLabIITGN/NDVI_PERG/resolve/main/IFS.jpg" style="width: 10%; margin-left: auto;">
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## 1. Provide Input Geometry")
            gr.Markdown("Use either file upload OR a URL.")
            file_input = gr.File(label="Upload KML/GeoJSON File", file_types=[".kml", ".geojson"])
            url_input = gr.Textbox(label="Or Provide File URL", placeholder="e.g., https://.../my_file.kml")
            buffer_input = gr.Number(label="Buffer (meters)", value=50)
            process_button = gr.Button("Process Input", variant="primary")

            with gr.Accordion("Advanced Settings", open=False):
                gr.Markdown("### Select Vegetation Indices")
                all_veg_indices = ["GujVDI", "NDVI", "EVI", "EVI2", "RandomForest", "CI", "MNDWI", "SAVI", "MVI", "NBR", "GCI"]
                veg_indices_checkboxes = gr.CheckboxGroup(all_veg_indices, label="Indices", value=["NDVI"])

                gr.Markdown("### EVI/EVI2 Parameters")
                with gr.Row():
                    evi_g = gr.Number(label="G", value=2.5)
                    evi_c1 = gr.Number(label="C1", value=6.0)
                    evi_c2 = gr.Number(label="C2", value=7.5)
                with gr.Row():
                    evi_l = gr.Number(label="L", value=1.0)
                    evi_c = gr.Number(label="C", value=2.4)

                gr.Markdown("### Date Range")
                today = datetime.now()
                date_start_input = gr.Textbox(label="Start Date (MM-DD)", value="11-15")
                date_end_input = gr.Textbox(label="End Date (MM-DD)", value="12-15")

                with gr.Row():
                    min_year_input = gr.Number(label="Start Year", value=2019, precision=0)
                    max_year_input = gr.Number(label="End Year", value=today.year, precision=0)

            calculate_button = gr.Button("Calculate Vegetation Indices", variant="primary")


        with gr.Column(scale=2):
            gr.Markdown("## 2. Results")
            info_box = gr.Textbox(label="Status", interactive=False, placeholder="Status messages will appear here...")
            map_output = gr.HTML(label="Map View")
            stats_output = gr.DataFrame(label="Geometry Metrics")

            gr.Markdown("### Digital Elevation Model (DEM) and Slope")
            with gr.Row():
                dem_map_output = gr.HTML(label="DEM Map")
                slope_map_output = gr.HTML(label="Slope Map")

            with gr.Tabs():
                with gr.TabItem("Time Series Plot"):
                    plot_output = gr.Plot(label="Time Series Plot")
                with gr.TabItem("Time Series Data"):
                    timeseries_table = gr.DataFrame(label="Time Series Data")

    gr.Markdown("---") # Visual separator
    gr.Markdown("## 3. Year-over-Year Index Comparison")
    with gr.Row():
        comparison_index_select = gr.Radio(all_veg_indices, label="Select Index for Comparison", value="NDVI")
        comparison_years_select = gr.CheckboxGroup(label="Select Two Years to Compare", choices=[])

    compare_button = gr.Button("Generate Comparison Maps", variant="secondary")

    with gr.Row():
        map_year_1_output = gr.HTML(label="Comparison Map 1")
        map_year_2_output = gr.HTML(label="Comparison Map 2")


    # --- Event Handlers ---
    def process_on_load(request: gr.Request):
        """Checks for a 'file_url' query parameter when the app loads."""
        return request.query_params.get("file_url", "")

    demo.load(process_on_load, None, url_input)

    process_button.click(
        fn=process_and_display,
        inputs=[file_input, url_input, buffer_input],
        outputs=[map_output, info_box, stats_output, dem_map_output, slope_map_output, geometry_data, buffer_geometry_data]
    )

    def calculate_wrapper(geometry_json, buffer_json, veg_indices,
                          g, c1, c2, l, c, start_date_str, end_date_str,
                          min_year, max_year, progress=gr.Progress()):
        """Wrapper to parse inputs and handle outputs for the main calculation function."""
        try:
            evi_vars = {'G': g, 'C1': c1, 'C2': c2, 'L': l, 'C': c}
            start_month, start_day = map(int, start_date_str.split('-'))
            end_month, end_day = map(int, end_date_str.split('-'))
            date_range = (datetime(2000, start_month, start_day), datetime(2000, end_month, end_day))

            error_msg, df, plots, success_msg = calculate_indices(
                geometry_json, buffer_json, veg_indices,
                evi_vars, date_range, int(min_year), int(max_year), progress
            )

            status_message = error_msg or success_msg
            first_plot = plots[0] if plots else None
            df_display = df.round(3) if df is not None else None

            available_years = []
            if df is not None and 'Year' in df.columns:
                available_years = sorted(df['Year'].unique().tolist())

            return status_message, df_display, df, first_plot, gr.update(choices=available_years, value=[])

        except Exception as e:
            return f"An error occurred in the wrapper: {e}", None, None, None, gr.update(choices=[], value=[])

    calculate_button.click(
        fn=calculate_wrapper,
        inputs=[
            geometry_data, buffer_geometry_data, veg_indices_checkboxes,
            evi_g, evi_c1, evi_c2, evi_l, evi_c,
            date_start_input, date_end_input,
            min_year_input, max_year_input
        ],
        outputs=[info_box, timeseries_table, timeseries_df_state, plot_output, comparison_years_select]
    )

    def comparison_wrapper(geometry_json, selected_index, selected_years, g, c1, c2, l, c, start_date_str, end_date_str, progress=gr.Progress()):
        """Wrapper for the comparison map generation."""
        try:
            evi_vars = {'G': g, 'C1': c1, 'C2': c2, 'L': l, 'C': c}
            status, map1, map2 = generate_comparison_maps(
                geometry_json, selected_index, selected_years, evi_vars,
                start_date_str, end_date_str, progress
            )
            return status, map1, map2
        except Exception as e:
            return f"Error during comparison: {e}", "", ""

    compare_button.click(
        fn=comparison_wrapper,
        inputs=[
            geometry_data, comparison_index_select, comparison_years_select,
            evi_g, evi_c1, evi_c2, evi_l, evi_c,
            date_start_input, date_end_input
        ],
        outputs=[info_box, map_year_1_output, map_year_2_output]
    )

    gr.HTML("""
        <div style="text-align: center; margin-top: 20px;">
            <p>Developed by <a href="https://sustainability-lab.github.io/">Sustainability Lab</a>, <a href="https://www.iitgn.ac.in/">IIT Gandhinagar</a></p>
            <p>Supported by <a href="https://forests.gujarat.gov.in/">Gujarat Forest Department</a></p>
        </div>
    """)

def compute_index_histogram(file_url: str, index_name: str, start_date: str, end_date: str):
    """Computes histogram percentage for a vegetation index within a polygon and date range."""
    one_time_setup()
    decoded_url = unquote(file_url)
    gdf = get_gdf_from_url(decoded_url)
    gdf = preprocess_gdf(gdf)

    geometry_gdf = next((gdf.iloc[[i]] for i in range(len(gdf)) if is_valid_polygon(gdf.iloc[[i]])), None)
    if geometry_gdf is None:
        raise HTTPException(status_code=400, detail="No valid polygon found.")

    ee_geometry = ee.Geometry(json.loads(geometry_gdf.to_crs(4326).to_json())['features'][0]['geometry'])

    evi_vars = {'G': 2.5, 'C1': 6.0, 'C2': 7.5, 'L': 1.0, 'C': 2.4}
    collection = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterDate(start_date, end_date)
        .select(
            ["B2", "B3", "B4", "B8", "B11", "B12", "MSK_CLDPRB"],
            ["Blue", "Green", "Red", "NIR", "SWIR", "SWIR2", "MSK_CLDPRB"]
        )
        .map(lambda img: add_indices(img, 'NIR', 'Red', 'Blue', 'Green', 'SWIR', 'SWIR2', evi_vars))
        .filterBounds(ee_geometry)
    )

    if collection.size().getInfo() == 0:
        raise HTTPException(status_code=404, detail="No imagery found for the polygon in given date range.")

    # Use median composite to avoid cloud spikes
    # mosaic = collection.median()
    mosaic = collection.qualityMosaic(index_name)
    mean_val = mosaic.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=ee_geometry,
        scale=10,
        maxPixels=1e9
    ).get(index_name).getInfo()

    if mean_val is None:
        raise HTTPException(status_code=500, detail=f"Failed to compute mean for {index_name}.")

    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    histogram, _ = get_histogram(index_name, mosaic.select(index_name), ee_geometry, bins)

    total = histogram.sum()
    
    # result = {f'mean': mean_val}
    # for i in range(len(bins) - 1):
    #     label = f"{bins[i]}-{bins[i+1]}"
    #     pct = (histogram[i] / total * 100) if total > 0 else 0
    #     result[label] = round(pct, 2)


    output_parts = [f'mean, {round(mean_val, 4)}']

    for i in range(len(bins) - 1):
        label = f"{bins[i]}-{bins[i+1]}"
        pct = (histogram[i] / total * 100) if total > 0 else 0
        # Append each key-value pair string to the list
        output_parts.append(f'{label}, {round(pct, 2)}')

    # Join all parts into a single comma-separated string
    final_string = ", ".join(output_parts)

    # Return the string as a plain text response
    return Response(content=final_string, media_type="text/csv")

# Register endpoints for multiple indices with date range
for idx in ["NDVI", "EVI", "EVI2", "RandomForest", "CI", "GujVDI", "MNDWI", "SAVI", "MVI", "NBR", "GCI"]:
    endpoint_path = f"/api/{idx}"

    async def index_hist_endpoint(file_url: str, start_date: str, end_date: str, index_name=idx):
        return compute_index_histogram(file_url, index_name, start_date, end_date)

    app.get(endpoint_path)(index_hist_endpoint)

# Mount the Gradio app onto the FastAPI app
app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)