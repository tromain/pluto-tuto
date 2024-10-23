from typing import Tuple

import folium
import numpy as np
import rasterio
from folium.plugins import MeasureControl
from rasterio.plot import reshape_as_image
from rasterio.warp import transform_bounds

import matplotlib.pyplot as plt


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize the values of an image to be between 0 and 1.

    Parameters
    ----------
    image : np.ndarray
        The input image as a NumPy array.

    Returns
    -------
    np.ndarray
        The normalized image with values between 0 and 1.
    """
    image_normalized = image.astype(np.float32)
    image_min = np.min(image_normalized)
    image_max = np.max(image_normalized)
    image_normalized = (image_normalized - image_min) / (image_max - image_min)
    return image_normalized


def get_wgs84_bounds(raster_path: str) -> Tuple[float, float, float, float]:
    """
    Convert the bounds of a raster to EPSG:4326 (WGS 84) coordinates.

    Parameters
    ----------
    raster_path : str
        The file path to the raster.

    Returns
    -------
    tuple of float
        The raster bounds in EPSG:4326 coordinates as (left, bottom, right, top).
    """
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        src_crs = src.crs
        # Transform the bounds to EPSG:4326 if necessary
        if src_crs != 'EPSG:4326':
            bounds_wgs84 = transform_bounds(src_crs, 'EPSG:4326', *bounds)
        else:
            bounds_wgs84 = bounds
    return bounds_wgs84


def save_normalized_image(raster_path: str, output_path: str) -> None:
    """
    Normalize the raster data and save it as an image.

    Parameters
    ----------
    raster_path : str
        The file path to the raster.
    output_path : str
        The output path for the normalized image.
    """
    with rasterio.open(raster_path) as src:
        if src.count == 3:
            data = src.read([1, 2, 3])  # Read RGB bands
            image = reshape_as_image(data)
            normalized_image = normalize_image(image)
            plt.imsave(output_path, normalized_image)
        else:
            data = src.read(1)[None, :, :]  # Read first band
            image = reshape_as_image(data)
            normalized_image = normalize_image(image)
            plt.imshow(normalized_image, cmap='RdYlGn')
            plt.axis('off')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)


def add_raster_to_map(map_object: folium.Map, image_path: str,
                      bounds: Tuple[float, float, float, float], layer_name: str) -> None:
    """
    Add a raster layer to a Folium map.

    Parameters
    ----------
    map_object : folium.Map
        The Folium map object to which the raster will be added.
    image_path : str
        The file path to the image (PNG).
    bounds : tuple of float
        The bounds of the image in EPSG:4326 coordinates as (left, bottom, right, top).
    layer_name : str
        The name of the layer for the map's layer control.
    """
    raster_layer = folium.raster_layers.ImageOverlay(
        image=image_path,
        bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],  # [[south, west], [north, east]]
        name=layer_name
    )
    raster_layer.add_to(map_object)


def create_map_with_rasters(raster_path: str, title_raster: str,
                            quicklook_img: str) -> folium.Map:
    """
    Create a Folium map with two raster layers: a quicklook image and a resampled raster.

    This function takes a raster file path and a quicklook image, generates a normalized version of the raster,
    and adds both as layers to a Folium map. It also adds a layer control and measurement tool for map interaction.

    Parameters
    ----------
    raster_path : str
        The file path to the raster that will be normalized and added to the map.
    title_raster : str
        The title of the raster layer that will be displayed in the layer control.
    quicklook_img : str
        The file path to the quicklook image that will be overlaid on the map.

    Returns
    -------
    folium.Map
        A Folium map object containing the quicklook image and the resampled raster with interactive controls.
    """
    bounds = get_wgs84_bounds(raster_path)
    center_lat = (bounds[3] + bounds[1]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    m = folium.Map(location=[center_lat, center_lon], zoom_start=9)
    add_raster_to_map(m, quicklook_img, bounds, "Quicklook")
    save_normalized_image(raster_path, "tmp.png")
    add_raster_to_map(m, "tmp.png", bounds, title_raster)
    folium.LayerControl().add_to(m)
    m.add_child(MeasureControl())
    return m
