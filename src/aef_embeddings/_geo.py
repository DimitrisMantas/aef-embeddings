"""CRS and raster utilities."""

import threading
import warnings
from typing import Final

import geopandas as gpd
import numpy as np
import pyproj

from aef_embeddings._types import Array1D

_SPATIAL_RES_METERS: Final[int] = 10
"""The spatial resolution of the dataset in meters."""

# ``Transformer`` is not thread-safe, so each thread must
# maintain its own instance.
_thread_data = threading.local()


def _compute_standard_utm_zone(lon: float) -> int:
    """Compute the standard UTM zone for a given longitude.

    Norway and Svalbard exceptions are not applied.

    Args:
        lon:
            The longitude in decimal degrees, in EPSG:4326.

    Returns:
        The corresponding UTM zone.
    """
    return int((lon + 180) / 6) % 60 + 1


def _compute_extended_utm_zone(lon: float, lat: float) -> int:
    """Compute the UTM zone for a given coordinate pair.

    Applies the Norway and Svalbard zone exceptions.

    Args:
        lon:
            The longitude in decimal degrees, in EPSG:4326.
        lat:
            The latitude in decimal degrees, in EPSG:4326.

    Returns:
        The corresponding UTM zone.
    """
    zone = _compute_standard_utm_zone(lon)

    # Norway
    if 56 <= lat < 64 and 3 <= lon < 12:
        return 32

    # Svalbard
    if 72 <= lat < 84:
        if lon < 9:
            return 31
        if lon < 21:
            return 33
        if lon < 33:
            return 35
        return 37

    return zone


def _compute_utm_crs(
    points: gpd.GeoDataFrame,
) -> Array1D[np.str_]:
    """Compute the UTM CRS EPSG identifier for each point.

    Points not already in EPSG:4326 are copied and reprojected
    before zone determination.
    The original data is not modified.

    Args:
        points:
            The query points in their original CRS.

    Returns:
        The corresponding EPSG identifiers.
    """
    points = points if points.crs.to_epsg() == 4326 else points.to_crs("EPSG:4326")

    lon = points.geometry.x.to_numpy()
    lat = points.geometry.y.to_numpy()

    utm = np.vectorize(_compute_extended_utm_zone)(lon, lat)
    return np.char.add(
        "EPSG:",
        np.where(lat >= 0, 32600 + utm, 32700 + utm).astype(np.str_),
    )


def _create_transformer(input_crs: str, output_crs: str) -> pyproj.Transformer:
    """Create a ``Transformer`` for the provided CRS pair.

    The best-accuracy transformation grid available is always used.
    Ballpark transforms are never used.

    Args:
        input_crs:
            The EPSG identifier of the input CRS.
        output_crs:
            The EPSG identifier of the output CRS.

    Returns:
        The corresponding ``Transformer``.
    """
    common_kwargs = dict(always_xy=True, allow_ballpark=False)

    try:
        return pyproj.Transformer.from_crs(
            input_crs, output_crs, only_best=True, **common_kwargs
        )
    except pyproj.exceptions.ProjError:
        warnings.warn(
            "The best-accuracy transformation grid is unavailable."
            " The next-best transformation will be used instead.",
            UserWarning,
            stacklevel=2,
        )
        return pyproj.Transformer.from_crs(
            input_crs, output_crs, only_best=False, **common_kwargs
        )


def _get_or_create_transformer(input_crs: str, output_crs: str) -> pyproj.Transformer:
    """Get or create a cached ``Transformer`` for the provided CRS pair.

    Each thread maintains its own cache.
    The best-accuracy transformation grid available is always used.
    Ballpark transforms are never used.

    Args:
        input_crs:
            The EPSG identifier of the input CRS.
        output_crs:
            The EPSG identifier of the output CRS.

    Returns:
        The corresponding ``Transformer``.
    """
    if not hasattr(_thread_data, "cache"):
        _thread_data.cache: dict[tuple[str, str], pyproj.Transformer] = {}

    key = (input_crs, output_crs)
    if key not in _thread_data.cache:
        _thread_data.cache[key] = _create_transformer(input_crs, output_crs)

    return _thread_data.cache[key]


def _compute_pixel_index(coordinate: float, resolution: float) -> int:
    """Compute the pixel index that contains a given coordinate.

    This function assumes a regular pixel grid.

    Args:
        coordinate:
            The coordinate in raster units.
        resolution:
            The spatial resolution in linear units.

    Returns:
        The corresponding pixel index.
    """
    return int(np.floor(coordinate / resolution))


def _compute_pixel_center(index: float, resolution: float) -> float:
    """Compute the center coordinate of a pixel given its index.

    This function assumes a regular pixel grid.

    Args:
        index:
            The pixel index.
        resolution:
            The spatial resolution in linear units.

    Returns:
        The corresponding coordinate.
    """
    return (index + 0.5) * resolution


def _compute_raster_origin_x(center: float, offset: int, resolution: float) -> float:
    """Compute the raster origin along the x-axis.

    This function assumes a regular pixel grid.

    Args:
        center:
            The center x-coordinate of the center pixel in
            raster units.
        offset:
            The number of pixels from the center pixel to the
            raster edge.
            This corresponds to half the raster width.
        resolution:
            The spatial resolution in linear units.

    Returns:
        The corresponding coordinate.
    """
    return center - (offset + 0.5) * resolution


def _compute_raster_origin_y(center: float, offset: int, resolution: float) -> float:
    """Compute the raster origin along the y-axis.

    This function assumes a regular pixel grid.

    Args:
        center:
            The center y-coordinate of the center pixel in
            raster units.
        offset:
            The number of pixels from the center pixel to the
            raster edge.
            This corresponds to half the raster height.
        resolution:
            The spatial resolution in linear units.

    Returns:
        The corresponding coordinate.
    """
    return center + (offset + 0.5) * resolution


def _snap_to_pixel_center(x: float, y: float) -> tuple[float, float]:
    """Snap a coordinate pair to the center of the containing pixel.

    This function assumes a regular pixel grid.

    Args:
        x:
            The easting in meters.
            The coordinate must be in UTM.
        y:
            The northing in meters.
            The coordinate must be in UTM.

    Returns:
        The snapped ``(x, y)`` coordinates.
    """
    col = _compute_pixel_index(x, _SPATIAL_RES_METERS)
    row = _compute_pixel_index(y, _SPATIAL_RES_METERS)
    return (
        _compute_pixel_center(col, _SPATIAL_RES_METERS),
        _compute_pixel_center(row, _SPATIAL_RES_METERS),
    )


def _compute_raster_half_side_length(region_size_pixels: int) -> float:
    """Compute half the side length of a query region.

    This function assumes a regular pixel grid.

    Args:
        region_size_pixels:
            The side length of the query region in pixels.

    Returns:
        The corresponding length in meters.
    """
    return region_size_pixels * _SPATIAL_RES_METERS * 0.5
