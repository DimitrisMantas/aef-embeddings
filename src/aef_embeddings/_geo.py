"""Geospatial utilities for the aef-embeddings package."""

import threading
import warnings
from typing import Final

import geopandas as gpd
import numpy as np
import pyproj

from aef_embeddings._types import Array1D

_SPATIAL_RES_METERS: Final[int] = 10

# Thread-local cache for pyproj.Transformer instances.
# pyproj.Transformer is not thread-safe; one instance per thread is required.
_transformer_local = threading.local()


def _standard_utm_zone(lon: float) -> int:
    """Returns the standard UTM zone number for a longitude.

    Args:
        lon: Longitude in decimal degrees.

    Returns:
        UTM zone number in [1, 60].
    """
    return int((lon + 180) / 6) % 60 + 1


def _utm_zone(lon: float, lat: float) -> int:
    """Returns the UTM zone number, including Norway/Svalbard exceptions.

    Args:
        lon: Longitude in decimal degrees.
        lat: Latitude in decimal degrees.

    Returns:
        UTM zone number accounting for special zones in Norway and Svalbard.
    """
    zone = _standard_utm_zone(lon)
    if 56 <= lat < 64 and 3 <= lon < 12:
        return 32
    if 72 <= lat < 84:
        if lon < 9:
            return 31
        if lon < 21:
            return 33
        if lon < 33:
            return 35
        return 37
    return zone


def _resolve_utm_crss(
    points: gpd.GeoDataFrame,
) -> Array1D[np.str_]:
    """Returns the UTM CRS EPSG code string for each point.

    Args:
        points: GeoDataFrame of query points in any CRS.

    Returns:
        1-D array of EPSG code strings (e.g. ``"EPSG:32632"``), one per point.
    """
    points = points if points.crs.to_epsg() == 4326 else points.to_crs("EPSG:4326")
    lon = points.geometry.x.to_numpy()
    lat = points.geometry.y.to_numpy()
    zones = np.vectorize(_utm_zone)(lon, lat)
    return np.char.add(
        "EPSG:",
        np.where(lat >= 0, 32600 + zones, 32700 + zones).astype(np.str_),
    )


def _build_proj_transformer(src_crs: str, dst_crs: str) -> pyproj.Transformer:
    """Creates a new pyproj Transformer for the given CRS pair.

    Attempts to use the best-accuracy transformation grid. Falls back to
    next-best with a UserWarning when the best grid is unavailable (e.g.
    offline or restricted environments).

    Args:
        src_crs: Source CRS identifier string.
        dst_crs: Destination CRS identifier string.

    Returns:
        A Transformer configured for strict reprojection.
    """
    try:
        return pyproj.Transformer.from_crs(
            src_crs,
            dst_crs,
            always_xy=True,
            allow_ballpark=False,
            only_best=True,
        )
    except pyproj.exceptions.ProjError:
        warnings.warn(
            f"Best-accuracy grid for {src_crs}→{dst_crs} unavailable."
            " Falling back to next-best transformation.",
            UserWarning,
            stacklevel=2,
        )
        return pyproj.Transformer.from_crs(
            src_crs,
            dst_crs,
            always_xy=True,
            allow_ballpark=False,
            only_best=False,
        )


def _get_transformer(src_crs: str, dst_crs: str) -> pyproj.Transformer:
    """Returns a thread-local cached Transformer for the given CRS pair.

    pyproj.Transformer is not thread-safe; one instance per thread is
    required. This function ensures each thread builds at most one
    Transformer per (src_crs, dst_crs) pair.

    Args:
        src_crs: Source CRS identifier string.
        dst_crs: Destination CRS identifier string.

    Returns:
        A thread-local Transformer configured for strict reprojection.
    """
    if not hasattr(_transformer_local, "cache"):
        _transformer_local.cache: dict[tuple[str, str], pyproj.Transformer] = {}
    key = (src_crs, dst_crs)
    if key not in _transformer_local.cache:
        _transformer_local.cache[key] = _build_proj_transformer(src_crs, dst_crs)
    return _transformer_local.cache[key]


def _pixel_index(coord: float, res: float) -> float:
    """Returns the integer pixel index containing coord on a regular grid.

    Args:
        coord: Coordinate value in the same units as res.
        res: Pixel size (spatial resolution).

    Returns:
        Floor-based pixel index.
    """
    return np.floor(coord / res)


def _pixel_center(index: float, res: float) -> float:
    """Returns the center coordinate of a pixel given its index.

    Args:
        index: Pixel index (typically from _pixel_index).
        res: Pixel size.

    Returns:
        Center coordinate of the pixel.
    """
    return (index + 0.5) * res


def _grid_top_left_x(center_x: float, half_side: int, res: float) -> float:
    """Returns the x-coordinate of the top-left corner of a pixel grid.

    The grid is centred on center_x with pixels of size res. The top-left
    corner is the origin of the affine transform (scaleX > 0).

    Args:
        center_x: Easting of the centre pixel's centre.
        half_side: Number of pixels from the centre to the edge
            (buffer // 2).
        res: Pixel size in meters.

    Returns:
        Easting of the top-left grid corner.
    """
    return center_x - (half_side + 0.5) * res


def _grid_top_left_y(center_y: float, half_side: int, res: float) -> float:
    """Returns the y-coordinate of the top-left corner of a pixel grid.

    scaleY is negative (Y increases downward in raster convention), so the
    top-left corner has the largest northing.

    Args:
        center_y: Northing of the centre pixel's centre.
        half_side: Number of pixels from the centre to the edge.
        res: Pixel size in meters.

    Returns:
        Northing of the top-left grid corner.
    """
    return center_y + (half_side + 0.5) * res


def _snap_to_pixel_center(x: float, y: float) -> tuple[float, float]:
    """Snaps a coordinate pair to the centre of the pixel it falls within.

    Args:
        x: Easting in the local UTM CRS (meters).
        y: Northing in the local UTM CRS (meters).

    Returns:
        (snapped_x, snapped_y) at the centre of the intersecting pixel.
    """
    col = _pixel_index(x, _SPATIAL_RES_METERS)
    row = _pixel_index(y, _SPATIAL_RES_METERS)
    return (
        _pixel_center(col, _SPATIAL_RES_METERS),
        _pixel_center(row, _SPATIAL_RES_METERS),
    )


def _half_patch_side_m(region_size_pixels: int) -> float:
    """Returns the buffer radius that produces the correct patch extent.

    Computes half the square patch's side length in meters. This value is
    passed to ``ee.Geometry.Point.buffer()`` so that ``.bounds()`` yields a
    bounding box equal to the full patch extent.

    Args:
        region_size_pixels: Side length of the sampled patch in pixels.

    Returns:
        Half the patch side length in meters.
    """
    return region_size_pixels * _SPATIAL_RES_METERS * 0.5
