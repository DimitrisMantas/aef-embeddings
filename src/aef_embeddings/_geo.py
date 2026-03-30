"""Geospatial utilities for the aef-embeddings package."""

import threading
import warnings
from typing import Final

import geopandas as gpd
import numpy as np
import pyproj

from aef_embeddings._types import Array1D

_SPATIAL_RES_METERS: Final[int] = 10

# ``pyproj.Transformer`` is not thread-safe, so one instance per thread is required.
# A thread-local dict maps ``(src_crs, dst_crs)`` pairs to cached transformer instances.
_transformer_local = threading.local()


def _standard_utm_zone(lon: float) -> int:
    """Return the standard UTM zone number for a longitude.

    Args:
        lon:
            Longitude in decimal degrees.

    Returns:
        UTM zone number in the range [1, 60].
    """
    return int((lon + 180) / 6) % 60 + 1


def _utm_zone(lon: float, lat: float) -> int:
    """Return the UTM zone number, including Norway and Svalbard exceptions.

    Standard UTM zone boundaries are overridden in two regions:

    - **Norway** (56-64 N, 3-12 E): forced to zone 32.
    - **Svalbard** (72-84 N): zones 31, 33, 35, or 37 depending on longitude.

    Args:
        lon:
            Longitude in decimal degrees.
        lat:
            Latitude in decimal degrees.

    Returns:
        UTM zone number accounting for special zones.
    """
    zone = _standard_utm_zone(lon)
    # Norway exception.
    if 56 <= lat < 64 and 3 <= lon < 12:
        return 32
    # Svalbard exception.
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
    """Return the UTM CRS EPSG code string for each point.

    Points are reprojected to WGS 84 if they are not already in that CRS, so that
    longitude and latitude can be used for UTM zone determination.

    Args:
        points:
            GeoDataFrame of query points in any CRS.

    Returns:
        One-dimensional array of EPSG code strings (e.g.,
        ``"EPSG:32632"``), one per point.
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
    """Create a pyproj transformer for the given CRS pair.

    The function first attempts to use the best-accuracy transformation grid.
    If the grid is unavailable (e.g., in offline or restricted environments), it falls
    back to the next-best transformation and emits a ``UserWarning``.

    Args:
        src_crs:
            Source CRS identifier string.
        dst_crs:
            Destination CRS identifier string.

    Returns:
        A ``Transformer`` configured for strict reprojection.
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
            f"Best-accuracy grid for {src_crs} -> {dst_crs} is unavailable. "
            "Falling back to the next-best transformation.",
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
    """Return a thread-local cached transformer for the given CRS pair.

    Each thread builds at most one ``Transformer`` per ``(src_crs, dst_crs)`` pair.

    Args:
        src_crs:
            Source CRS identifier string.
        dst_crs:
            Destination CRS identifier string.

    Returns:
        A thread-local ``Transformer`` configured for strict reprojection.
    """
    if not hasattr(_transformer_local, "cache"):
        _transformer_local.cache: dict[tuple[str, str], pyproj.Transformer] = {}
    key = (src_crs, dst_crs)
    if key not in _transformer_local.cache:
        _transformer_local.cache[key] = _build_proj_transformer(src_crs, dst_crs)
    return _transformer_local.cache[key]


def _pixel_index(coord: float, res: float) -> float:
    """Return the integer pixel index containing a coordinate on a regular grid.

    Args:
        coord:
            Coordinate value in the same units as *res*.
        res:
            Pixel size (spatial resolution).

    Returns:
        Floor-based pixel index.
    """
    return np.floor(coord / res)


def _pixel_center(index: float, res: float) -> float:
    """Return the center coordinate of a pixel given its index.

    Args:
        index:
            Pixel index, typically from ``_pixel_index``.
        res:
            Pixel size (spatial resolution).

    Returns:
        Center coordinate of the pixel.
    """
    return (index + 0.5) * res


def _grid_top_left_x(center_x: float, half_side: int, res: float) -> float:
    """Return the x-coordinate of the top-left corner of a pixel grid.

    The grid is centered on *center_x* with pixels of size *res*.
    The top-left corner is the origin of the affine transform where ``scaleX > 0``.

    Args:
        center_x:
            Easting of the center pixel's center.
        half_side:
            Number of pixels from the center to the edge.
        res:
            Pixel size in meters.

    Returns:
        Easting of the top-left grid corner.
    """
    return center_x - (half_side + 0.5) * res


def _grid_top_left_y(center_y: float, half_side: int, res: float) -> float:
    """Return the y-coordinate of the top-left corner of a pixel grid.

    ``scaleY`` is negative (y increases downward in raster convention), so the top-left
    corner has the largest northing.

    Args:
        center_y:
            Northing of the center pixel's center.
        half_side:
            Number of pixels from the center to the edge.
        res:
            Pixel size in meters.

    Returns:
        Northing of the top-left grid corner.
    """
    return center_y + (half_side + 0.5) * res


def _snap_to_pixel_center(x: float, y: float) -> tuple[float, float]:
    """Snap a coordinate pair to the center of the containing pixel.

    Args:
        x:
            Easting in the local UTM CRS, in meters.
        y:
            Northing in the local UTM CRS, in meters.

    Returns:
        A ``(snapped_x, snapped_y)`` tuple at the center of the intersecting pixel.
    """
    col = _pixel_index(x, _SPATIAL_RES_METERS)
    row = _pixel_index(y, _SPATIAL_RES_METERS)
    return (
        _pixel_center(col, _SPATIAL_RES_METERS),
        _pixel_center(row, _SPATIAL_RES_METERS),
    )


def _half_patch_side_m(region_size_pixels: int) -> float:
    """Return the buffer radius that produces the correct patch extent.

    This value is passed to ``ee.Geometry.Point.buffer`` so that ``.bounds`` yields a
    bounding box equal to the full patch extent.

    Args:
        region_size_pixels:
            Side length of the sampled patch in pixels.

    Returns:
        Half the patch side length in meters.
    """
    return region_size_pixels * _SPATIAL_RES_METERS * 0.5
