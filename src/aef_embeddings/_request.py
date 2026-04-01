"""Pixel data fetching and tile merging."""

from typing import Any, Final, cast

import ee
import numpy as np
import numpy.lib.recfunctions as rfn

from aef_embeddings._geo import (
    _SPATIAL_RES_METERS,
    _compute_raster_origin_x,
    _compute_raster_origin_y,
)
from aef_embeddings._types import _AffineTransform, _Grid, _Request, _Response

_FLOAT_NODATA: Final[float] = -((128 / 127.5) ** 2)
"""The floating-point no-data sentinel in the dataset."""

_BAND_NAMES: Final[list[str]] = [f"A{i:02d}" for i in range(64)]
"""The band names in the dataset."""


def _build_pixel_request(
    x: float,
    y: float,
    region_size_pixels: int,
    utm_crs: str,
) -> _Request:
    """Build a pixel data request for a query region.

    The request does not include a tile identifier.
    The caller must set one before dispatching.

    Args:
        x:
            Snapped easting of the center pixel in meters.
            The coordinate is in UTM.
        y:
            Snapped northing of the center pixel in meters.
            The coordinate is in UTM.
        region_size_pixels:
            Side length of the query region in pixels.
        utm_crs:
            The local UTM CRS identifier.

    Returns:
        A request dict ready to be dispatched after setting the
        tile identifier.
    """
    half_side_px = region_size_pixels // 2
    affine: _AffineTransform = {
        "scaleX": float(_SPATIAL_RES_METERS),
        "shearX": 0.0,
        "translateX": _compute_raster_origin_x(x, half_side_px, _SPATIAL_RES_METERS),
        "shearY": 0.0,
        "scaleY": float(-_SPATIAL_RES_METERS),
        "translateY": _compute_raster_origin_y(y, half_side_px, _SPATIAL_RES_METERS),
    }
    grid: _Grid = {
        "dimensions": {
            "width": region_size_pixels,
            "height": region_size_pixels,
        },
        "affineTransform": affine,
        "crsCode": utm_crs,
    }
    return {
        "fileFormat": "NUMPY_NDARRAY",
        "grid": grid,
        "bandIds": _BAND_NAMES,
    }


def _fetch_pixels(request: _Request) -> _Response:
    """Fetch embedding pixels for a single tile.

    The structured per-band result is converted to a plain float64
    array of shape ``(S, S, 64)``.

    Args:
        request:
            A complete request dict including the tile identifier.

    Returns:
        Embedding array of shape ``(S, S, 64)`` for the query
        region.

    Raises:
        ValueError:
            If the result is an unstructured array instead of the
            expected structured per-band array.
    """
    structured = ee.data.getPixels(cast(dict[str, Any], request))
    band_names = structured.dtype.names
    if band_names is None:
        raise ValueError(
            "Expected a structured per-band array but received an unstructured array."
        )
    return rfn.structured_to_unstructured(structured[[*band_names]])


def _find_tile_conflicts(
    merged: _Response,
    incoming: _Response,
) -> np.ndarray[tuple[int, ...], np.dtype[np.bool_]] | None:
    """Return a boolean mask of pixels that are valid in both arrays but differ.

    A conflict occurs when a pixel is finite and non-no-data in
    both arrays but the two values disagree.

    Args:
        merged:
            The accumulated pixel array of shape ``(S, S, 64)``.
        incoming:
            The new tile pixel array of shape ``(S, S, 64)``.

    Returns:
        Boolean mask of conflicting elements, or ``None`` if there
        are no conflicts.
    """

    def _is_invalid(
        arr: _Response,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.bool_]]:
        return np.isinf(arr) | np.isnan(arr) | np.isclose(arr, _FLOAT_NODATA)

    both_valid = ~_is_invalid(merged) & ~_is_invalid(incoming)
    if not np.any(both_valid):
        return None
    differs = ~np.isclose(merged, incoming, equal_nan=True)
    mask = both_valid & differs
    return mask if np.any(mask) else None


def _merge_tile_pixels(
    merged: _Response,
    incoming: _Response,
) -> None:
    """Fill invalid pixels in the accumulated array from the incoming array.

    This is a pure gap-fill operation.
    Call ``_find_tile_conflicts`` first if conflict detection is
    needed.

    Args:
        merged:
            The accumulated pixel array, modified in place.
        incoming:
            The new tile pixel array.
    """
    incoming_valid = ~(
        np.isinf(incoming) | np.isnan(incoming) | np.isclose(incoming, _FLOAT_NODATA)
    )
    merged_invalid = (
        np.isinf(merged) | np.isnan(merged) | np.isclose(merged, _FLOAT_NODATA)
    )
    fill = incoming_valid & merged_invalid
    merged[fill] = incoming[fill]
