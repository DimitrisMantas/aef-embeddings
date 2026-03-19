"""GEE request building, fetching, and response merging."""

from typing import Any, Final, cast

import ee
import numpy as np
import numpy.lib.recfunctions as rfn

from aef_embeddings._geo import (
    _SPATIAL_RES_METERS,
    _grid_top_left_x,
    _grid_top_left_y,
)
from aef_embeddings._types import _AffineTransform, _Grid, _Request, _Response

_NODATA_VALUE_F64: Final[float] = -((128 / 127.5) ** 2)
_BAND_IDS: Final[list[str]] = [f"A{i:02d}" for i in range(64)]


def _build_base_request(
    x: float,
    y: float,
    region_size_pixels: int,
    utm_crs: str,
) -> _Request:
    """Builds a complete GEE getPixels request body, minus assetId.

    Args:
        x: Snapped easting of the centre pixel (meters, UTM).
        y: Snapped northing of the centre pixel (meters, UTM).
        region_size_pixels: Side length of the requested patch in pixels.
        utm_crs: EPSG code string for the local UTM CRS.

    Returns:
        A _Request dict ready for ``assetId`` to be set before dispatch.
    """
    half_side_px = region_size_pixels // 2
    affine: _AffineTransform = {
        "scaleX": float(_SPATIAL_RES_METERS),
        "shearX": 0.0,
        "translateX": _grid_top_left_x(x, half_side_px, _SPATIAL_RES_METERS),
        "shearY": 0.0,
        "scaleY": float(-_SPATIAL_RES_METERS),
        "translateY": _grid_top_left_y(y, half_side_px, _SPATIAL_RES_METERS),
    }
    grid: _Grid = {
        "dimensions": {
            "width": region_size_pixels,
            "height": region_size_pixels,
        },
        "affineTransform": affine,
        "crsCode": utm_crs,
    }
    request: _Request = {
        "fileFormat": "NUMPY_NDARRAY",
        "grid": grid,
        "bandIds": _BAND_IDS,
    }
    return request


def _fetch_response(request: _Request) -> _Response:
    """Fetches pixel data from GEE and converts to an unstructured array.

    Args:
        request: A complete _Request dict including ``assetId``.

    Returns:
        Float64 array of shape (S, S, 64) for the requested patch.

    Raises:
        ValueError: If GEE returns an unstructured array instead of a
            structured NUMPY_NDARRAY.
    """
    response = ee.data.getPixels(cast(dict[str, Any], request))
    band_names = response.dtype.names
    if band_names is None:
        raise ValueError(
            "Expected a structured array from ee.data.getPixels, got an"
            " unstructured array. Verify the fileFormat is NUMPY_NDARRAY."
        )
    return rfn.structured_to_unstructured(response[[*band_names]])


def _find_response_conflicts(
    parent_response: _Response,
    child_response: _Response,
) -> np.ndarray[tuple[int, ...], np.dtype[np.bool_]] | None:
    """Returns a boolean mask of pixels valid in both responses but differing.

    A conflict occurs when a pixel is valid (finite, non-nodata) in both
    the parent and child response but the two values disagree.

    Args:
        parent_response: Accumulated response array of shape (S, S, 64).
        child_response: New tile response array of shape (S, S, 64).

    Returns:
        Boolean mask of conflicting elements, or None if there are no
        conflicts.
    """

    def _is_invalid(
        arr: _Response,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.bool_]]:
        return np.isinf(arr) | np.isnan(arr) | np.isclose(arr, _NODATA_VALUE_F64)

    both_are_valid = ~_is_invalid(parent_response) & ~_is_invalid(child_response)
    if not np.any(both_are_valid):
        return None
    differs = ~np.isclose(parent_response, child_response, equal_nan=True)
    mask = both_are_valid & differs
    return mask if np.any(mask) else None


def _merge_child_into_parent_response(
    parent_response: _Response,
    child_response: _Response,
) -> None:
    """Fills invalid parent pixels from valid child pixels (pure merge).

    Does not check for conflicts; call ``_find_response_conflicts`` first
    if conflict detection is needed.

    Args:
        parent_response: Accumulated response array, modified in place.
        child_response: New tile response array.
    """
    child_is_invalid = (
        np.isinf(child_response)
        | np.isnan(child_response)
        | np.isclose(child_response, _NODATA_VALUE_F64)
    )
    child_is_valid = ~child_is_invalid

    parent_is_invalid = (
        np.isinf(parent_response)
        | np.isnan(parent_response)
        | np.isclose(parent_response, _NODATA_VALUE_F64)
    )

    to_replace = child_is_valid & parent_is_invalid
    parent_response[to_replace] = child_response[to_replace]
