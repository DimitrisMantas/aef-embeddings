"""Type aliases and TypedDicts for the aef-embeddings package."""

from typing import NotRequired, TypedDict

import numpy as np

# 1-D array of arbitrary dtype (used for ids, xs, ys, utm_crs_codes).
type Array1D[T: np.generic] = np.ndarray[tuple[int], np.dtype[T]]

# 4-D memory-mapped embedding array (N, S, S, 64).
type Embeddings = np.memmap[tuple[int, int, int, int], np.dtype[np.float64]]

# Per-point response from getPixels after structured→unstructured: (S, S, 64).
type _Response = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]


class _GridDimensions(TypedDict):
    width: int
    height: int


class _AffineTransform(TypedDict):
    scaleX: float
    shearX: float
    translateX: float
    shearY: float
    scaleY: float
    translateY: float


class _Grid(TypedDict):
    dimensions: _GridDimensions
    affineTransform: _AffineTransform
    crsCode: str


class _Request(TypedDict):
    fileFormat: str
    grid: _Grid
    assetId: NotRequired[str]
    bandIds: NotRequired[list[str]]


class ConflictInfo(TypedDict):
    parent_tile_id: str
    child_tile_id: str
    conflict_count: int
    # argwhere result as nested list.
    conflict_pixel_indices: list[list[int]]
    parent_values: list[float]
    child_values: list[float]
