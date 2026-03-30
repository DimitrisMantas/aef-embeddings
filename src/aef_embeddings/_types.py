"""Type aliases and TypedDicts for the aef-embeddings package."""

from typing import NotRequired, TypedDict

import numpy as np

# One-dimensional array of arbitrary dtype.
# Used for point IDs, coordinates, and UTM CRS codes.
type Array1D[T: np.generic] = np.ndarray[tuple[int], np.dtype[T]]

# Four-dimensional memory-mapped embedding array of shape (N, S, S, 64).
type Embeddings = np.memmap[tuple[int, int, int, int], np.dtype[np.float64]]

# Single-point response from ``getPixels`` after structured-to-unstructured conversion,
# with shape (S, S, 64).
type _Response = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]


class _GridDimensions(TypedDict):
    """Pixel dimensions of a ``getPixels`` request grid."""

    width: int
    height: int


class _AffineTransform(TypedDict):
    """Six-parameter affine transform for a ``getPixels`` request grid."""

    scaleX: float
    shearX: float
    translateX: float
    shearY: float
    scaleY: float
    translateY: float


class _Grid(TypedDict):
    """Sampling grid specification for a ``getPixels`` request."""

    dimensions: _GridDimensions
    affineTransform: _AffineTransform
    crsCode: str


class _Request(TypedDict):
    """Complete ``getPixels`` request body."""

    fileFormat: str
    grid: _Grid
    assetId: NotRequired[str]
    bandIds: NotRequired[list[str]]


class ConflictInfo(TypedDict):
    """Metadata describing pixel-level disagreements between overlapping tiles."""

    parent_tile_id: str
    child_tile_id: str
    conflict_count: int
    # Result of ``np.argwhere`` serialized as a nested list.
    conflict_pixel_indices: list[list[int]]
    parent_values: list[float]
    child_values: list[float]
