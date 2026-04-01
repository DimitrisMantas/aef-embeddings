"""Type aliases and TypedDicts."""

from typing import NotRequired, TypedDict

import numpy as np

# One-dimensional array of arbitrary dtype.
# Used for query point IDs, coordinates, and UTM CRS codes.
type Array1D[T: np.generic] = np.ndarray[tuple[int], np.dtype[T]]

# Four-dimensional embedding memory map of shape (N, S, S, 64).
type Embeddings = np.memmap[tuple[int, int, int, int], np.dtype[np.float64]]


# Single-point embedding pixel array of shape (S, S, 64).
type _Response = np.ndarray[tuple[int, int, int], np.dtype[np.float64]]


class _GridDimensions(TypedDict):
    """Pixel dimensions of a sampling grid."""

    width: int
    height: int


class _AffineTransform(TypedDict):
    """Six-parameter affine transform of a sampling grid."""

    scaleX: float
    shearX: float
    translateX: float
    shearY: float
    scaleY: float
    translateY: float


class _Grid(TypedDict):
    """Sampling grid specification for a pixel data request."""

    dimensions: _GridDimensions
    affineTransform: _AffineTransform
    crsCode: str


class _Request(TypedDict):
    """Pixel data request body."""

    fileFormat: str
    grid: _Grid
    assetId: NotRequired[str]
    bandIds: NotRequired[list[str]]
