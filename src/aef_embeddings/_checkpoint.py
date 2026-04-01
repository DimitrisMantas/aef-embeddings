"""Checkpoint creation and restoration utilities."""

import hashlib
import pathlib
from enum import IntEnum, auto, unique
from typing import Final

import numpy as np

from aef_embeddings._types import Array1D, Embeddings

_NUM_BANDS: Final[int] = 64
"""The total number of bands in the dataset."""


@unique
class _StatusCode(IntEnum):
    """Request status codes."""

    PENDING = auto()
    SUCCESS = auto()
    FAILURE = auto()


def _compute_request_checksum(
    ids: Array1D[np.int64],
    xs: Array1D[np.float64],
    ys: Array1D[np.float64],
    year: int,
    region_size_pixels: int,
    crs: str,
) -> str:
    """Compute a deterministic SHA-256 checksum of a query.

    Used to verify that an existing checkpoint on disk matches
    the current query.

    Args:
        ids:
            The query point IDs.
        xs:
            The query point x-coordinates in their original CRS.
        ys:
            The query point y-coordinates in their original CRS.
        year:
            The dataset year.
            Valid values range from 2017 to 2025.
        region_size_pixels:
            The side length of the query region in pixels.
        crs:
            The EPSG identifier of the query CRS.

    Returns:
        The hex-encoded SHA-256 digest.
    """
    checksum = hashlib.sha256()
    for arg in (ids, xs, ys, year, region_size_pixels, crs):
        if isinstance(arg, str):
            checksum.update(arg.encode())
        elif isinstance(arg, int):
            checksum.update(str(arg).encode())
        else:
            checksum.update(arg.tobytes())
    return checksum.hexdigest()


def _maybe_create_checkpoint_directory(
    root: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Ensure the checkpoint directory exists and return its paths.

    Args:
        root:
            The parent directory for checkpoint storage.

    Returns:
        The memory map, status, and sidecar paths.
    """
    dirpath = root / ".internal"
    dirpath.mkdir(parents=True, exist_ok=True)

    return (
        dirpath / ".output.bin",
        dirpath / ".status.npy",
        dirpath / ".request.sha256",
    )


def _restore_or_initialize_checkpoint(
    num_points: int,
    region_size_pixels: int,
    output_path: pathlib.Path,
    status_path: pathlib.Path,
    checksum_path: pathlib.Path,
    expected_checksum: str,
) -> tuple[Embeddings, Array1D[np.uint8]]:
    """Restore a checkpoint from disk, or initialize a fresh one.

    Existing artifacts are validated against the expected checksum
    before they are reloaded.

    Args:
        num_points:
            The total number of query points.
        region_size_pixels:
            The side length of the query region in pixels.
        output_path:
            The memory map.
        status_path:
            The request status checklist.
        checksum_path:
            The sidecar.
        expected_checksum:
            The checksum of the current query.

    Returns:
        The memory map and the request status.

    Raises:
        ValueError:
            If the existing artifacts belong to a different
            query or cannot be identified.
    """
    output_shape = (
        num_points,
        region_size_pixels,
        region_size_pixels,
        _NUM_BANDS,
    )

    output_and_status_exist = output_path.exists() and status_path.exists()
    checksum_exists = checksum_path.exists()

    if output_and_status_exist and not checksum_exists:
        raise ValueError(
            "The specified checkpoint directory contains unidentifiable artifacts."
        )

    if checksum_exists:
        stored_id = checksum_path.read_text().strip()
        if stored_id != expected_checksum:
            raise ValueError(
                "The specified checkpoint directory contains artifacts"
                " that correspond to a different request."
            )

    if output_and_status_exist:
        output = np.memmap(
            output_path,
            dtype=np.float64,
            mode="r+",
            shape=output_shape,
        )
        status = np.load(status_path)
    else:
        checksum_path.write_text(expected_checksum)
        output = np.memmap(
            output_path,
            dtype=np.float64,
            mode="w+",
            shape=output_shape,
        )
        # NaN-prefill is required so that the merge logic can
        # distinguish uninitialized pixels from valid zeros.
        output[:] = np.nan
        status = np.full(num_points, _StatusCode.PENDING, dtype=np.uint8)

    return output, status
