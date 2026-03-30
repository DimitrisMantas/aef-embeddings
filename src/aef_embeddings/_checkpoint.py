"""Checkpoint preparation and restoration for the aef-embeddings package."""

import hashlib
import pathlib
from enum import IntEnum
from typing import Final

import numpy as np

from aef_embeddings._types import Array1D, Embeddings

_NUM_BANDS: Final[int] = 64


class _PointStatus(IntEnum):
    """Per-point download status codes stored in the checkpoint array."""

    PENDING = 0
    COMPLETED = 1
    FAILED = 2


def _compute_request_checksum(
    ids: np.ndarray,
    xs: Array1D[np.float64],
    ys: Array1D[np.float64],
    year: int,
    region_size_pixels: int,
    crs: str,
) -> str:
    """Compute a deterministic SHA-256 fingerprint of request parameters.

    The fingerprint is used to verify that a checkpoint on disk belongs to the same
    request that is currently being executed.

    Args:
        ids:
            One-dimensional point ID array.
        xs:
            One-dimensional source-CRS x-coordinates.
        ys:
            One-dimensional source-CRS y-coordinates.
        year:
            Dataset year.
        region_size_pixels:
            Side length of the sampled region in pixels.
        crs:
            Source CRS identifier string.

    Returns:
        Hex digest of the SHA-256 hash.
    """
    h = hashlib.sha256()
    h.update(xs.tobytes())
    h.update(ys.tobytes())
    h.update(ids.tobytes())
    h.update(year.to_bytes(4, "little"))
    h.update(region_size_pixels.to_bytes(4, "little"))
    h.update(crs.encode("utf-8"))
    return h.hexdigest()


def _prepare_checkpoint_dir(
    output_dirpath: pathlib.Path,
) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    """Create the output directory and return artifact paths.

    Args:
        output_dirpath:
            Path to the output directory.

    Returns:
        A tuple of ``(output_path, status_path, request_checksum_path)``.
    """
    internal = output_dirpath / ".internal"
    internal.mkdir(parents=True, exist_ok=True)
    return (
        internal / ".output.bin",
        internal / ".status.npy",
        internal / ".request.sha256",
    )


def _initialize_or_restore_checkpoint(
    num_points: int,
    region_size_pixels: int,
    output_path: pathlib.Path,
    status_path: pathlib.Path,
    request_checksum_path: pathlib.Path,
    request_checksum: str,
) -> tuple[Embeddings, Array1D[np.uint8]]:
    """Initialize a new checkpoint or restore an existing one.

    If checkpoint files already exist on disk, they are validated against the current
    request fingerprint before restoration.

    Args:
        num_points:
            Number of query points.
        region_size_pixels:
            Side length of the sampled region in pixels.
        output_path:
            Path to the memory-mapped output file.
        status_path:
            Path to the status array file.
        request_checksum_path:
            Path to the request fingerprint file.
        request_checksum:
            Expected request fingerprint for this run.

    Returns:
        A tuple of ``(output_memmap, status_array)``.

    Raises:
        ValueError:
            If the checkpoint contains data from a different request or contains
            unidentified artifacts.
    """
    output_shape = (
        num_points,
        region_size_pixels,
        region_size_pixels,
        _NUM_BANDS,
    )

    has_checkpoint = output_path.exists() and status_path.exists()
    has_request_checksum = request_checksum_path.exists()

    if has_checkpoint and not has_request_checksum:
        raise ValueError("Checkpoint directory contains unidentified artifacts.")

    if has_request_checksum:
        stored_id = request_checksum_path.read_text().strip()
        if stored_id != request_checksum:
            raise ValueError(
                "Checkpoint directory contains data from a different request."
            )

    if has_checkpoint:
        output = np.memmap(
            output_path,
            dtype=np.float64,
            mode="r+",
            shape=output_shape,
        )
        status = np.load(status_path)
    else:
        request_checksum_path.write_text(request_checksum)
        output = np.memmap(
            output_path,
            dtype=np.float64,
            mode="w+",
            shape=output_shape,
        )
        # NaN-prefill is required for correctness.
        # The merge logic in ``_request.py`` uses ``np.isnan`` to distinguish
        # uninitialized pixels from valid zeros.
        # Without this, zero-initialized memmap entries would look like valid embedding
        # data.
        output[:] = np.nan
        status = np.zeros(num_points, dtype=np.uint8)

    return output, status
