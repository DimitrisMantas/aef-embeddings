"""Google Earth Engine client for AlphaEarth Foundation embeddings."""

import concurrent.futures
import datetime
import hashlib
import os
import pathlib
import warnings
from typing import Any, Final

import ee
import geopandas as gpd
import google.api_core.retry
import h5py
import loguru
import numpy as np
import tqdm

from aef_embeddings._checkpoint import (
    _compute_request_checksum,
    _maybe_create_checkpoint_directory,
    _restore_or_initialize_checkpoint,
    _StatusCode,
)
from aef_embeddings._geo import (
    _SPATIAL_RES_METERS,
    _compute_raster_half_side_length,
    _compute_utm_crs,
    _get_or_create_transformer,
    _snap_to_pixel_center,
)
from aef_embeddings._logging import (
    _configure_logging,
    _PointLog,
    _redirect_warnings_to_tqdm,
    _write_point_log,
)
from aef_embeddings._request import (
    _FLOAT_NODATA,
    _build_pixel_request,
    _fetch_pixels,
    _find_tile_conflicts,
    _merge_tile_pixels,
)
from aef_embeddings._types import Array1D, _Request, _Response

_YEAR_MIN: Final[int] = 2017
_YEAR_MAX: Final[int] = 2025
_MAX_GEE_WORKERS: Final[int] = 40


class AEFEmbeddingStore:
    """Client for the AlphaEarth Foundation Embedding dataset.

    This class provides methods for downloading, quantizing, and
    pooling 64-band embeddings at 10 m spatial resolution.

    Use the ``create`` class method to authenticate and initialize a
    session in one step.

    See Also:
        - [Dataset catalog](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)
        - [GCS distribution](https://developers.google.com/earth-engine/guides/aef_on_gcs_readme)
    """

    _DATASET_NAME: Final[str] = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

    def __init__(
        self,
        *,
        use_high_volume_endpoint: bool = True,
    ) -> None:
        """Construct a store bound to an already-initialized session.

        Prefer ``AEFEmbeddingStore.create`` for standard usage.
        Use this constructor directly only when testing without live
        credentials.

        Args:
            use_high_volume_endpoint:
                Whether the session was initialized with the
                high-volume endpoint.
                Stored for informational purposes only; the endpoint
                is set during ``ee.Initialize``.

        See Also:
            [High-volume endpoint](https://developers.google.com/earth-engine/guides/processing_environments#high-volume_endpoint)
        """
        self._use_high_volume_endpoint = use_high_volume_endpoint
        self._dataset = ee.ImageCollection(self._DATASET_NAME)

    @classmethod
    def create(
        cls,
        project_id: str | None = None,
        *,
        use_high_volume_endpoint: bool = True,
    ) -> "AEFEmbeddingStore":
        """Authenticate, initialize a session, and return a new store.

        This is the standard entry point for interactive and
        production use.

        Args:
            project_id:
                GEE project ID string.
                If ``None``, the default project configured in the
                Earth Engine credentials is used.
            use_high_volume_endpoint:
                Whether to use the high-volume endpoint for server
                requests.

        Returns:
            A new ``AEFEmbeddingStore`` bound to the initialized
            session.

        See Also:
            [High-volume endpoint](https://developers.google.com/earth-engine/guides/processing_environments#high-volume_endpoint)
        """
        ee.Authenticate()
        ee.Initialize(
            project=project_id,
            opt_url=(
                "https://earthengine-highvolume.googleapis.com"
                if use_high_volume_endpoint
                else None
            ),
        )
        return cls(use_high_volume_endpoint=use_high_volume_endpoint)

    @staticmethod
    def quantize(
        values: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.int8]]:
        """Quantize a float64 embedding array to int8.

        Applies the signed square-root quantization scheme used for
        the GCS distribution of the dataset [1]_.
        The input array must not contain NaN or infinity values.

        Args:
            values:
                Float64 array of any shape.

        Returns:
            Int8 array with the same shape as *values*.

        References:
            .. [1] Google, "AlphaEarth Foundation Satellite Embeddings on Google Cloud
                Storage," *Google Earth Engine Guides*, 2025. [Online].
                Available: https://developers.google.com/earth-engine/guides/aef_on_gcs_readme#de-quantization
        """
        # sqrt(|x|) * sign(x) * 127.5, clamped to [-127, 127].
        return np.clip(
            np.sqrt(np.abs(values)) * 127.5 * np.sign(values),
            a_min=-127,
            a_max=127,
        ).astype(np.int8)

    @staticmethod
    def dequantize(
        values: np.ndarray[tuple[int, ...], np.dtype[np.int8]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Dequantize an int8 embedding array back to float64.

        Applies the inverse of the signed square-root quantization
        scheme used for the GCS distribution of the dataset [1]_.
        The input array must not contain NaN or infinity values.

        Args:
            values:
                Int8 array of any shape.

        Returns:
            Float64 array with the same shape as *values*.

        References:
            .. [1] Google, "AlphaEarth Foundation Satellite Embeddings on Google Cloud
                Storage," *Google Earth Engine Guides*, 2025. [Online].
                Available: https://developers.google.com/earth-engine/guides/aef_on_gcs_readme#de-quantization
        """
        data = values.astype(np.float64)
        # (x / 127.5)^2 * sign(x).
        return ((data / 127.5) ** 2) * np.sign(data)

    @staticmethod
    def gem_pool(
        values: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        p: float = 3.0,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Pool spatial dimensions using Generalized Mean (GeM) pooling.

        GeM interpolates between average pooling (``p = 1``) and max
        pooling (``p -> inf``) via a tunable power parameter [1]_.
        NaN values are automatically excluded from the mean (masked
        pooling via ``np.nanmean``).

        For input of shape ``(N, S, S, D)``, returns ``(N, D)``.
        For input of shape ``(S, S, D)``, returns ``(D,)``.

        Args:
            values:
                Float64 array with at least three dimensions.
                The last two spatial dimensions precede the band
                dimension.
            p:
                Power parameter.
                Must be positive.

        Returns:
            Array with spatial dimensions collapsed to shape
            ``(..., D)``.

        Raises:
            ValueError:
                If *p* is not positive.

        References:
            .. [1] F. Radenovic, G. Tolias, and O. Chum, "Fine-tuning CNN image
                retrieval with no human annotation,"
                *IEEE Trans. Pattern Anal. Mach. Intell.*,
                vol. 41, no. 7, pp. 1655-1668, Jul. 2019,
                doi: `10.1109/TPAMI.2018.2846566 <https://doi.org/10.1109/TPAMI.2018.2846566>`_.

        See Also:
            - I. Corley, C. Robinson, I. Becker-Reshef, and J. M. Lavista Ferres,
                "From pixels to patches: Pooling strategies for Earth embeddings,"
                in *Proc. ICLR Workshop Mach. Learn. Remote Sens. (ML4RS)*, 2026.
                `[GitHub] <https://github.com/isaaccorley/geopool>`_
        """
        if p <= 0:
            raise ValueError(f"p must be positive, got {p}.")
        # Raise each element to the p-th power, preserving sign.
        powered = np.sign(values) * np.abs(values) ** p
        # Average over the two spatial dimensions.
        pooled = np.nanmean(powered, axis=(-3, -2))
        # Invert the power to recover the generalized mean.
        return np.sign(pooled) * np.abs(pooled) ** (1.0 / p)

    @staticmethod
    def stat_pool(
        values: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Pool spatial dimensions using first- and second-order statistics.

        Concatenates per-band mean, standard deviation, minimum, and
        maximum over the spatial dimensions, producing a descriptor
        four times the original band count [1]_.
        NaN values are automatically excluded via ``np.nan*``
        functions.

        For input of shape ``(N, S, S, D)``, returns ``(N, 4*D)``.
        For input of shape ``(S, S, D)``, returns ``(4*D,)``.

        Args:
            values:
                Float64 array with at least three dimensions.
                The last two spatial dimensions precede the band
                dimension.

        Returns:
            Array with spatial dimensions collapsed and band dimension
            expanded to shape ``(..., 4*D)``.

        References:
            .. [1] I. Corley, C. Robinson, I. Becker-Reshef, and J. M. Lavista Ferres,
                "From pixels to patches: Pooling strategies for Earth embeddings,"
                in *Proc. ICLR Workshop Mach. Learn. Remote Sens. (ML4RS)*, 2026.
                `[GitHub] <https://github.com/isaaccorley/geopool>`_
        """
        axes = (-3, -2)
        return np.concatenate(
            [
                np.nanmean(values, axis=axes),
                np.nanstd(values, axis=axes),
                np.nanmin(values, axis=axes),
                np.nanmax(values, axis=axes),
            ],
            axis=-1,
        )

    def sample_region(
        self,
        points: gpd.GeoDataFrame,
        point_id_column: str | None,
        region_size_pixels: int,
        year: int,
        max_workers: int | None = None,
        output_dirpath: str | os.PathLike[str] = "data",
        checkpoint_period_points: int = 5000,
        debug: bool = False,
    ) -> pathlib.Path:
        """Sample a square region of embeddings around each query point.

        Each point is reprojected to its local UTM zone, snapped to
        the nearest pixel center, and a square region of
        ``region_size_pixels`` x ``region_size_pixels`` is sampled
        around it.
        When a region spans multiple tiles, the rasters are merged
        with conflict detection.

        Downloads are checkpointed periodically so that interrupted
        jobs can be resumed without re-downloading completed points.

        Args:
            points:
                GeoDataFrame of query points in any CRS.
            point_id_column:
                Name of the column containing query point IDs.
                Pass ``None`` to use the DataFrame index.
            region_size_pixels:
                Side length of the query region in pixels.
                Must be a positive odd integer (1, 3, 5, ...).
                Each pixel covers 10 m.
            year:
                Dataset year.
                Must be between 2017 and 2025 inclusive.
            max_workers:
                Maximum number of worker threads for parallel
                requests.
                Cannot exceed 40 (GEE quota).
                Defaults to the ``ThreadPoolExecutor`` default.
            output_dirpath:
                Directory for output files and checkpoint artifacts.
            checkpoint_period_points:
                Number of query points between checkpoint saves.
                If larger than the total number of query points, a
                single checkpoint is saved on completion.
            debug:
                If ``True``, show all log levels on the console and
                force single-threaded execution for deterministic
                ordering.

        Returns:
            Path to the HDF5 output file containing datasets
            ``values``, ``ids``, ``x``, ``y``, and ``status``.

        Raises:
            ValueError:
                If *year* is outside [2017, 2025], if *max_workers*
                exceeds 40, or if *region_size_pixels* is not a
                positive odd integer.

        See Also:
            [Adjustable quota limits](https://developers.google.com/earth-engine/guides/usage#adjustable_quota_limits)
        """
        if not (_YEAR_MIN <= year <= _YEAR_MAX):
            raise ValueError(
                f"year must be between {_YEAR_MIN} and {_YEAR_MAX}, got {year}."
            )
        if max_workers is not None and max_workers > _MAX_GEE_WORKERS:
            raise ValueError(
                f"max_workers cannot exceed {_MAX_GEE_WORKERS}"
                f" without special project configuration, got {max_workers}."
            )
        if region_size_pixels < 1 or region_size_pixels % 2 == 0:
            raise ValueError(
                f"region_size_pixels must be a positive odd integer,"
                f" got {region_size_pixels}."
            )

        ids, xs, ys, utm_crs_codes = _get_point_info(points, point_id_column)

        source_crs = points.crs.to_string()
        request_checksum = _compute_request_checksum(
            ids, xs, ys, year, region_size_pixels, source_crs
        )

        output_dirpath = pathlib.Path(output_dirpath)
        memmap_path, status_path, request_checksum_path = (
            _maybe_create_checkpoint_directory(output_dirpath)
        )

        output, status = _restore_or_initialize_checkpoint(
            len(points),
            region_size_pixels,
            memmap_path,
            status_path,
            request_checksum_path,
            request_checksum,
        )
        output_path = output_dirpath / "embeddings.h5"

        # Initialize per-point log entries.
        logs: list[_PointLog] = [
            _PointLog(
                point_index=i,
                source_x=float(xs[i]),
                source_y=float(ys[i]),
                source_crs=source_crs,
                utm_crs=str(utm_crs_codes[i]),
                year=year,
                region_size_pixels=region_size_pixels,
            )
            for i in range(len(points))
        ]
        for i in range(len(points)):
            if status[i] == _StatusCode.SUCCESS:
                logs[i].mark_restored()

        remaining = np.where(status != _StatusCode.SUCCESS)[0]
        if len(remaining) > 0:
            _configure_logging(console=debug)

            if debug:
                if max_workers is not None and max_workers > 1:
                    warnings.warn(
                        "Debug mode forces single-threaded execution"
                        " for deterministic log ordering."
                        " The max_workers setting will be overridden to 1.",
                        UserWarning,
                        stacklevel=2,
                    )
                max_workers = 1

            with (
                _redirect_warnings_to_tqdm(),
                concurrent.futures.ThreadPoolExecutor(max_workers) as executor,
            ):
                futures = {
                    executor.submit(
                        self._sample_point_region,
                        xs[point_index],
                        ys[point_index],
                        source_crs,
                        # GEE and PROJ do not accept NumPy string types.
                        str(utm_crs_codes[point_index]),
                        region_size_pixels,
                        year,
                    ): point_index
                    for point_index in remaining
                }

                progress = tqdm.tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                )
                for i, future in enumerate(progress, 1):
                    point_index = futures[future]
                    try:
                        pixels, events = future.result()
                        output[point_index] = pixels
                        status[point_index] = _StatusCode.SUCCESS
                        logs[point_index].record_success(events)
                    except Exception as e:
                        loguru.logger.warning(
                            f"Download failed for point {point_index}"
                            f" ({type(e).__name__}): {e}"
                        )
                        output[point_index] = np.nan
                        status[point_index] = _StatusCode.FAILURE
                        logs[point_index].record_failure(e)

                    if i % checkpoint_period_points == 0:
                        output.flush()
                        np.save(status_path, status)
                        _write_point_log(
                            [log.to_dict() for log in logs], output_dirpath
                        )

            output.flush()
            np.save(status_path, status)

        if len(status) > 0 and np.all(status == _StatusCode.FAILURE):
            warnings.warn(
                "All points failed to download."
                " The output file contains no valid data.",
                UserWarning,
                stacklevel=2,
            )

        _write_point_log([log.to_dict() for log in logs], output_dirpath)
        return self._write_hdf5(
            output_path,
            output,
            ids,
            xs,
            ys,
            status,
            source_crs,
            year,
            region_size_pixels,
        )

    @staticmethod
    def _write_hdf5(
        output_path: pathlib.Path,
        output: np.ndarray[Any, np.dtype[np.float64]],
        ids: np.ndarray[tuple[int], Any],
        xs: Array1D[np.float64],
        ys: Array1D[np.float64],
        status: Array1D[np.uint8],
        source_crs: str,
        year: int,
        region_size_pixels: int,
    ) -> pathlib.Path:
        """Write a self-describing HDF5 file with all outputs.

        The file contains five datasets (``values``, ``ids``, ``x``,
        ``y``, ``status``) and five file-level attributes (``crs``,
        ``year``, ``region_size_pixels``, ``spatial_res_meters``,
        ``created_at``).
        A companion ``.sha256`` sidecar is written alongside it.

        Args:
            output_path:
                Destination file path.
            output:
                The embedding values.
            ids:
                The query point IDs.
            xs:
                The query point x-coordinates in source CRS.
            ys:
                The query point y-coordinates in source CRS.
            status:
                The request status.
            source_crs:
                The source CRS identifier.
            year:
                Dataset year.
            region_size_pixels:
                The query region side length in pixels.

        Returns:
            The output path.
        """
        with h5py.File(output_path, "w") as f:
            f.create_dataset("values", data=output)
            f.create_dataset("ids", data=ids)
            f.create_dataset("x", data=xs)
            f.create_dataset("y", data=ys)
            f.create_dataset("status", data=status)
            f.attrs["crs"] = source_crs
            f.attrs["year"] = year
            f.attrs["region_size_pixels"] = region_size_pixels
            f.attrs["spatial_res_meters"] = _SPATIAL_RES_METERS
            f.attrs["created_at"] = datetime.datetime.now(datetime.UTC).isoformat()

        # Compute a SHA-256 checksum for integrity verification.
        sha256 = hashlib.sha256()
        with open(output_path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                sha256.update(chunk)
        checksum_path = output_path.parent / (output_path.name + ".sha256")
        checksum_path.write_text(f"{sha256.hexdigest()}  {output_path.name}\n")

        return output_path

    def _get_intersecting_tile_ids(
        self,
        x: float,
        y: float,
        utm_crs: str,
        region_size_pixels: int,
        year: int,
    ) -> list[str]:
        """Return tile identifiers that intersect the query region.

        Args:
            x:
                Snapped easting of the center pixel in meters.
                The coordinate is in UTM.
            y:
                Snapped northing of the center pixel in meters.
                The coordinate is in UTM.
            utm_crs:
                The local UTM CRS identifier.
            region_size_pixels:
                Side length of the query region in pixels.
            year:
                Dataset year.

        Returns:
            List of tile identifiers for intersecting tiles.
        """
        point = ee.Geometry.Point((x, y), proj=utm_crs)
        # For single-pixel queries there is no need to buffer.
        bounds = (
            point
            if region_size_pixels == 1
            else point.buffer(
                _compute_raster_half_side_length(region_size_pixels)
            ).bounds()
        )

        result: list[str] | None = (
            self._dataset.filterBounds(bounds)
            .filterDate(f"{year}-01-01", f"{year + 1}-01-01")
            .aggregate_array("system:id")
            .getInfo()
        )
        return result if result is not None else []

    @google.api_core.retry.Retry()
    def _sample_point_region(
        self,
        source_x: float,
        source_y: float,
        source_crs: str,
        utm_crs: str,
        region_size_pixels: int,
        year: int,
    ) -> tuple[_Response, dict[str, Any]]:
        """Sample a region around a single query point.

        The query point is reprojected from the source CRS to its
        local UTM zone, snapped to the nearest pixel center, and the
        requested region is fetched.
        If the region spans multiple tiles, the rasters are merged.

        Args:
            source_x:
                The query point x-coordinate in source CRS.
            source_y:
                The query point y-coordinate in source CRS.
            source_crs:
                The source CRS identifier.
            utm_crs:
                The local UTM CRS identifier.
            region_size_pixels:
                Side length of the query region in pixels.
            year:
                Dataset year.

        Returns:
            The merged pixel array and the accumulated log events.

        Raises:
            RuntimeError:
                If the query point does not intersect any tiles in
                the dataset.
        """
        utm_x, utm_y = _get_or_create_transformer(source_crs, utm_crs).transform(
            source_x, source_y
        )
        snapped_x, snapped_y = _snap_to_pixel_center(utm_x, utm_y)

        request = _build_pixel_request(
            snapped_x, snapped_y, region_size_pixels, utm_crs
        )
        tile_ids = self._get_intersecting_tile_ids(
            snapped_x, snapped_y, utm_crs, region_size_pixels, year
        )
        if not tile_ids:
            raise RuntimeError(
                "No tiles found for this point. The location may be outside"
                f" the spatial coverage of the dataset for {year}."
            )

        merged, conflicts = _fetch_and_merge_tiles(request, tile_ids)

        band0 = merged[..., 0]
        valid_pixels = int(
            np.sum(
                ~(np.isnan(band0) | np.isinf(band0) | np.isclose(band0, _FLOAT_NODATA))
            )
        )
        total_pixels = merged.shape[0] * merged.shape[1]
        events = {
            "utm_easting": utm_x,
            "utm_northing": utm_y,
            "snapped_easting": snapped_x,
            "snapped_northing": snapped_y,
            "tiles": tile_ids,
            "conflicts": conflicts,
            "valid_pixels": valid_pixels,
            "missing_pixels": total_pixels - valid_pixels,
            "checksum_md5": hashlib.md5(
                np.ascontiguousarray(merged).tobytes()
            ).hexdigest(),
        }
        return merged, events


def _fetch_and_merge_tiles(
    request: _Request,
    tile_ids: list[str],
) -> tuple[_Response, list[dict[str, Any]]]:
    """Fetch pixel data from all tiles and merge into a single array.

    The first tile becomes the base raster.
    Subsequent tiles fill gaps via gap-fill merging.
    Conflicts between overlapping valid pixels are recorded but
    resolved in favor of the base tile.

    Args:
        request:
            A pixel data request without a tile identifier set.
        tile_ids:
            The tile identifiers to fetch.

    Returns:
        The merged pixel array and a list of conflict records.
    """
    request = {**request}
    conflicts: list[dict[str, Any]] = []

    base_tile_id = tile_ids[0]
    request["assetId"] = base_tile_id
    merged = _fetch_pixels(request)

    for tile_id in tile_ids[1:]:
        request["assetId"] = tile_id
        tile_pixels = _fetch_pixels(request)

        conflict_mask = _find_tile_conflicts(merged, tile_pixels)
        if conflict_mask is not None:
            conflict_count = int(conflict_mask[..., 0].sum())
            conflicts.append(
                {
                    "tiles": [base_tile_id, tile_id],
                    "pixel_count": conflict_count,
                    "pixel_indices": np.argwhere(conflict_mask[..., 0]).tolist(),
                }
            )
            loguru.logger.warning(
                f"Tiles {base_tile_id} and {tile_id} disagree"
                f" on {conflict_count} pixel(s); keeping values from the base tile."
            )
        _merge_tile_pixels(merged, tile_pixels)

    return merged, conflicts


def _get_point_info(
    points: gpd.GeoDataFrame,
    point_id_column: str | None,
) -> tuple[
    np.ndarray[tuple[int], Any],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.str_],
]:
    """Extract IDs, coordinates, and UTM CRS codes from a GeoDataFrame.

    Args:
        points:
            GeoDataFrame of query points.
        point_id_column:
            Column name for query point IDs, or ``None`` to use the
            index.

    Returns:
        A tuple of ``(ids, xs, ys, utm_crs_codes)``.
    """
    return (
        (
            points.index.to_numpy()
            if point_id_column is None
            else points[point_id_column].to_numpy()
        ),
        points.geometry.x.to_numpy(),
        points.geometry.y.to_numpy(),
        _compute_utm_crs(points),
    )
