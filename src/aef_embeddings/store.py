"""AEFSatelliteEmbeddingStore: GEE client for satellite embeddings."""

import concurrent.futures
import datetime
import hashlib
import os
import pathlib
import textwrap
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
    _initialize_or_restore_checkpoint,
    _PointStatus,
    _prepare_checkpoint_dir,
)
from aef_embeddings._geo import (
    _SPATIAL_RES_METERS,
    _get_transformer,
    _half_patch_side_m,
    _resolve_utm_crss,
    _snap_to_pixel_center,
)
from aef_embeddings._logging import _configure_logging, _redirect_warnings_to_tqdm
from aef_embeddings._request import (
    _build_base_request,
    _fetch_response,
    _find_response_conflicts,
    _merge_child_into_parent_response,
)
from aef_embeddings._types import Array1D, _Response

_YEAR_MIN: Final[int] = 2017
_YEAR_MAX: Final[int] = 2025
_MAX_GEE_WORKERS: Final[int] = 40


class AEFSatelliteEmbeddingStore:
    """GEE client for the AlphaEarth Foundation Satellite Embedding Dataset."""

    _DATASET_NAME: Final[str] = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"

    def __init__(
        self,
        *,
        use_high_volume_endpoint: bool = True,
    ) -> None:
        """Constructs a store bound to an already-initialised EE session.

        Call ``AEFSatelliteEmbeddingStore.create(project_id)`` to
        authenticate, initialise, and construct in one step.

        Args:
            use_high_volume_endpoint: Whether the session was initialised
                with the high-volume endpoint. Stored for informational
                purposes only; the endpoint is set during
                ``ee.Initialize``.

        References:
            https://developers.google.com/earth-engine/guides/processing_environments#high-volume_endpoint
        """
        self._use_high_volume_endpoint = use_high_volume_endpoint
        self._dataset = ee.ImageCollection(self._DATASET_NAME)

    @classmethod
    def create(
        cls,
        project_id: str | None = None,
        *,
        use_high_volume_endpoint: bool = True,
    ) -> "AEFSatelliteEmbeddingStore":
        """Authenticates, initialises GEE, and returns a new store.

        This is the standard entry point for interactive and production
        use. Pass an already-initialised session to ``__init__`` directly
        when testing without live GEE credentials.

        Args:
            project_id: GEE project ID string. If None, the default
                project configured in the EE credentials is used.
            use_high_volume_endpoint: Whether to use the high-volume
                endpoint for server requests. Defaults to True.

        Returns:
            A new AEFSatelliteEmbeddingStore bound to the initialised
            session.

        References:
            https://developers.google.com/earth-engine/guides/processing_environments#high-volume_endpoint
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
        """Quantizes a float64 array to an int8 representation.

        The array is assumed to not contain any invalid values.

        Args:
            values: The values to process. May be any rank.

        Returns:
            The quantized values as int8, same shape as input.

        References:
            https://developers.google.com/earth-engine/guides/aef_on_gcs_readme#de-quantization
        """
        return np.clip(
            np.sqrt(np.abs(values)) * 127.5 * np.sign(values),
            a_min=-127,
            a_max=127,
        ).astype(np.int8)

    @staticmethod
    def dequantize(
        values: np.ndarray[tuple[int, ...], np.dtype[np.int8]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """De-quantizes an int8 array to a float64 representation.

        The array is assumed to not contain any invalid values.

        Args:
            values: The values to process. May be any rank.

        Returns:
            The de-quantized values as float64, same shape as input.

        References:
            https://developers.google.com/earth-engine/guides/aef_on_gcs_readme#de-quantization
        """
        data = values.astype(np.float64)
        return ((data / 127.5) ** 2) * np.sign(data)

    @staticmethod
    def gem_pool(
        values: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
        p: float = 3.0,
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Pools spatial dimensions using Generalized Mean pooling.

        GeM interpolates between average pooling (p=1) and max
        pooling (p->inf). NaN values are automatically excluded
        from the mean (masked pooling via ``np.nanmean``).

        For input of shape (N, S, S, D), returns (N, D).
        For input of shape (S, S, D), returns (D,).

        Args:
            values: Float64 array with at least 3 dimensions,
                where the last two spatial dims precede the band
                dimension.
            p: Power parameter. Must be positive. Defaults to 3.0.

        Returns:
            Array with spatial dimensions collapsed. Shape (..., D).

        Raises:
            ValueError: If ``p`` is not positive.

        References:
            https://github.com/isaaccorley/geopool
        """
        if p <= 0:
            raise ValueError(f"p must be positive, got {p}.")
        powered = np.sign(values) * np.abs(values) ** p
        pooled = np.nanmean(powered, axis=(-3, -2))
        return np.sign(pooled) * np.abs(pooled) ** (1.0 / p)

    @staticmethod
    def stat_pool(
        values: np.ndarray[tuple[int, ...], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
        """Pools spatial dimensions using statistics.

        Concatenates per-band mean, standard deviation, minimum,
        and maximum over the spatial dimensions, producing a
        descriptor 4x the original band count. NaN values are
        automatically excluded via ``np.nan*`` functions.

        For input of shape (N, S, S, D), returns (N, 4*D).
        For input of shape (S, S, D), returns (4*D,).

        Args:
            values: Float64 array with at least 3 dimensions,
                where the last two spatial dims precede the band
                dimension.

        Returns:
            Array with spatial dimensions collapsed and band
            dimension expanded 4x. Shape (..., 4*D).

        References:
            https://github.com/isaaccorley/geopool
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
        """Samples a square region around a collection of points.

        Args:
            points: The query points.
            point_id_column: The name of a column containing point IDs.
                If None, the point index is used.
            region_size_pixels: The side length of the sampled region in
                pixels. Must be a positive odd integer. The spatial
                resolution of the dataset is 10 m.
            year: The dataset year. Must be between ``_YEAR_MIN`` and
                ``_YEAR_MAX`` inclusive.
            max_workers: The maximum number of worker threads for parallel
                requests. Cannot exceed ``_MAX_GEE_WORKERS`` (40) unless
                the project is specifically configured. Defaults to the
                ``ThreadPoolExecutor`` default.
            output_dirpath: Path to a directory for saving output and
                checkpoint data. Defaults to ``"data"``.
            checkpoint_period_points: Number of processed points between
                checkpoint saves. If larger than the total number of
                query points, a single checkpoint is saved on completion.
                Defaults to 5000.
            debug: If True, enables structured logging to stdout and a
                JSONL file and forces single-threaded execution. Defaults
                to False.

        Returns:
            Path to the HDF5 output file containing datasets
            ``values``, ``ids``, ``x``, ``y``, and ``status``.

        Raises:
            ValueError: If ``year`` is outside [``_YEAR_MIN``,
                ``_YEAR_MAX``], if ``max_workers`` exceeds
                ``_MAX_GEE_WORKERS``, or if ``region_size_pixels`` is not
                a positive odd integer.

        References:
            https://developers.google.com/earth-engine/guides/usage#adjustable_quota_limits
        """
        if not (_YEAR_MIN <= year <= _YEAR_MAX):
            raise ValueError(
                f"year must be between {_YEAR_MIN} and {_YEAR_MAX}, got {year}."
            )
        if max_workers is not None and max_workers > _MAX_GEE_WORKERS:
            raise ValueError(
                f"max_workers cannot exceed {_MAX_GEE_WORKERS} without"
                f" special project configuration, got {max_workers}."
            )
        if region_size_pixels < 1 or region_size_pixels % 2 == 0:
            raise ValueError(
                f"region_size_pixels must be a positive odd integer,"
                f" got {region_size_pixels}."
            )

        ids, xs, ys, utm_crs_codes = _get_point_info(points, point_id_column)

        crs = points.crs.to_string()
        request_checksum = _compute_request_checksum(
            ids, xs, ys, year, region_size_pixels, crs
        )

        output_dirpath = pathlib.Path(output_dirpath)
        memmap_path, status_path, request_checksum_path = _prepare_checkpoint_dir(
            output_dirpath
        )

        output, status = _initialize_or_restore_checkpoint(
            len(points),
            region_size_pixels,
            memmap_path,
            status_path,
            request_checksum_path,
            request_checksum,
        )
        output_path = output_dirpath / "embeddings.h5"

        remaining = np.where(status != _PointStatus.COMPLETED)[0]
        if len(remaining) == 0:
            return self._write_hdf5(
                output_path,
                output,
                ids,
                xs,
                ys,
                status,
                crs,
                year,
                region_size_pixels,
            )

        _configure_logging(output_dirpath, console=debug)

        if debug:
            if max_workers is not None and max_workers > 1:
                warnings.warn(
                    "Debug mode forces single-threaded execution for"
                    " deterministic log ordering. The maximum number"
                    " of worker threads will be set to 1.",
                    UserWarning,
                    stacklevel=2,
                )
            max_workers = 1

        num_completed = 0
        with (
            _redirect_warnings_to_tqdm(),
            concurrent.futures.ThreadPoolExecutor(max_workers) as executor,
        ):
            futures = {
                executor.submit(
                    self._sample_point_region,
                    xs[request_index],
                    ys[request_index],
                    crs,
                    # GEE and PROJ don't work with NumPy strings.
                    str(utm_crs_codes[request_index]),
                    region_size_pixels,
                    year,
                ): request_index
                for request_index in remaining
            }

            iterator = concurrent.futures.as_completed(futures)
            iterator = tqdm.tqdm(iterator, total=len(futures))
            for future in iterator:
                request_index = futures[future]
                try:
                    output[request_index] = future.result()
                    status[request_index] = _PointStatus.COMPLETED
                except Exception as e:
                    loguru.logger.warning(
                        "Embedding download failed for point"
                        f" {request_index} ({type(e).__name__}):"
                        f"\n{textwrap.indent(str(e), prefix='\t')}"
                    )
                    output[request_index] = np.nan
                    status[request_index] = _PointStatus.FAILED

                num_completed += 1
                if num_completed % checkpoint_period_points == 0:
                    output.flush()
                    np.save(status_path, status)

        output.flush()
        np.save(status_path, status)

        return self._write_hdf5(
            output_path,
            output,
            ids,
            xs,
            ys,
            status,
            crs,
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
        crs: str,
        year: int,
        region_size_pixels: int,
    ) -> pathlib.Path:
        """Writes a self-describing HDF5 file with all outputs.

        Args:
            output_path: Destination HDF5 file path.
            output: Float64 output array.
            ids: 1-D point ID array.
            xs: 1-D source CRS x-coordinates.
            ys: 1-D source CRS y-coordinates.
            status: 1-D uint8 status array.
            crs: Source CRS string.
            year: Dataset year.
            region_size_pixels: Patch side length in pixels.

        Returns:
            The output path.
        """
        with h5py.File(output_path, "w") as f:
            f.create_dataset("values", data=output)
            f.create_dataset("ids", data=ids)
            f.create_dataset("x", data=xs)
            f.create_dataset("y", data=ys)
            f.create_dataset("status", data=status)
            f.attrs["crs"] = crs
            f.attrs["year"] = year
            f.attrs["region_size_pixels"] = region_size_pixels
            f.attrs["spatial_res_meters"] = _SPATIAL_RES_METERS
            f.attrs["created_at"] = datetime.datetime.now(datetime.UTC).isoformat()

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
        """Returns GEE asset IDs for tiles intersecting the patch footprint.

        Args:
            x: Snapped easting of the centre pixel (meters, UTM).
            y: Snapped northing of the centre pixel (meters, UTM).
            utm_crs: EPSG code string for the local UTM CRS.
            region_size_pixels: Side length of the requested patch in
                pixels.
            year: Dataset year.

        Returns:
            List of ``system:id`` strings for intersecting tiles.

        Raises:
            RuntimeError: If the EE collection returns no tile IDs for
                the requested location and year (``getInfo`` returns
                None).
        """
        point = ee.Geometry.Point((x, y), proj=utm_crs)
        bounds = (
            point
            if region_size_pixels == 1
            else point.buffer(_half_patch_side_m(region_size_pixels)).bounds()
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
        x: float,
        y: float,
        crs: str,
        utm_crs: str,
        region_size_pixels: int,
        year: int,
    ) -> _Response:
        """Samples a patch around a single point from GEE.

        Args:
            x: Input x-coordinate in the source CRS.
            y: Input y-coordinate in the source CRS.
            crs: Source CRS identifier string.
            utm_crs: Local UTM CRS identifier string.
            region_size_pixels: Side length of the sampled patch in
                pixels.
            year: Dataset year.

        Returns:
            Float64 array of shape (S, S, 64) for the sampled patch.

        Raises:
            RuntimeError: If the point does not intersect any dataset
                tiles.
        """
        log = loguru.logger.bind(
            source_crs=crs,
            utm_crs=utm_crs,
            dataset_year=year,
            patch_size_px=region_size_pixels,
            source_x=x,
            source_y=y,
        )

        utm_x, utm_y = _get_transformer(crs, utm_crs).transform(x, y)
        log = log.bind(utm_easting=utm_x, utm_northing=utm_y)
        log.debug("Reprojected query point from source CRS to UTM.")

        snapped_x, snapped_y = _snap_to_pixel_center(utm_x, utm_y)
        log = log.bind(snapped_easting=snapped_x, snapped_northing=snapped_y)
        log.debug("Snapped UTM coordinates to nearest pixel centre.")

        request = _build_base_request(snapped_x, snapped_y, region_size_pixels, utm_crs)

        tile_ids = self._get_intersecting_tile_ids(
            snapped_x, snapped_y, utm_crs, region_size_pixels, year
        )
        log = log.bind(tile_count=len(tile_ids), tile_asset_ids=tile_ids)
        if len(tile_ids) != 1:
            log.debug(
                f"Point intersects {len(tile_ids)} dataset tiles"
                f" (multi-tile merge required)."
            )

        parent_response: _Response | None = None
        parent_tile_id: str | None = None
        for tile_id in tile_ids:
            request["assetId"] = tile_id
            log.bind(gee_request=request).debug(
                f"Fetching pixel data from GEE tile {tile_id}."
            )

            child_response = _fetch_response(request)

            if parent_response is None:
                parent_response = child_response
                parent_tile_id = tile_id
            else:
                conflict_mask = _find_response_conflicts(
                    parent_response, child_response
                )
                if conflict_mask is not None:
                    log.bind(
                        existing_tile_id=parent_tile_id,
                        overlapping_tile_id=tile_id,
                        conflicting_pixel_count=int(conflict_mask[..., 0].sum()),
                        conflicting_pixel_indices=np.argwhere(
                            conflict_mask[..., 0]
                        ).tolist(),
                        existing_values_int8=self.quantize(
                            parent_response[conflict_mask]
                        ).tolist(),
                        overlapping_values_int8=self.quantize(
                            child_response[conflict_mask]
                        ).tolist(),
                    ).warning(
                        f"Overlapping tiles {parent_tile_id} and {tile_id}"
                        f" disagree on {int(conflict_mask[..., 0].sum())}"
                        f" pixel(s); keeping values from the first tile."
                    )
                _merge_child_into_parent_response(parent_response, child_response)

        if parent_response is None:
            raise RuntimeError(
                "No dataset tiles found for this point — the location may be"
                " outside the spatial coverage of the dataset for the"
                f" requested year ({year})."
            )

        valid_px = int(np.sum(~np.isnan(parent_response[..., 0])))
        total_px = parent_response.shape[0] * parent_response.shape[1]
        log.bind(
            response_checksum_md5=hashlib.md5(
                np.ascontiguousarray(parent_response).tobytes()
            ).hexdigest(),
            valid_pixel_count=valid_px,
            missing_pixel_count=total_px - valid_px,
        ).debug(f"Point complete: {valid_px}/{total_px} pixels valid.")

        return parent_response


def _get_point_info(
    points: gpd.GeoDataFrame,
    point_id_column: str | None,
) -> tuple[
    np.ndarray[tuple[int], Any],
    Array1D[np.float64],
    Array1D[np.float64],
    Array1D[np.str_],
]:
    """Extracts IDs, coordinates, and UTM CRS codes from a GeoDataFrame.

    Args:
        points: GeoDataFrame of query points.
        point_id_column: Column name for point IDs, or None to use the
            index.

    Returns:
        A tuple of (ids, xs, ys, utm_crs_codes).
    """
    return (
        (
            points.index.to_numpy()
            if point_id_column is None
            else points[point_id_column].to_numpy()
        ),
        points.geometry.x.to_numpy(),
        points.geometry.y.to_numpy(),
        _resolve_utm_crss(points),
    )
