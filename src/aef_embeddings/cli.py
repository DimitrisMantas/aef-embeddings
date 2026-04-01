"""Command-line interface."""

import pathlib
import sys

import click
import geopandas as gpd
import h5py
import numpy as np

from aef_embeddings.store import AEFEmbeddingStore


@click.group()
@click.version_option(package_name="aef-embeddings")
def cli() -> None:
    """Download and process AlphaEarth Foundation embeddings."""


@cli.command()
@click.argument("points", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--id-column",
    default=None,
    help="Column name for point IDs. Uses the DataFrame index if omitted.",
)
@click.option(
    "--region-size",
    required=True,
    type=int,
    help="Side length of the query region in pixels. Must be a positive odd integer.",
)
@click.option(
    "--year",
    required=True,
    type=click.IntRange(2017, 2025),
    help="Dataset year (2017-2025).",
)
@click.option(
    "--project-id",
    default=None,
    help="GEE project ID. Uses the default from EE credentials if omitted.",
)
@click.option(
    "--max-workers",
    default=None,
    type=click.IntRange(1, 40),
    help="Maximum worker threads for parallel requests (1-40).",
)
@click.option(
    "--output-dir",
    default="data",
    type=click.Path(file_okay=False),
    help="Directory for output and checkpoint files.",
    show_default=True,
)
@click.option(
    "--checkpoint-period",
    default=5000,
    type=int,
    help="Number of points between checkpoint saves.",
    show_default=True,
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable structured logging and force single-threaded execution.",
)
def download(
    points: str,
    id_column: str | None,
    region_size: int,
    year: int,
    project_id: str | None,
    max_workers: int | None,
    output_dir: str,
    checkpoint_period: int,
    debug: bool,
) -> None:
    """Download embedding patches for every point in a vector file.

    POINTS is the path to a GeoPackage, Shapefile, or any other vector
    format supported by GeoPandas.
    """
    gdf = gpd.read_file(points)
    store = AEFEmbeddingStore.create(project_id)
    output_path = store.sample_region(
        gdf,
        point_id_column=id_column,
        region_size_pixels=region_size,
        year=year,
        max_workers=max_workers,
        output_dirpath=output_dir,
        checkpoint_period_points=checkpoint_period,
        debug=debug,
    )
    click.echo(f"Output: {output_path}")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(dir_okay=False),
    help="Output HDF5 path. Defaults to INPUT_FILE with a '.quantized.h5' suffix.",
)
def quantize(input_file: str, output: str | None) -> None:
    """Quantize embeddings from float64 to int8.

    Reads the 'values' dataset from INPUT_FILE, applies signed
    square-root quantization, and writes all datasets and attributes to
    the output file with the quantized values.
    """
    input_path = pathlib.Path(input_file)
    output_path = (
        pathlib.Path(output) if output else input_path.with_suffix(".quantized.h5")
    )
    _transform_values(input_path, output_path, AEFEmbeddingStore.quantize)
    click.echo(f"Output: {output_path}")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(dir_okay=False),
    help=("Output HDF5 path. Defaults to INPUT_FILE with a '.dequantized.h5' suffix."),
)
def dequantize(input_file: str, output: str | None) -> None:
    """Dequantize embeddings from int8 back to float64.

    Reads the 'values' dataset from INPUT_FILE, applies inverse
    square-root dequantization, and writes all datasets and attributes
    to the output file with the restored values.
    """
    input_path = pathlib.Path(input_file)
    output_path = (
        pathlib.Path(output) if output else input_path.with_suffix(".dequantized.h5")
    )
    _transform_values(input_path, output_path, AEFEmbeddingStore.dequantize)
    click.echo(f"Output: {output_path}")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--method",
    required=True,
    type=click.Choice(["gem", "stat"], case_sensitive=False),
    help="Pooling method: 'gem' for Generalized Mean, 'stat' for statistical.",
)
@click.option(
    "-p",
    "--power",
    default=3.0,
    type=float,
    help="Power parameter for GeM pooling. Ignored for 'stat'.",
    show_default=True,
)
@click.option(
    "-o",
    "--output",
    default=None,
    type=click.Path(dir_okay=False),
    help="Output HDF5 path. Defaults to INPUT_FILE with a '.pooled.h5' suffix.",
)
def pool(input_file: str, method: str, power: float, output: str | None) -> None:
    """Pool spatial dimensions into a single vector per point.

    Reads the 'values' dataset from INPUT_FILE, collapses the spatial
    grid using the selected pooling method, and writes all datasets and
    attributes to the output file with the pooled values.
    """
    input_path = pathlib.Path(input_file)
    output_path = (
        pathlib.Path(output) if output else input_path.with_suffix(".pooled.h5")
    )

    if method == "gem":

        def transform(v: np.ndarray) -> np.ndarray:
            return AEFEmbeddingStore.gem_pool(v, p=power)
    else:
        transform = AEFEmbeddingStore.stat_pool

    _transform_values(input_path, output_path, transform)
    click.echo(f"Output: {output_path}")


def _transform_values(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    fn: object,
) -> None:
    """Read an HDF5 file, apply a transform to 'values', and write the result.

    All other datasets and file-level attributes are copied unchanged.

    Args:
        input_path:
            Source HDF5 file path.
        output_path:
            Destination HDF5 file path.
        fn:
            Callable that accepts a NumPy array and returns a
            transformed NumPy array.
    """
    if output_path == input_path:
        click.echo("Error: output path must differ from input path.", err=True)
        sys.exit(1)

    with h5py.File(input_path, "r") as src:
        if "values" not in src:
            click.echo(
                f"Error: '{input_path}' does not contain a 'values' dataset.",
                err=True,
            )
            sys.exit(1)

        values = src["values"][:]
        transformed = fn(values)

        with h5py.File(output_path, "w") as dst:
            dst.create_dataset("values", data=transformed)

            # Copy all non-values datasets.
            for name in src:
                if name != "values":
                    dst.create_dataset(name, data=src[name][:])

            # Copy all file-level attributes.
            for key, value in src.attrs.items():
                dst.attrs[key] = value
