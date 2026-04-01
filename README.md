# aef-embeddings

Python client for downloading and processing [AlphaEarth Foundation
embeddings][aef-catalog] from Google Earth Engine.

## Why aef-embeddings?

Sampling embeddings from Earth Engine sounds simple until you run into
the tiling grid. The AEF dataset is split across thousands of tiles,
each in its own projection. The standard workflow is to mosaic tiles
into a common CRS before sampling, but this forces a reprojection step
that subtly degrades pixel values through resampling interpolation.

aef-embeddings avoids this entirely. Each query point is sampled in its
own local UTM zone, directly on the native pixel grid, so the values
you get back are the exact values stored in the dataset. Multi-tile
regions are merged automatically with conflict detection, and downloads
are checkpointed so interrupted jobs resume where they left off.

The package also includes built-in spatial pooling (GeM and statistical)
and quantization helpers, so you can go from raw coordinates to
analysis-ready feature vectors in a single script.

```python
import geopandas as gpd
import h5py
from aef_embeddings import AEFEmbeddingStore

store = AEFEmbeddingStore.create("your-gee-project-id")

# Sample a 5x5-pixel region (50 m x 50 m at 10 m resolution) around each point.
path = store.sample_region(
    gpd.read_file("points.gpkg"),
    point_id_column="id",
    region_size_pixels=5,
    year=2024,
)

with h5py.File(path) as f:
    # Each point has a 5x5 spatial grid of 64-band embeddings.
    values = f["values"][:]

# GeM pooling collapses the spatial grid into a single 64-d vector per point.
pooled = AEFEmbeddingStore.gem_pool(values)
```

## Features

- **Batch downloading.** Sample 64-band embeddings for arbitrary point
  locations via the Earth Engine `getPixels` API, with checkpointed
  downloads and automatic resume.
- **Quantization.** Convert between float64 and int8 for compact
  storage when full precision is not needed.
- **Spatial pooling.** Collapse multi-pixel regions into a single
  vector per point using Generalized Mean (GeM) pooling or statistical
  pooling.

## Installation

Requires **Python 3.12+**.

```bash
pip install aef-embeddings
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add aef-embeddings
```

### Developer Setup

```bash
git clone https://github.com/DimitrisMantas/aef-embeddings.git
cd aef-embeddings
uv sync --all-extras --all-groups
```

This creates a virtual environment with the correct Python version
automatically. The dev group includes
[ruff](https://docs.astral.sh/ruff/) (linter/formatter),
[ty](https://docs.astral.sh/ty/) (type checker), and
[pytest](https://docs.pytest.org/) (test runner):

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest
```

## Guide

### Downloading

Create a store, then call `sample_region` with a GeoDataFrame of query
points. Each point is reprojected to its local UTM zone, snapped to the
nearest pixel center, and a square region of `region_size_pixels` x
`region_size_pixels` is sampled around it. The region size must be a
**positive odd integer** (1, 3, 5, ...); even values are not allowed
because the region is always centered on the query point.

```python
store = AEFEmbeddingStore.create("your-gee-project-id")

path = store.sample_region(
    points,
    point_id_column="id",
    region_size_pixels=5,
    year=2024,
    # The GEE quota allows up to 40 concurrent requests.
    max_workers=40,
    # Output files and checkpoint artifacts are written here.
    output_dirpath="data",
    checkpoint_period_points=5000,
    # Enables structured logging and forces single-threaded execution.
    debug=False,
)
```

The dataset covers years **2017 through 2025** at **10 m** spatial
resolution with **64 bands**. When a region spans multiple GEE tiles,
responses are merged automatically with conflict detection.

Downloads are checkpointed periodically. If a job is interrupted, call
`sample_region` again with the same arguments and it picks up where it
left off.

### Pooling

When `region_size_pixels > 1`, each point has a spatial grid of
embeddings. Pooling collapses that grid into a single vector.

**GeM pooling** produces a 64-d vector per point. The power parameter
`p` interpolates between average pooling (`p=1`) and max pooling
(`p -> inf`); the default `p=3` is a good general-purpose choice
[[1](#references), [2](#references)]:

```python
# The result is a single 64-d vector per point.
pooled = AEFEmbeddingStore.gem_pool(values, p=3.0)
```

**Statistical pooling** concatenates per-band mean, standard deviation,
minimum, and maximum, producing a richer 256-d descriptor
[[2](#references)]:

```python
# The result is a 256-d descriptor per point.
pooled = AEFEmbeddingStore.stat_pool(values)
```

Both methods ignore NaN values automatically (masked pooling), so failed
or partially-downloaded points do not corrupt the output.

### Quantization

Convert embeddings to int8 for ~8x storage reduction, and back to
float64 when needed. The quantization scheme matches the one used for
the [GCS distribution][aef-gcs] of the dataset:

```python
# Convert from float64 to int8.
quantized = AEFEmbeddingStore.quantize(values)
# Convert back from int8 to float64.
restored = AEFEmbeddingStore.dequantize(quantized)
```

## Output Format

`sample_region` writes an HDF5 file (`.h5`) alongside a
`.h5.sha256` checksum.

### Datasets

| Name     | Shape          | Dtype   | Description                                                 |
|----------|----------------|---------|-------------------------------------------------------------|
| `values` | (N, S, S, 64)  | float64 | Embedding arrays, one per point (S = `region_size_pixels`). |
| `ids`    | (N,)           | varies  | Point identifiers from the input GeoDataFrame.              |
| `x`      | (N,)           | float64 | X-coordinate of each query point in source CRS.             |
| `y`      | (N,)           | float64 | Y-coordinate of each query point in source CRS.             |
| `status` | (N,)           | uint8   | 1 = pending, 2 = success, 3 = failure.                      |

### File Attributes

| Name                 | Type | Description                                      |
|----------------------|------|--------------------------------------------------|
| `crs`                | str  | Source CRS authority string (e.g. `EPSG:4326`).  |
| `year`               | int  | Dataset year (2017-2025).                         |
| `region_size_pixels` | int  | Patch side length in pixels.                      |
| `spatial_res_meters` | int  | Spatial resolution (always 10).                   |
| `created_at`         | str  | ISO 8601 UTC timestamp.                           |

## References

1. F. Radenovic, G. Tolias, and O. Chum, "Fine-tuning CNN Image
   Retrieval with No Human Annotation," *IEEE TPAMI*, vol. 41, no. 7,
   pp. 1655-1668, 2019.
   [arXiv:1711.02512](https://arxiv.org/abs/1711.02512)

2. I. Corley, C. Robinson, I. Becker-Reshef, and J. M. Lavista Ferres,
   "From Pixels to Patches: Pooling Strategies for Earth Embeddings,"
   *ICLR 2026 ML4RS Workshop*, 2026.
   [GitHub](https://github.com/isaaccorley/geopool)

## License

[MIT](LICENSE)

[aef-catalog]: https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL
[aef-gcs]: https://developers.google.com/earth-engine/guides/aef_on_gcs_readme
