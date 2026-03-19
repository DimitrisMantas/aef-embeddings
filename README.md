# aef-embeddings

Python client for downloading and processing
[AlphaEarth Foundation satellite embeddings](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL)
from Google Earth Engine.

## Features

- Batch download of 64-band satellite embeddings for arbitrary point
  locations
- Automatic UTM reprojection and pixel-center snapping
- Multi-tile merging with conflict detection
- Checkpointed downloads with automatic resume
- HDF5 output with SHA-256 integrity verification
- Quantization (float64 <-> int8) and spatial pooling (GeM, statistical)

## Installation

```bash
uv add aef-embeddings
```

## Quick Start

```python
import geopandas as gpd
import h5py

from aef_embeddings import AEFSatelliteEmbeddingStore

# Authenticate and initialize a GEE session.
store = AEFSatelliteEmbeddingStore.create("your-gee-project-id")

# Load query points and download embeddings.
points = gpd.read_file("expected.gpkg")
path = store.sample_region(
    points,
    point_id_column="id",
    region_size_pixels=1,
    year=2024,
)

# Read the output.
with h5py.File(path) as f:
    values = f["values"][:]

# Pool spatial dimensions into a single vector per point.
pooled = AEFSatelliteEmbeddingStore.gem_pool(values)
```

## License

[MIT](LICENSE)
