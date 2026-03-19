"""Sample satellite embeddings for a set of LUCAS survey points.

Authenticates with Google Earth Engine, downloads per-point embeddings
for a single year, and prints summary statistics of the pooled result.

Usage:
    uv run python example.py
"""

import os

import dotenv
import geopandas as gpd
import h5py
import numpy as np

from aef_embeddings import AEFSatelliteEmbeddingStore


def main() -> None:
    dotenv.load_dotenv()

    points = gpd.read_file("../../tests/expected.gpkg", use_arrow=True)
    store = AEFSatelliteEmbeddingStore.create(os.environ["GEE_PROJECT_ID"])

    path = store.sample_region(
        points,
        point_id_column="id",
        region_size_pixels=1,
        year=2024,
    )

    with h5py.File(path) as f:
        ids = f["ids"][:]
        values = f["values"][:]

    pooled = AEFSatelliteEmbeddingStore.gem_pool(values)

    print(f"Points:     {len(ids)}")
    print(f"Embeddings: {values.shape}")
    print(f"Pooled:     {pooled.shape}")
    print(f"Mean norm:  {np.mean(np.linalg.norm(pooled, axis=-1)):.4f}")
    print(f"Output:     {path}")


if __name__ == "__main__":
    main()
