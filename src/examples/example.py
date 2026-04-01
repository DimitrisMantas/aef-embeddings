"""Sample AEF embeddings for a set of LUCAS survey points.

Authenticates with Google Earth Engine, downloads 5x5-pixel regions around each point,
and demonstrates spatial pooling to produce a single embedding vector per point.

Usage::

    uv run python example.py
"""

import os
import pathlib

import dotenv
import geopandas as gpd
import h5py
import numpy as np

from aef_embeddings import AEFEmbeddingStore

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> None:
    dotenv.load_dotenv()

    points = gpd.read_file(_PROJECT_ROOT / "tests" / "points.gpkg", use_arrow=True)
    store = AEFEmbeddingStore.create(os.environ["GEE_PROJECT_ID"])

    # Download a 5x5-pixel region (50 m x 50 m at 10 m resolution) around each point,
    # yielding 25 embedding vectors per point.
    path = store.sample_region(
        points,
        point_id_column="id",
        region_size_pixels=5,
        year=2024,
    )

    with h5py.File(path) as f:
        ids = f["ids"][:]
        values = f["values"][:]

    print(f"Points:     {len(ids)}")
    print(f"Embeddings: {values.shape}")

    # GeM pooling collapses the 5x5 spatial grid into a single 64D vector per point.
    gem = AEFEmbeddingStore.gem_pool(values)
    print(f"GeM pool:   {gem.shape}")
    print(f"Mean norm:  {np.mean(np.linalg.norm(gem, axis=-1)):.4f}")

    # Statistical pooling concatenates per-band mean, standard deviation, minimum, and
    # maximum, producing a 256D descriptor per point.
    stat = AEFEmbeddingStore.stat_pool(values)
    print(f"Stat pool:  {stat.shape}")

    print(f"Output:     {path}")


if __name__ == "__main__":
    main()
