import pathlib

import geopandas as gpd
import h5py
import numpy as np
import pytest

from aef_embeddings import AEFSatelliteEmbeddingStore

_FIXTURES = pathlib.Path(__file__).parent / "res"

# Indices where quantized store output is known to diverge from
# the precomputed GeoPackage values.
_KNOWN_MISMATCHES = np.array([1180, 33679, 43008, 66185, 69679])


@pytest.fixture
def points() -> gpd.GeoDataFrame:
    return gpd.read_file(_FIXTURES / "points.gpkg", use_arrow=True)


@pytest.fixture
def received_output(points: gpd.GeoDataFrame) -> np.ndarray:
    with h5py.File(_FIXTURES / "embeddings.h5", "r") as f:
        return f["values"][:]


def test_quantized_output_matches_precomputed(
    points: gpd.GeoDataFrame,
    received_output: np.ndarray,
) -> None:
    """Quantized store output matches GeoPackage columns A00-A63.

    Exactly 5 known disagreements are expected at specific indices
    due to GEE data version differences.
    """
    received = AEFSatelliteEmbeddingStore.quantize(received_output).squeeze()
    expected = points[[f"A{i:02d}" for i in range(64)]].to_numpy()
    matches = np.all(received == expected, axis=1)
    mismatch_indices = np.flatnonzero(~matches)

    assert len(mismatch_indices) == len(_KNOWN_MISMATCHES)
    np.testing.assert_array_equal(mismatch_indices, _KNOWN_MISMATCHES)


def test_gem_pool_shape():
    x = np.random.default_rng(42).standard_normal((10, 5, 5, 64))
    result = AEFSatelliteEmbeddingStore.gem_pool(x)
    assert result.shape == (10, 64)


def test_stat_pool_shape():
    x = np.random.default_rng(42).standard_normal((10, 5, 5, 64))
    result = AEFSatelliteEmbeddingStore.stat_pool(x)
    assert result.shape == (10, 256)


def test_gem_pool_nan_masking():
    x = np.ones((1, 3, 3, 2))
    x[0, 1, 1, :] = np.nan
    result = AEFSatelliteEmbeddingStore.gem_pool(x)
    assert not np.any(np.isnan(result))


def test_quantize_dequantize_roundtrip():
    rng = np.random.default_rng(42)
    original = rng.uniform(-1, 1, size=(100, 64))
    roundtripped = AEFSatelliteEmbeddingStore.dequantize(
        AEFSatelliteEmbeddingStore.quantize(original)
    )
    np.testing.assert_allclose(original, roundtripped, atol=0.02)
