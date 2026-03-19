"""aef-embeddings: AlphaEarth Foundation Satellite Embedding client."""

from importlib.metadata import version

from aef_embeddings.store import AEFSatelliteEmbeddingStore

__all__ = ["AEFSatelliteEmbeddingStore"]
__version__ = version("aef-embeddings")
