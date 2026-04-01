"""AlphaEarth Foundation embedding client for Google Earth Engine."""

from importlib.metadata import version

from aef_embeddings.store import AEFEmbeddingStore

__all__ = ["AEFEmbeddingStore"]
__version__ = version("aef-embeddings")
