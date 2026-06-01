"""Dataset registry: register built-in datasets by name on import."""

from nexus_align.registry import registry
from nexus_align.datasets.lmdb_latents import LMDBLatentsDataset

registry.register("dataset", "lmdb_latents", LMDBLatentsDataset)
