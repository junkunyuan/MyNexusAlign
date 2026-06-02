"""Dataset registry: register built-in datasets by name on import."""

from nexus_align.registry import registry
from nexus_align.datasets.lmdb_latents import LMDBLatentsDataset
from nexus_align.datasets.imagenet_1k import ImageNet1K

registry.register("dataset", "lmdb_latents", LMDBLatentsDataset)
registry.register("dataset", "imagenet-1k", ImageNet1K)
