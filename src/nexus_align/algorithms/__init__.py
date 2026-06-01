"""Algorithm registry: register built-in algorithms by name on import."""

from nexus_align.registry import registry
from nexus_align.algorithms.meanflow import SILoss

registry.register("algorithm", "meanflow", SILoss)
