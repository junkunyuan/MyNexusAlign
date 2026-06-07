"""Trainer registry: register built-in trainers by name on import."""

from nexus_align.registry import registry
from nexus_align.trainers.meanflow_trainer import MeanFlowTrainer

registry.register("trainer", "meanflow", MeanFlowTrainer)
